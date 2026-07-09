#!/usr/bin/env python3
"""
EC2 extraction daemon — polls S3 for queued batches and processes them.

Architecture (producer/consumer via S3, not SSH-tunneled):

    GitHub Actions (producer)
        1. fetches URLs
        2. writes s3://phishnet-data/queue/batch_<TS>.csv
        3. writes s3://phishnet-data/queue/url_features_<TS>.csv
        4. writes s3://phishnet-data/queue/READY_<TS>.marker    (last write)
        5. starts EC2 if stopped
        6. exits (job takes ~2 min total)

    EC2 daemon (this script, systemd-managed)
        loop:
            list s3://phishnet-data/queue/READY_*.marker
            for each marker:
                extract features + append to masters (existing logic)
                write s3://phishnet-data/pipeline_state/<TS>/done.json
                delete marker + queue files
            if queue empty for IDLE_SHUTDOWN_MIN minutes: shutdown -h now

    GitHub Actions (verifier, hourly)
        checks: masters recent AND queue empty AND either EC2 running
        with backlog OR EC2 stopped with nothing to do.

Design goals:
- No long-lived SSH session (that's what caused the SSH-drop failures).
- Instance stops itself when idle → cost stays near current ~$1-3/month.
- Every batch's status is written to S3 so we can inspect without SSH-ing.
- Robust to crashes: systemd restarts us; a batch we didn't finish stays
  in the queue and gets picked up on the next iteration.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Extraction logic is factored out so we can call it as a function here.
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_vm_features_aws import run_batch_in_scratch  # noqa: E402


S3_BUCKET = os.environ.get("S3_BUCKET", "phishnet-data")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Config — tunable via env for one-off runs but sensible defaults
POLL_INTERVAL_SEC = int(os.environ.get("PHISHNET_POLL_INTERVAL_SEC", "60"))
IDLE_SHUTDOWN_MIN = int(os.environ.get("PHISHNET_IDLE_SHUTDOWN_MIN", "10"))
DRY_RUN = os.environ.get("PHISHNET_DRY_RUN", "0") == "1"

# S3 key layout
QUEUE_PREFIX = "queue/"
READY_PREFIX = "queue/READY_"
STATE_PREFIX = "pipeline_state/"

# Systemd/journal-friendly logging: stdout, no color, structured-ish.
logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("phishnet-daemon")

_stop_requested = False


def _handle_sig(signum, _frame):
    global _stop_requested
    log.info("Received signal %s — will finish current batch then exit.", signum)
    _stop_requested = True


signal.signal(signal.SIGTERM, _handle_sig)
signal.signal(signal.SIGINT, _handle_sig)


def _s3():
    return boto3.client("s3", region_name=AWS_REGION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_pending_batches(s3) -> list[str]:
    """Return batch_dates for every READY_<batch_date>.marker in the queue.

    Sorted chronologically so FIFO ordering holds.
    """
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=READY_PREFIX)
    contents = resp.get("Contents", [])
    batch_dates = []
    for obj in contents:
        key = obj["Key"]
        # queue/READY_<batch_date>.marker  →  <batch_date>
        base = key[len(READY_PREFIX):]
        if base.endswith(".marker"):
            batch_dates.append(base[: -len(".marker")])
    return sorted(batch_dates)


def _write_state(s3, batch_date: str, payload: dict) -> None:
    """Write a small JSON state file so the outside world can watch progress."""
    key = f"{STATE_PREFIX}{batch_date}/status.json"
    body = json.dumps(payload, indent=2).encode("utf-8")
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except ClientError as e:
        # State writes are best-effort; the daemon shouldn't die if S3 blips.
        log.warning("Failed to write state %s: %s", key, e)


def _write_done(s3, batch_date: str, ok: bool, detail: dict) -> None:
    key = f"{STATE_PREFIX}{batch_date}/done.json"
    payload = {
        "ok": ok,
        "finished_at": _now_iso(),
        "batch_date": batch_date,
        **detail,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except ClientError as e:
        log.error("Failed to write done marker for %s: %s", batch_date, e)


def _delete_queue_files(s3, batch_date: str) -> None:
    """Delete the batch's queue files once accumulation succeeded.

    We keep the pipeline_state/<batch>/done.json around for observability; a
    separate S3 lifecycle rule will expire those after 7 days.
    """
    keys = [
        f"{QUEUE_PREFIX}batch_{batch_date}.csv",
        f"{QUEUE_PREFIX}url_features_{batch_date}.csv",
        f"{READY_PREFIX}{batch_date}.marker",
    ]
    for k in keys:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=k)
        except ClientError as e:
            log.warning("Failed to delete %s: %s", k, e)


def _process_one(batch_date: str) -> None:
    s3 = _s3()
    log.info("Processing batch %s", batch_date)
    _write_state(
        s3,
        batch_date,
        {"batch_date": batch_date, "started_at": _now_iso(), "phase": "running"},
    )
    try:
        # This is the same function the SSH pipeline used to call. It downloads
        # the batch, extracts DNS/WHOIS, appends to masters in S3, uploads.
        # run_batch_in_scratch() also guarantees /tmp cleanup for us.
        run_batch_in_scratch(batch_date)
    except Exception as e:
        log.exception("Batch %s failed: %s", batch_date, e)
        _write_done(s3, batch_date, ok=False, detail={"error": repr(e)})
        # Leave the queue files so a fix + restart can retry; but move the
        # marker aside so we don't hot-loop on the same broken batch.
        try:
            marker_key = f"{READY_PREFIX}{batch_date}.marker"
            failed_key = f"queue/FAILED_{batch_date}.marker"
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": marker_key},
                Key=failed_key,
            )
            s3.delete_object(Bucket=S3_BUCKET, Key=marker_key)
        except ClientError as ce:
            log.warning("Could not mark batch %s as FAILED: %s", batch_date, ce)
        return

    _write_done(s3, batch_date, ok=True, detail={"phase": "accumulated"})
    _delete_queue_files(s3, batch_date)
    log.info("Batch %s done", batch_date)


def _shutdown_ec2():
    if DRY_RUN:
        log.info("[DRY_RUN] Would shutdown -h now here.")
        return
    log.info("Idle → shutting down instance now.")
    # `sudo` on a systemd-managed daemon: we'll configure sudoers so the
    # daemon user can run `/sbin/shutdown -h now` without a password.
    subprocess.run(
        ["sudo", "/sbin/shutdown", "-h", "now"],
        check=False,
    )


def main() -> int:
    log.info(
        "phishnet-daemon starting (bucket=%s poll=%ds idle_shutdown=%dm dry_run=%s)",
        S3_BUCKET,
        POLL_INTERVAL_SEC,
        IDLE_SHUTDOWN_MIN,
        DRY_RUN,
    )
    s3 = _s3()
    idle_since = time.time()

    while not _stop_requested:
        try:
            pending = _list_pending_batches(s3)
        except ClientError as e:
            # Network hiccups → try again next poll; don't crash the daemon.
            log.warning("S3 list failed: %s — retrying next poll.", e)
            pending = []

        if pending:
            log.info("Found %d pending batch(es): %s", len(pending), pending)
            idle_since = time.time()
            for batch_date in pending:
                if _stop_requested:
                    break
                _process_one(batch_date)
        else:
            idle_minutes = (time.time() - idle_since) / 60.0
            log.info("Queue empty for %.1f min.", idle_minutes)
            if idle_minutes >= IDLE_SHUTDOWN_MIN:
                _shutdown_ec2()
                # If we get past the shutdown call (e.g. in DRY_RUN), avoid
                # tight looping — sleep the poll interval before checking again.

        for _ in range(POLL_INTERVAL_SEC):
            if _stop_requested:
                break
            time.sleep(1)

    log.info("phishnet-daemon exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
