#!/usr/bin/env python3
"""
Extract DNS and WHOIS features on AWS EC2 and accumulate to master dataset in S3.

Includes:
- Checkpointing every 50 URLs (survives SSH drops / crashes)
- Progress heartbeat every 30s to keep SSH alive
- Per-URL timeout to prevent hanging on unresponsive domains
"""

import sys
import os
import time
import json
import signal
import threading
import multiprocessing
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_whois_features, extract_domain_from_url

S3_BUCKET = "phishnet-data"
AWS_REGION = "us-east-1"
CHECKPOINT_EVERY = 50  # Save progress every N URLs
HEARTBEAT_INTERVAL = 30  # Print heartbeat every N seconds to keep SSH alive
PER_URL_TIMEOUT = 45  # Max seconds per URL - reduced to kill faster


def _whois_for_url(url):
    """Extract WHOIS features directly (no cache read/write per URL)."""
    domain = extract_domain_from_url(url) or url
    feats = extract_whois_features(domain, mode="single")
    feats.pop("domain", None)
    return feats


class Heartbeat:
    """Background thread that prints periodic heartbeat to keep SSH alive."""
    def __init__(self, interval=HEARTBEAT_INTERVAL):
        self.interval = interval
        self.status = "initializing"
        self.count = 0
        self.total = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def update(self, status, count=0, total=0):
        self.status = status
        self.count = count
        self.total = total

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self._start_time))
            print(f"💓 [{elapsed}] {self.status} ({self.count}/{self.total})", flush=True)

    def __enter__(self):
        self._start_time = time.time()
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def _run_in_process(func, url, timeout=PER_URL_TIMEOUT):
    """Run extraction in a subprocess that can be killed on timeout."""
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()

    def target(queue, fn, u):
        try:
            result = fn(u)
            queue.put(("ok", result))
        except Exception as e:
            queue.put(("error", str(e)))

    p = ctx.Process(target=target, args=(q, func, url))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.kill()
        p.join(2)
        print(f"  ⏰ TIMEOUT after {timeout}s for {url[:50]} (process killed)", flush=True)
        return {}

    if q.empty():
        print(f"  ❌ Process died for {url[:50]}", flush=True)
        return {}

    status, data = q.get_nowait()
    if status == "error":
        print(f"  ❌ Error for {url[:50]}: {data}", flush=True)
        return {}

    return data or {}


def extract_and_accumulate(batch_date: str):
    """
    Extract DNS/WHOIS features and accumulate to master dataset in S3.
    Includes checkpointing to survive SSH disconnections.
    """

    s3 = boto3.client('s3', region_name=AWS_REGION)

    # Setup directories. Because we chdir into a fresh tempdir at __main__,
    # the whois.py / dns_ipwhois.py libraries' import-time os.makedirs calls
    # created these dirs in the ORIGINAL cwd, not this tempdir. Recreate them
    # here so the libraries can write their cache/latency-log files.
    os.makedirs("vm_data/url_queue", exist_ok=True)
    os.makedirs("vm_data/incremental", exist_ok=True)
    os.makedirs("vm_data/master", exist_ok=True)
    os.makedirs("vm_data/checkpoints", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    batch_name = f"batch_{batch_date}.csv"
    url_features_name = f"url_features_{batch_date}.csv"
    checkpoint_file = f"vm_data/checkpoints/checkpoint_{batch_date}.json"
    dns_checkpoint = f"vm_data/checkpoints/dns_{batch_date}.csv"
    whois_checkpoint = f"vm_data/checkpoints/whois_{batch_date}.csv"

    # Step 1: Download batch files from S3
    print("=" * 60, flush=True)
    print("STEP 1: Download batch data from S3", flush=True)
    print("=" * 60, flush=True)

    s3.download_file(S3_BUCKET, f"queue/{batch_name}", f"vm_data/url_queue/{batch_name}")
    s3.download_file(S3_BUCKET, f"queue/{url_features_name}", f"vm_data/url_queue/{url_features_name}")

    batch_file = f"vm_data/url_queue/{batch_name}"
    url_features_file = f"vm_data/url_queue/{url_features_name}"

    df_batch = pd.read_csv(batch_file)
    df_url_features = pd.read_csv(url_features_file)

    total_urls = len(df_batch)
    print(f"✅ Downloaded {total_urls} URLs", flush=True)
    print(f"✅ Downloaded URL features for {len(df_url_features)} URLs", flush=True)

    # Load checkpoint if exists (resume from crash)
    dns_start_idx = 0
    whois_start_idx = 0
    dns_features = []
    whois_features = []

    # A checkpoint is only valid if its cached URLs match THIS batch's URLs.
    # Otherwise (stale checkpoint from a same-named earlier batch) we'd resume
    # as "done" and re-upload old data without extracting the new URLs.
    def _checkpoint_matches_batch(ckpt_csv, batch_urls):
        if not os.path.exists(ckpt_csv):
            return False
        try:
            cached = pd.read_csv(ckpt_csv)
            if 'url' not in cached.columns:
                return False
            # Cached URLs must be a prefix of the current batch (resume scenario)
            cached_urls = list(cached['url'])
            return cached_urls == batch_urls[:len(cached_urls)]
        except Exception:
            return False

    batch_urls = list(df_batch['url'])

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            cp = json.load(f)
        cand_dns = cp.get("dns_completed", 0)
        cand_whois = cp.get("whois_completed", 0)

        dns_valid = cand_dns == 0 or _checkpoint_matches_batch(dns_checkpoint, batch_urls)
        whois_valid = cand_whois == 0 or _checkpoint_matches_batch(whois_checkpoint, batch_urls)

        if dns_valid and whois_valid:
            dns_start_idx = cand_dns
            whois_start_idx = cand_whois
            print(f"📋 Resuming from checkpoint: DNS={dns_start_idx}, WHOIS={whois_start_idx}", flush=True)
            if os.path.exists(dns_checkpoint) and dns_start_idx > 0:
                dns_features = pd.read_csv(dns_checkpoint).to_dict('records')
                print(f"  Loaded {len(dns_features)} cached DNS results", flush=True)
            if os.path.exists(whois_checkpoint) and whois_start_idx > 0:
                whois_features = pd.read_csv(whois_checkpoint).to_dict('records')
                print(f"  Loaded {len(whois_features)} cached WHOIS results", flush=True)
        else:
            print(
                f"⚠️ Stale checkpoint for {batch_date} does not match current batch URLs — "
                f"discarding and extracting fresh.",
                flush=True,
            )
            for stale in (checkpoint_file, dns_checkpoint, whois_checkpoint):
                if os.path.exists(stale):
                    os.remove(stale)

    with Heartbeat() as hb:
        # Step 2: Extract DNS features
        if dns_start_idx < total_urls:
            print(f"\n{'=' * 60}", flush=True)
            print(f"STEP 2: Extract DNS features (starting at {dns_start_idx})", flush=True)
            print("=" * 60, flush=True)

            for idx in range(dns_start_idx, total_urls):
                row = df_batch.iloc[idx]
                hb.update("DNS extraction", idx + 1, total_urls)
                print(f"[{idx+1}/{total_urls}] {row['url'][:50]}...", flush=True)

                features = _run_in_process(extract_single_domain_features, row['url'])
                features['url'] = row['url']
                dns_features.append(features)

                # Checkpoint every N URLs
                if (idx + 1) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(dns_features).to_csv(dns_checkpoint, index=False)
                    with open(checkpoint_file, 'w') as f:
                        json.dump({"dns_completed": idx + 1, "whois_completed": whois_start_idx}, f)
                    print(f"  💾 DNS checkpoint saved at {idx + 1}/{total_urls}", flush=True)

            # Final DNS checkpoint
            pd.DataFrame(dns_features).to_csv(dns_checkpoint, index=False)
            with open(checkpoint_file, 'w') as f:
                json.dump({"dns_completed": total_urls, "whois_completed": whois_start_idx}, f)

        df_dns = pd.DataFrame(dns_features)
        print(f"✅ Extracted DNS features for {len(df_dns)} URLs", flush=True)

        # Step 3: Extract WHOIS features
        if whois_start_idx < total_urls:
            print(f"\n{'=' * 60}", flush=True)
            print(f"STEP 3: Extract WHOIS features (starting at {whois_start_idx})", flush=True)
            print("=" * 60, flush=True)

            for idx in range(whois_start_idx, total_urls):
                row = df_batch.iloc[idx]
                hb.update("WHOIS extraction", idx + 1, total_urls)
                print(f"[{idx+1}/{total_urls}] {row['url'][:50]}...", flush=True)

                features = _run_in_process(_whois_for_url, row['url'])
                features['url'] = row['url']
                whois_features.append(features)

                # Checkpoint every N URLs
                if (idx + 1) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(whois_features).to_csv(whois_checkpoint, index=False)
                    with open(checkpoint_file, 'w') as f:
                        json.dump({"dns_completed": total_urls, "whois_completed": idx + 1}, f)
                    print(f"  💾 WHOIS checkpoint saved at {idx + 1}/{total_urls}", flush=True)

            # Final WHOIS checkpoint
            pd.DataFrame(whois_features).to_csv(whois_checkpoint, index=False)

        df_whois = pd.DataFrame(whois_features)
        print(f"✅ Extracted WHOIS features for {len(df_whois)} URLs", flush=True)
        hb.update("Accumulating masters", 0, 0)

    # Step 4: Prepare THREE SEPARATE master files
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 4: Prepare SEPARATE master files (URL, DNS, WHOIS)", flush=True)
    print("=" * 60, flush=True)

    # Add label column to DNS and WHOIS from batch
    df_dns = df_dns.merge(df_batch[['url', 'label']], on='url', how='left')
    df_whois = df_whois.merge(df_batch[['url', 'label']], on='url', how='left')

    print(f"✅ URL features: {len(df_url_features)} rows × {len(df_url_features.columns)} columns", flush=True)
    print(f"✅ DNS features: {len(df_dns)} rows × {len(df_dns.columns)} columns", flush=True)
    print(f"✅ WHOIS features: {len(df_whois)} rows × {len(df_whois.columns)} columns", flush=True)

    # Step 5: Download existing master datasets from S3
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 5: Download existing SEPARATE master datasets from S3", flush=True)
    print("=" * 60, flush=True)

    masters = {
        'url': {'new': df_url_features, 'file': 'vm_data/master/url_features_master.csv', 's3_key': 'master/url_features_master.csv'},
        'dns': {'new': df_dns, 'file': 'vm_data/master/dns_features_master.csv', 's3_key': 'master/dns_features_master.csv'},
        'whois': {'new': df_whois, 'file': 'vm_data/master/whois_features_master.csv', 's3_key': 'master/whois_features_master.csv'}
    }

    for feature_type, config in masters.items():
        # Only treat a genuine "object does not exist" (404 NoSuchKey) as a fresh
        # start. ANY other error (timeout, network blip, throttling) must ABORT —
        # otherwise a transient read failure silently overwrites the whole master.
        # This is exactly what wiped the 609k-row WHOIS master on 2026-05-22.
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=config['s3_key'])
            exists = True
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code in ('404', 'NoSuchKey', 'NotFound'):
                print(f"ℹ️  No existing {feature_type.upper()} master - creating new", flush=True)
                config['existing'] = None
                exists = False
            else:
                raise RuntimeError(
                    f"FATAL: could not stat {feature_type.upper()} master "
                    f"(code={code}). Aborting to avoid overwriting accumulated data."
                ) from e

        if exists:
            # Retry the download a few times — never give up and create-new on
            # a transient error for a file we KNOW exists.
            last_err = None
            for attempt in range(3):
                try:
                    s3.download_file(S3_BUCKET, config['s3_key'], config['file'])
                    config['existing'] = pd.read_csv(config['file'])
                    print(f"📊 Existing {feature_type.upper()} master: {len(config['existing'])} rows", flush=True)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"  ⚠️ Download attempt {attempt + 1} failed for {feature_type.upper()}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            if last_err is not None:
                raise RuntimeError(
                    f"FATAL: {feature_type.upper()} master exists in S3 but could not "
                    f"be downloaded after 3 attempts. Aborting to avoid data loss."
                ) from last_err

    # Step 6: Accumulate each master separately
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 6: Accumulate EACH master dataset separately", flush=True)
    print("=" * 60, flush=True)

    for feature_type, config in masters.items():
        print(f"\n--- {feature_type.upper()} ---", flush=True)
        print(f"📊 New batch: {len(config['new'])} rows", flush=True)

        if config['existing'] is not None:
            df_combined = pd.concat([config['existing'], config['new']], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')

            added = len(df_combined) - len(config['existing'])
            duplicates = len(config['new']) - added

            print(f"✅ Combined: {len(df_combined)} rows (+{added} new, {duplicates} duplicates)", flush=True)
        else:
            df_combined = config['new']
            print(f"✅ Initial: {len(df_combined)} rows", flush=True)

        config['combined'] = df_combined
        df_combined.to_csv(config['file'], index=False)

    # Step 7: Upload all three masters back to S3
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 7: Upload THREE SEPARATE master datasets to S3", flush=True)
    print("=" * 60, flush=True)

    for feature_type, config in masters.items():
        # Shrink guard: never replace a master with one that's drastically smaller.
        # Accumulation should only ever grow (or stay same after dedup). A large
        # drop means something went wrong upstream — refuse to upload.
        existing_n = len(config['existing']) if config['existing'] is not None else 0
        new_n = len(config['combined'])
        if existing_n > 100 and new_n < existing_n * 0.5:
            print(
                f"🚫 ABORT upload for {feature_type.upper()}: new master ({new_n} rows) "
                f"is <50% of existing ({existing_n} rows). Refusing to shrink master.",
                flush=True,
            )
            raise RuntimeError(
                f"{feature_type.upper()} master would shrink from {existing_n} to {new_n} rows — aborting."
            )
        s3.upload_file(config['file'], S3_BUCKET, config['s3_key'])
        print(f"✅ Uploaded {feature_type.upper()} master: {new_n} rows", flush=True)

    # Also upload combined for backwards compatibility
    merged = df_url_features.merge(df_dns.drop(columns=['label'], errors='ignore'), on='url', how='left')
    merged = merged.merge(df_whois.drop(columns=['label'], errors='ignore'), on='url', how='left')
    combined_file = "vm_data/master/phishing_features_master.csv"

    try:
        s3.download_file(S3_BUCKET, "master/phishing_features_master.csv", combined_file)
        df_existing_combined = pd.read_csv(combined_file)
        df_combined_all = pd.concat([df_existing_combined, merged], ignore_index=True)
        df_combined_all = df_combined_all.drop_duplicates(subset=['url'], keep='last')
    except Exception:
        df_combined_all = merged

    df_combined_all.to_csv(combined_file, index=False)
    s3.upload_file(combined_file, S3_BUCKET, "master/phishing_features_master.csv")
    print(f"✅ Uploaded combined master (backwards compat): {len(df_combined_all)} rows", flush=True)

    # Cleanup checkpoint files on success
    for f in [checkpoint_file, dns_checkpoint, whois_checkpoint]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n{'=' * 60}", flush=True)
    print("✅ COMPLETE: THREE separate feature masters created!", flush=True)
    print("=" * 60, flush=True)
    print(f"URL master:   {len(masters['url']['combined'])} rows", flush=True)
    print(f"DNS master:   {len(masters['dns']['combined'])} rows", flush=True)
    print(f"WHOIS master: {len(masters['whois']['combined'])} rows", flush=True)


def _cleanup_ec2_scratch():
    """Remove every local file this script may have written on EC2.

    EC2 is stateless scratch space — S3 is the only durable store. Anything
    this script writes locally is a copy-of-a-copy and should not survive
    the run, or the 8 GB root disk fills up in a few weeks.

    Runs in a finally block so it happens even on crash/timeout.
    """
    import shutil

    targets = [
        # Scratch dirs the script itself creates
        "vm_data",
        # Local caches the whois.py / dns_ipwhois.py libraries write to
        "data/processed/whois_results.csv",
        "data/processed/dns_ipwhois_results.csv",
        "logs/lookup_times.csv",
    ]
    for path in targets:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            # Cleanup must not raise — we might be in a finally after a crash.
            print(f"  (cleanup skip: {path}: {e})", flush=True)


def run_batch_in_scratch(batch_date: str) -> None:
    """Public entry point — runs extract_and_accumulate() in an ephemeral tempdir.

    Used by both:
      * the CLI (`python scripts/extract_vm_features_aws.py <batch_date>`)
      * the ec2_daemon.py polling loop (one call per queued batch)

    Guarantees:
      * The whole run happens in a fresh /tmp/phishnet-extract-XXX/ so nothing
        accumulates on the EC2 root disk between runs.
      * The tempdir is auto-removed on process exit; a belt-and-suspenders
        finally block also explicitly cleans up known relative paths
        (`vm_data/`, `data/processed/`, `logs/`) in case tempdir cleanup ever
        fails (SIGKILL, out-of-space during rm, etc).
    """
    import tempfile

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="phishnet-extract-") as scratch:
        os.chdir(scratch)
        try:
            extract_and_accumulate(batch_date)
        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass
            _cleanup_ec2_scratch()
            print(
                "🧹 EC2 scratch cleanup done — disk returned to clean state.",
                flush=True,
            )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        batch_date = sys.argv[1]
    elif len(sys.argv) == 3:
        batch_date = sys.argv[2]
    else:
        print("Usage: python extract_vm_features_aws.py <batch_date>")
        print("Example: python extract_vm_features_aws.py 20260125")
        sys.exit(1)

    run_batch_in_scratch(batch_date)
