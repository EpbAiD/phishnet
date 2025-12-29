# ===============================================================
# fetch_all_urls.py  🚀 Final Balanced Fetch Script
# ---------------------------------------------------------------
# 🔹 Fetches phishing + legitimate URLs from 6 real sources
# 🔹 Deduplicates ONLY on full URL (domain missing is allowed!)
# 🔹 Creates 6 equal buckets (≈8k each → ~48k final dataset)
# 🔹 NO cleaning, NO domain logic here
# ===============================================================

import os, pandas as pd, requests, kagglehub
from io import StringIO
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
import random

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
OUT_PATH = os.path.join(RAW_DIR, "final_urls_balanced.csv")

TARGET_PER_BUCKET = 8000  # ~8k × 6 buckets ≈ 48k total


# ---------------- Helper Utilities ----------------
def safe_download_csv(url):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return pd.read_csv(StringIO(r.text))
    except:
        pass
    return pd.DataFrame()


def safe_fetch_lines(url):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return [x.strip() for x in r.text.splitlines() if x.strip()]
    except:
        pass
    return []


# ---------------- Kaggle (Brand Spoof + Mixed) ----------------
def get_kaggle_bucket():
    try:
        print("📦 Downloading Kaggle dataset...")
        path = kagglehub.dataset_download("hassaanmustafavi/phishing-urls-dataset")
        for f in os.listdir(path):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, f), usecols=["url", "type"])
                df["bucket"] = "kaggle_brand"  # SUBCATEGORY
                df = df.rename(columns={"type": "label"})
                return df
    except:
        pass
    return pd.DataFrame(columns=["url", "label", "bucket"])


# ---------------- Phishing (Feeds) ----------------
def get_credential_bucket():
    # OpenPhish + (PhishTank only if accessible)
    urls = set(safe_fetch_lines("https://openphish.com/feed.txt"))

    # Try PhishTank Public
    pt = safe_fetch_lines("https://data.phishtank.com/data/online-valid.csv")
    if len(pt) > 100:  # Only use if accessible
        urls.update(pt)
        source = "openphish_phishtank"
    else:
        source = "openphish_only"

    return pd.DataFrame({"url": list(urls), "label": "phishing", "bucket": "credential_" + source})


def get_malware_bucket():
    urls = set()

    # URLHaus
    try:
        t = requests.get("https://urlhaus.abuse.ch/downloads/csv_recent/", timeout=20).text
        rows = [r.split(",") for r in t.splitlines() if "http" in r]
        urls.update([r[2].strip('"') for r in rows if len(r) > 2])
    except:
        pass

    # ThreatFox
    for l in safe_fetch_lines("https://threatfox.abuse.ch/export/csv/host/"):
        if "." in l and not l.startswith("#"):
            urls.add(l.split(",")[0])

    return pd.DataFrame({"url": list(urls), "label": "phishing", "bucket": "malware"})


# ---------------- Legitimate Buckets ----------------
def get_majestic_bucket():
    df = safe_download_csv("https://downloads.majestic.com/majestic_million.csv")
    if df.empty:
        return pd.DataFrame(columns=["url", "label", "bucket"])
    urls = ["https://" + d for d in df["Domain"].dropna()]
    return pd.DataFrame({"url": urls, "label": "legit", "bucket": "legit_majestic"})


def get_tranco_bucket():
    try:
        with NamedTemporaryFile(suffix=".zip") as tmp:
            r = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=20)
            tmp.write(r.content); tmp.flush()
            with ZipFile(tmp.name) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None, names=["rank", "domain"])
                    urls = ["https://" + d for d in df["domain"].dropna()]
        return pd.DataFrame({"url": urls, "label": "legit", "bucket": "legit_tranco"})
    except:
        return pd.DataFrame(columns=["url", "label", "bucket"])


def get_modern_tech_bucket():
    """
    Fetch modern legitimate URLs with complex structures.
    Includes GitHub repos, StackOverflow questions, modern SaaS platforms.
    These URLs have patterns (long paths, many digits, special chars) that
    older training data lacks, causing false positives.
    """
    urls = []

    # GitHub repositories - diverse languages, organizations, projects
    github_repos = [
        "https://github.com/microsoft/vscode",
        "https://github.com/tensorflow/tensorflow",
        "https://github.com/facebook/react",
        "https://github.com/torvalds/linux",
        "https://github.com/python/cpython",
        "https://github.com/kubernetes/kubernetes",
        "https://github.com/nodejs/node",
        "https://github.com/rust-lang/rust",
        "https://github.com/golang/go",
        "https://github.com/apple/swift",
        "https://github.com/docker/docker-ce",
        "https://github.com/ansible/ansible",
        "https://github.com/elastic/elasticsearch",
        "https://github.com/django/django",
        "https://github.com/rails/rails",
        "https://github.com/vuejs/vue",
        "https://github.com/angular/angular",
        "https://github.com/apache/spark",
        "https://github.com/mongodb/mongo",
        "https://github.com/redis/redis",
        "https://github.com/pytorch/pytorch",
        "https://github.com/opencv/opencv",
        "https://github.com/numpy/numpy",
        "https://github.com/pandas-dev/pandas",
        "https://github.com/scikit-learn/scikit-learn",
        "https://github.com/home-assistant/core",
        "https://github.com/grafana/grafana",
        "https://github.com/prometheus/prometheus",
        "https://github.com/hashicorp/terraform",
        "https://github.com/microsoft/TypeScript",
    ]

    # StackOverflow questions - various ID patterns, tags, digit-heavy
    stackoverflow_urls = [
        "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array",
        "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
        "https://stackoverflow.com/questions/1732348/regex-match-open-tags-except-xhtml-self-contained-tags",
        "https://stackoverflow.com/questions/927358/how-do-i-undo-the-most-recent-local-commits-in-git",
        "https://stackoverflow.com/questions/292357/what-is-private-implementation-class-idiom",
        "https://stackoverflow.com/questions/419163/what-does-if-name-main-do",
        "https://stackoverflow.com/questions/184618/what-is-the-best-comment-in-source-code-you-have-ever-encountered",
        "https://stackoverflow.com/questions/503093/how-do-i-redirect-to-another-webpage",
        "https://stackoverflow.com/questions/6841333/why-is-subtracting-these-two-times-in-1927-giving-a-strange-result",
        "https://stackoverflow.com/questions/1026069/how-do-i-make-the-first-letter-of-a-string-uppercase-in-javascript",
        "https://stackoverflow.com/questions/477816/what-is-the-correct-json-content-type",
        "https://stackoverflow.com/questions/34657/how-do-i-detect-a-click-outside-an-element",
        "https://stackoverflow.com/questions/927386/how-can-i-undo-a-git-commit-locally-and-remotely",
        "https://stackoverflow.com/questions/11828270/how-do-i-exit-the-vim-editor",
        "https://stackoverflow.com/questions/892782/what-is-the-difference-between-px-dip-dp-and-sp",
    ]

    # Modern SaaS and web apps - complex URLs with tokens, IDs, hashes
    saas_urls = [
        "https://app.slack.com/client/T123ABC456/C987XYZ654",
        "https://discord.com/channels/123456789012345678/987654321098765432",
        "https://trello.com/b/AbCdEfGh/project-board-name",
        "https://www.notion.so/workspace/Page-Title-123abc456def789012345678",
        "https://airtable.com/appABCDEF12345/tblXYZ987654/viwGHI246810",
        "https://app.asana.com/0/1234567890123456/9876543210987654",
        "https://linear.app/team/issue/ABC-123/feature-title",
        "https://vercel.com/username/project-name/deployments/abc123def456",
        "https://app.netlify.com/sites/project-name/deploys/5f8a7b6c9d0e1f2a3b4c5d6e",
        "https://console.aws.amazon.com/s3/buckets/my-bucket-name?region=us-east-1&tab=objects",
        "https://portal.azure.com/#view/Microsoft_Azure_Monitoring/AzureMonitoringBrowseBlade",
        "https://console.cloud.google.com/storage/browser/project-bucket-123",
        "https://app.terraform.io/app/org-name/workspaces/workspace-name/runs/run-abc123",
        "https://sentry.io/organizations/org-slug/issues/1234567890/?project=9876543",
        "https://app.datadoghq.com/dashboard/abc-123-def/overview?from_ts=1234567890",
    ]

    # Developer platforms and package managers - digit-heavy, technical patterns
    dev_platform_urls = [
        "https://pypi.org/project/numpy/1.24.3/",
        "https://www.npmjs.com/package/react/v/18.2.0",
        "https://hub.docker.com/r/library/nginx/tags",
        "https://crates.io/crates/tokio/0.1.22",
        "https://pub.dev/packages/flutter_bloc/versions/8.1.3",
        "https://search.maven.org/artifact/org.springframework.boot/spring-boot-starter-web/3.0.5/jar",
        "https://packagist.org/packages/symfony/symfony",
        "https://rubygems.org/gems/rails/versions/7.0.4",
        "https://www.nuget.org/packages/Newtonsoft.Json/13.0.3",
        "https://pkg.go.dev/golang.org/x/tools/cmd/goimports",
        "https://marketplace.visualstudio.com/items?itemName=ms-python.python",
        "https://plugins.jetbrains.com/plugin/6954-kotlin",
        "https://addons.mozilla.org/en-US/firefox/addon/ublock-origin/",
        "https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi",
    ]

    # E-commerce with complex URLs (product pages, search results, cart)
    ecommerce_urls = [
        "https://www.amazon.com/dp/B08N5WRWNW?ref=ppx_yo_dt_b_product_details",
        "https://www.ebay.com/itm/123456789012?hash=item1a2b3c4d5e:g:ABCdefGHIjkl",
        "https://www.etsy.com/listing/987654321/custom-handmade-item-name?ref=shop_home_active_1",
        "https://www.aliexpress.com/item/1234567890123.html?spm=a2g0o.productlist.0.0",
        "https://www.walmart.com/ip/Product-Name/123456789?athbdg=L1200",
        "https://www.target.com/p/product-slug/-/A-12345678",
        "https://www.bestbuy.com/site/product-name/1234567.p?skuId=9876543",
        "https://www.newegg.com/p/N82E16811352120?Item=N82E16811352120",
    ]

    # Modern media and social platforms (NOT just domain roots)
    social_media_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.reddit.com/r/programming/comments/abc123/interesting_discussion/",
        "https://medium.com/@username/article-title-slug-123abc456def",
        "https://dev.to/username/article-slug-12ab",
        "https://twitter.com/user/status/1234567890123456789",
        "https://www.linkedin.com/in/username/details/experience/",
        "https://www.instagram.com/p/AbC123dEfG4/",
        "https://www.tiktok.com/@username/video/1234567890123456789",
        "https://open.spotify.com/track/1a2b3c4d5e6f7g8h9i0j?si=abc123def456",
        "https://soundcloud.com/artist-name/track-title-12345",
    ]

    # Documentation and learning platforms - technical content with paths
    docs_urls = [
        "https://docs.python.org/3/library/asyncio-task.html",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map",
        "https://docs.docker.com/engine/reference/commandline/run/",
        "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
        "https://reactjs.org/docs/hooks-reference.html#useeffect",
        "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
        "https://nodejs.org/api/fs.html#fspromisesreadfilepath-options",
        "https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python",
        "https://cloud.google.com/storage/docs/json_api/v1/objects/insert",
    ]

    # Combine all modern URLs
    urls.extend(github_repos)
    urls.extend(stackoverflow_urls)
    urls.extend(saas_urls)
    urls.extend(dev_platform_urls)
    urls.extend(ecommerce_urls)
    urls.extend(social_media_urls)
    urls.extend(docs_urls)

    return pd.DataFrame({"url": urls, "label": "legit", "bucket": "legit_modern_tech"})


# ---------------- BALANCING LOGIC ----------------
def sample_bucket(df):
    """Soft balance: max 8000, but do NOT force minimum size."""
    df = df.copy()
    df["url"] = df["url"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["url"])

    if len(df) > TARGET_PER_BUCKET:
        df = df.sample(TARGET_PER_BUCKET, random_state=42)

    # If df is small, we still keep it fully (NO oversampling here)
    return df


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n🚀 Fetching All URL Sources...\n")

    buckets = [
        get_kaggle_bucket(),
        get_credential_bucket(),
        get_malware_bucket(),
        get_majestic_bucket(),
        get_tranco_bucket(),
        get_modern_tech_bucket()
    ]

    # Sample each bucket equally
    buckets = [sample_bucket(b) for b in buckets]

    # Combine all balanced buckets
    final_df = pd.concat(buckets, ignore_index=True).dropna(subset=["url"])
    final_df = final_df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    final_df.to_csv(OUT_PATH, index=False)
    print(f"\n🎉 FINAL BALANCED DATASET READY → {OUT_PATH}")
    print(f"📌 Total URLs: {len(final_df):,}")
    print(final_df.groupby(['label', 'bucket']).size())