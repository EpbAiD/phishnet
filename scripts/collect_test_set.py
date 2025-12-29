#!/usr/bin/env python3
"""
Collect Fresh Test Set
=======================
Fetches FRESH phishing URLs that were never used in training.
These are real, unseen URLs for proper ensemble evaluation.

Strategy:
1. Fetch latest phishing URLs from OpenPhish/PhishTank
2. Remove any URLs that exist in training data
3. Collect diverse phishing types
4. Save as holdout test set for ensemble testing
"""

import os
import requests
import pandas as pd
from datetime import datetime

# Paths
DATA_DIR = "data/processed"
TEST_SET_PATH = "data/test/fresh_phishing_test_set.csv"
TRAINING_DATA = f"{DATA_DIR}/url_features_modelready.csv"

# Number of fresh URLs to collect
TARGET_PHISHING = 100
TARGET_LEGITIMATE = 100


def fetch_fresh_phishing_urls():
    """Fetch latest phishing URLs from OpenPhish."""
    print("Fetching fresh phishing URLs from OpenPhish...")

    try:
        response = requests.get('https://openphish.com/feed.txt', timeout=30)
        if response.status_code == 200:
            urls = [url.strip() for url in response.text.split('\n') if url.strip()]
            print(f"  ✓ Fetched {len(urls)} phishing URLs")
            return urls
        else:
            print(f"  ✗ OpenPhish returned status {response.status_code}")
            return []
    except Exception as e:
        print(f"  ✗ Failed to fetch from OpenPhish: {e}")
        return []


def fetch_fresh_legitimate_urls():
    """Generate legitimate URLs from top domains."""
    print("Generating fresh legitimate URLs...")

    # Top legitimate domains (not in training data variations)
    domains = [
        'shopify.com', 'zoom.us', 'cloudflare.com', 'godaddy.com',
        'salesforce.com', 'oracle.com', 'adobe.com', 'nvidia.com',
        'tesla.com', 'airbnb.com', 'uber.com', 'lyft.com',
        'dropbox.com', 'slack.com', 'spotify.com', 'twitch.tv',
        'tiktok.com', 'snapchat.com', 'discord.com', 'telegram.org',
        'wordpress.com', 'blogger.com', 'tumblr.com', 'yelp.com',
        'tripadvisor.com', 'booking.com', 'expedia.com', 'zillow.com',
        'etsy.com', 'craigslist.org', 'quora.com', 'stackoverflow.com',
        'github.com', 'gitlab.com', 'bitbucket.org', 'npmjs.com',
        'pypi.org', 'docker.com', 'kubernetes.io', 'apache.org',
        'mozilla.org', 'w3.org', 'gnu.org', 'debian.org',
        'ubuntu.com', 'redhat.com', 'fedoraproject.org', 'archlinux.org',
        'reuters.com', 'bloomberg.com', 'forbes.com', 'wsj.com'
    ]

    urls = []
    for domain in domains[:TARGET_LEGITIMATE]:
        urls.append(f'https://{domain}')
        urls.append(f'https://www.{domain}')

    print(f"  ✓ Generated {len(urls)} legitimate URLs")
    return urls[:TARGET_LEGITIMATE]


def remove_training_overlap(urls, labels):
    """Remove URLs that exist in training data."""
    print("\nRemoving overlap with training data...")

    if not os.path.exists(TRAINING_DATA):
        print("  ⚠️  No training data found - using all URLs")
        return urls, labels

    # Load training URLs
    df_train = pd.read_csv(TRAINING_DATA)
    training_urls = set(df_train['url'].values)

    print(f"  Training data contains {len(training_urls)} URLs")

    # Filter out overlapping URLs
    filtered_urls = []
    filtered_labels = []

    for url, label in zip(urls, labels):
        if url not in training_urls:
            filtered_urls.append(url)
            filtered_labels.append(label)

    removed = len(urls) - len(filtered_urls)
    print(f"  ✓ Removed {removed} overlapping URLs")
    print(f"  ✓ Kept {len(filtered_urls)} fresh URLs")

    return filtered_urls, filtered_labels


def main():
    print("="*80)
    print("COLLECTING FRESH TEST SET")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Fetch fresh phishing URLs
    phishing_urls = fetch_fresh_phishing_urls()

    # Fetch legitimate URLs
    legitimate_urls = fetch_fresh_legitimate_urls()

    # Combine
    all_urls = phishing_urls[:TARGET_PHISHING] + legitimate_urls[:TARGET_LEGITIMATE]
    all_labels = ['phishing'] * len(phishing_urls[:TARGET_PHISHING]) + \
                 ['legitimate'] * len(legitimate_urls[:TARGET_LEGITIMATE])

    print(f"\nCollected {len(all_urls)} total URLs:")
    print(f"  Phishing: {len([l for l in all_labels if l == 'phishing'])}")
    print(f"  Legitimate: {len([l for l in all_labels if l == 'legitimate'])}")

    # Remove training overlap
    fresh_urls, fresh_labels = remove_training_overlap(all_urls, all_labels)

    if len(fresh_urls) < 50:
        print(f"\n⚠️  Warning: Only {len(fresh_urls)} fresh URLs - may not be enough for testing")

    # Save test set
    os.makedirs(os.path.dirname(TEST_SET_PATH), exist_ok=True)

    df_test = pd.DataFrame({
        'url': fresh_urls,
        'label': fresh_labels,
        'collected_at': datetime.now().isoformat()
    })

    df_test.to_csv(TEST_SET_PATH, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Fresh test set saved: {TEST_SET_PATH}")
    print(f"{'='*80}")
    print(f"Total URLs: {len(df_test)}")
    print(f"  Phishing: {len(df_test[df_test['label'] == 'phishing'])}")
    print(f"  Legitimate: {len(df_test[df_test['label'] == 'legitimate'])}")
    print()
    print("Next step: Run ensemble testing with this fresh test set")
    print("="*80)


if __name__ == "__main__":
    main()
