#!/usr/bin/env python3
"""
URL Fetcher - Download URLs from public sources
===============================================
Fetches URLs from PhishTank, OpenPhish, and generates legitimate URLs.
Called by daily_url_collector.sh
"""

import os
import sys
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_urls(output_file: str, target_count: int = 1000):
    """
    Fetch URLs from public sources until target is reached.

    Args:
        output_file: Path to save fetched URLs
        target_count: Target number of URLs to fetch (default: 1000)

    Returns:
        Number of URLs fetched
    """
    print(f"Fetching URLs from public sources...")
    print(f"  Target: {target_count} URLs")
    print()

    all_urls = []

    # 1. PhishTank (phishing URLs) - Get ALL available
    try:
        print("→ PhishTank...")
        response = requests.get(
            'http://data.phishtank.com/data/online-valid.csv',
            timeout=60
        )
        if response.status_code == 200:
            lines = response.text.split('\n')[1:]  # Skip header
            phishtank_count = 0
            for line in lines:  # Take ALL available
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        url = parts[1].strip('"')
                        all_urls.append({'url': url, 'label': 'phishing', 'source': 'phishtank'})
                        phishtank_count += 1
            print(f"  ✓ Fetched {phishtank_count} phishing URLs from PhishTank")
    except Exception as e:
        print(f"  ✗ PhishTank failed: {e}")

    # 2. OpenPhish (phishing URLs) - Get ALL available
    try:
        print("→ OpenPhish...")
        response = requests.get(
            'https://openphish.com/feed.txt',
            timeout=60
        )
        if response.status_code == 200:
            openphish_count = 0
            for url in response.text.split('\n'):  # Take ALL available
                if url.strip():
                    all_urls.append({'url': url.strip(), 'label': 'phishing', 'source': 'openphish'})
                    openphish_count += 1
            print(f"  ✓ Fetched {openphish_count} phishing URLs from OpenPhish")
    except Exception as e:
        print(f"  ✗ OpenPhish failed: {e}")

    # 3. URLhaus (malware/phishing URLs)
    try:
        print("→ URLhaus...")
        response = requests.get(
            'https://urlhaus.abuse.ch/downloads/csv_recent/',
            timeout=60
        )
        if response.status_code == 200:
            lines = response.text.split('\n')
            urlhaus_count = 0
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        url = parts[2].strip('"')
                        if url.startswith('http'):
                            all_urls.append({'url': url, 'label': 'phishing', 'source': 'urlhaus'})
                            urlhaus_count += 1
            print(f"  ✓ Fetched {urlhaus_count} malicious URLs from URLhaus")
    except Exception as e:
        print(f"  ✗ URLhaus failed: {e}")

    # 4. PhishStats (phishing URLs)
    try:
        print("→ PhishStats...")
        response = requests.get(
            'https://phishstats.info/phish_score.csv',
            timeout=60
        )
        if response.status_code == 200:
            lines = response.text.split('\n')[1:]  # Skip header
            phishstats_count = 0
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        url = parts[1].strip('"')
                        if url.startswith('http'):
                            all_urls.append({'url': url, 'label': 'phishing', 'source': 'phishstats'})
                            phishstats_count += 1
            print(f"  ✓ Fetched {phishstats_count} phishing URLs from PhishStats")
    except Exception as e:
        print(f"  ✗ PhishStats failed: {e}")

    # Remove duplicates before counting
    df_phishing = pd.DataFrame(all_urls)
    df_phishing = df_phishing.drop_duplicates(subset=['url'])
    phishing_total = len(df_phishing)
    print()
    print(f"📊 Total unique phishing URLs: {phishing_total}")

    # 5. Legitimate URLs - Always generate exactly half of target for balance
    legit_needed = target_count // 2
    print(f"→ Generating {legit_needed} legitimate URLs for balanced dataset...")

    legit_domains = [
        # Top websites
        'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
        'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'netflix.com',
        'microsoft.com', 'apple.com', 'adobe.com', 'zoom.us', 'dropbox.com',

        # News & Media
        'cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'reuters.com',
        'bloomberg.com', 'forbes.com', 'washingtonpost.com', 'wsj.com', 'espn.com',
        'usatoday.com', 'time.com', 'nationalgeographic.com', 'wired.com', 'techcrunch.com',

        # Tech & Development
        'github.com', 'stackoverflow.com', 'medium.com', 'dev.to', 'gitlab.com',
        'bitbucket.org', 'npmjs.com', 'pypi.org', 'docker.com', 'kubernetes.io',
        'aws.amazon.com', 'cloud.google.com', 'azure.microsoft.com', 'heroku.com', 'vercel.com',

        # E-commerce
        'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
        'aliexpress.com', 'shopify.com', 'wayfair.com', 'homedepot.com', 'lowes.com',
        'costco.com', 'macys.com', 'nordstrom.com', 'zappos.com', 'overstock.com',

        # Services
        'paypal.com', 'stripe.com', 'chase.com', 'bankofamerica.com', 'wellsfargo.com',
        'gmail.com', 'outlook.com', 'yahoo.com', 'icloud.com', 'protonmail.com',
        'venmo.com', 'squareup.com', 'citibank.com', 'usbank.com', 'capitalone.com',

        # Education
        'coursera.org', 'udemy.com', 'khanacademy.org', 'edx.org', 'mit.edu',
        'stanford.edu', 'harvard.edu', 'berkeley.edu', 'yale.edu', 'oxford.ac.uk',
        'cambridge.ac.uk', 'princeton.edu', 'columbia.edu', 'cornell.edu', 'upenn.edu',

        # Entertainment
        'spotify.com', 'twitch.tv', 'tiktok.com', 'vimeo.com', 'soundcloud.com',
        'hulu.com', 'disneyplus.com', 'hbo.com', 'primevideo.com', 'crunchyroll.com',
        'pandora.com', 'imgur.com', 'deviantart.com', 'behance.net', 'artstation.com',

        # Cloud & SaaS
        'salesforce.com', 'slack.com', 'notion.so', 'trello.com', 'asana.com',
        'atlassian.com', 'monday.com', 'zendesk.com', 'hubspot.com', 'mailchimp.com',
        'airtable.com', 'figma.com', 'canva.com', 'miro.com', 'clickup.com',

        # Travel & Transportation
        'booking.com', 'airbnb.com', 'expedia.com', 'tripadvisor.com', 'kayak.com',
        'uber.com', 'lyft.com', 'delta.com', 'united.com', 'southwest.com'
    ]

    legit_urls = []
    for domain in legit_domains:
        # Add root domain
        legit_urls.append({'url': f'https://{domain}', 'label': 'legitimate', 'source': 'known_good'})
        # Add www subdomain
        legit_urls.append({'url': f'https://www.{domain}', 'label': 'legitimate', 'source': 'known_good'})
        # Add common paths
        legit_urls.append({'url': f'https://{domain}/about', 'label': 'legitimate', 'source': 'known_good'})
        legit_urls.append({'url': f'https://{domain}/contact', 'label': 'legitimate', 'source': 'known_good'})
        legit_urls.append({'url': f'https://{domain}/products', 'label': 'legitimate', 'source': 'known_good'})

        if len(legit_urls) >= legit_needed:
            break

    legit_urls = legit_urls[:legit_needed]
    print(f"  ✓ Generated {len(legit_urls)} legitimate URLs")

    # Combine all URLs
    all_urls = df_phishing.to_dict('records') + legit_urls

    # Final deduplication and limiting
    df = pd.DataFrame(all_urls)
    df = df.drop_duplicates(subset=['url'])

    # If we have more than target, sample to get exact target with balanced classes
    if len(df) > target_count:
        df_phish = df[df['label'] == 'phishing']
        df_legit = df[df['label'] == 'legitimate']

        # Take equal amounts from each
        n_per_class = target_count // 2
        df_phish = df_phish.sample(n=min(len(df_phish), n_per_class), random_state=42)
        df_legit = df_legit.sample(n=min(len(df_legit), n_per_class), random_state=42)

        df = pd.concat([df_phish, df_legit], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    df.to_csv(output_file, index=False)

    print()
    print(f"✅ Total URLs collected: {len(df)}")
    print(f"   Phishing: {len(df[df['label'] == 'phishing'])}")
    print(f"   Legitimate: {len(df[df['label'] == 'legitimate'])}")
    print(f"   Saved to: {output_file}")

    return len(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fetch_urls.py <output_file> [target_count]")
        sys.exit(1)

    output_file = sys.argv[1]
    target_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = fetch_urls(output_file, target_count)
    sys.exit(0 if count > 0 else 1)
