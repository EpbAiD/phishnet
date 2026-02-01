#!/usr/bin/env python3
"""
URL Fetcher - Download URLs from multiple public sources
=========================================================
Fetches phishing URLs from 10+ threat intelligence feeds and generates legitimate URLs.
Maximizes data collection by pulling ALL available URLs from each source.
"""

import os
import sys
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Request timeout for API calls
REQUEST_TIMEOUT = 120


def fetch_phishtank():
    """Fetch ALL URLs from PhishTank (verified phishing)."""
    urls = []
    try:
        print("→ PhishTank...")
        response = requests.get(
            'http://data.phishtank.com/data/online-valid.csv',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            lines = response.text.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        url = parts[1].strip('"')
                        if url.startswith('http'):
                            urls.append({'url': url, 'label': 'phishing', 'source': 'phishtank'})
            print(f"  ✓ Fetched {len(urls)} phishing URLs from PhishTank")
    except Exception as e:
        print(f"  ✗ PhishTank failed: {e}")
    return urls


def fetch_openphish():
    """Fetch ALL URLs from OpenPhish (real-time phishing)."""
    urls = []
    try:
        print("→ OpenPhish...")
        response = requests.get(
            'https://openphish.com/feed.txt',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for url in response.text.split('\n'):
                if url.strip() and url.startswith('http'):
                    urls.append({'url': url.strip(), 'label': 'phishing', 'source': 'openphish'})
            print(f"  ✓ Fetched {len(urls)} phishing URLs from OpenPhish")
    except Exception as e:
        print(f"  ✗ OpenPhish failed: {e}")
    return urls


def fetch_urlhaus():
    """Fetch ALL URLs from URLhaus (malware distribution URLs)."""
    urls = []
    try:
        print("→ URLhaus...")
        response = requests.get(
            'https://urlhaus.abuse.ch/downloads/csv_recent/',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            lines = response.text.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        url = parts[2].strip('"')
                        if url.startswith('http'):
                            urls.append({'url': url, 'label': 'phishing', 'source': 'urlhaus'})
            print(f"  ✓ Fetched {len(urls)} malicious URLs from URLhaus")
    except Exception as e:
        print(f"  ✗ URLhaus failed: {e}")
    return urls


def fetch_phishstats():
    """Fetch ALL URLs from PhishStats (phishing score database)."""
    urls = []
    try:
        print("→ PhishStats...")
        response = requests.get(
            'https://phishstats.info/phish_score.csv',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            lines = response.text.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        url = parts[1].strip('"')
                        if url.startswith('http'):
                            urls.append({'url': url, 'label': 'phishing', 'source': 'phishstats'})
            print(f"  ✓ Fetched {len(urls)} phishing URLs from PhishStats")
    except Exception as e:
        print(f"  ✗ PhishStats failed: {e}")
    return urls


def fetch_phishing_army():
    """Fetch ALL domains from Phishing Army blocklist."""
    urls = []
    try:
        print("→ Phishing Army...")
        # Try extended blocklist first (more URLs)
        response = requests.get(
            'https://phishing.army/download/phishing_army_blocklist_extended.txt',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                domain = line.strip()
                if domain and not domain.startswith('#'):
                    # Convert domain to URL
                    urls.append({'url': f'https://{domain}', 'label': 'phishing', 'source': 'phishing_army'})
            print(f"  ✓ Fetched {len(urls)} phishing domains from Phishing Army")
    except Exception as e:
        print(f"  ✗ Phishing Army failed: {e}")
    return urls


def fetch_urlabuse():
    """Fetch ALL URLs from URLAbuse (multiple feeds)."""
    urls = []
    feeds = [
        ('https://urlabuse.com/public/data/phishing_url.txt', 'phishing'),
        ('https://urlabuse.com/public/data/malware_url.txt', 'malware'),
        ('https://urlabuse.com/public/data/hacked_url.txt', 'hacked'),
    ]

    try:
        print("→ URLAbuse...")
        total = 0
        for feed_url, feed_type in feeds:
            try:
                response = requests.get(feed_url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    count = 0
                    for line in response.text.split('\n'):
                        url = line.strip()
                        if url and url.startswith('http'):
                            urls.append({'url': url, 'label': 'phishing', 'source': f'urlabuse_{feed_type}'})
                            count += 1
                    total += count
            except:
                pass
        print(f"  ✓ Fetched {total} URLs from URLAbuse feeds")
    except Exception as e:
        print(f"  ✗ URLAbuse failed: {e}")
    return urls


def fetch_threatview():
    """Fetch ALL URLs from ThreatView.io high-confidence feed."""
    urls = []
    try:
        print("→ ThreatView.io...")
        response = requests.get(
            'https://threatview.io/Downloads/URL-High-Confidence-Feed.txt',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append({'url': url, 'label': 'phishing', 'source': 'threatview'})
            print(f"  ✓ Fetched {len(urls)} malicious URLs from ThreatView")
    except Exception as e:
        print(f"  ✗ ThreatView failed: {e}")
    return urls


def fetch_digitalside():
    """Fetch ALL URLs from DigitalSide Threat-Intel."""
    urls = []
    try:
        print("→ DigitalSide Threat-Intel...")
        response = requests.get(
            'https://osint.digitalside.it/Threat-Intel/lists/latesturls.txt',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append({'url': url, 'label': 'phishing', 'source': 'digitalside'})
            print(f"  ✓ Fetched {len(urls)} malicious URLs from DigitalSide")
    except Exception as e:
        print(f"  ✗ DigitalSide failed: {e}")
    return urls


def fetch_malwaredomainlist():
    """Fetch URLs from Malware Domain List."""
    urls = []
    try:
        print("→ Malware Domain List...")
        response = requests.get(
            'https://www.malwaredomainlist.com/mdlcsv.php',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                if line.strip() and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        url = parts[1].strip('"')
                        if url and '/' in url:
                            # It's a path, prepend http
                            if not url.startswith('http'):
                                url = f'http://{url}'
                            urls.append({'url': url, 'label': 'phishing', 'source': 'malwaredomainlist'})
            print(f"  ✓ Fetched {len(urls)} malicious URLs from Malware Domain List")
    except Exception as e:
        print(f"  ✗ Malware Domain List failed: {e}")
    return urls


def fetch_vxvault():
    """Fetch URLs from VXVault (malware)."""
    urls = []
    try:
        print("→ VXVault...")
        response = requests.get(
            'http://vxvault.net/URL_List.php',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append({'url': url, 'label': 'phishing', 'source': 'vxvault'})
            print(f"  ✓ Fetched {len(urls)} malware URLs from VXVault")
    except Exception as e:
        print(f"  ✗ VXVault failed: {e}")
    return urls


def fetch_cybercrime_tracker():
    """Fetch URLs from Cybercrime Tracker."""
    urls = []
    try:
        print("→ Cybercrime Tracker...")
        response = requests.get(
            'https://cybercrime-tracker.net/all.php',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                url = line.strip()
                if url and (url.startswith('http') or '/' in url):
                    if not url.startswith('http'):
                        url = f'http://{url}'
                    urls.append({'url': url, 'label': 'phishing', 'source': 'cybercrime_tracker'})
            print(f"  ✓ Fetched {len(urls)} C2/malware URLs from Cybercrime Tracker")
    except Exception as e:
        print(f"  ✗ Cybercrime Tracker failed: {e}")
    return urls


def fetch_mitchellkrogza():
    """Fetch URLs from mitchellkrogza phishing database on GitHub."""
    urls = []
    try:
        print("→ Mitchell Krogza Phishing Database...")
        response = requests.get(
            'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            for line in response.text.split('\n'):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append({'url': url, 'label': 'phishing', 'source': 'mitchellkrogza'})
            print(f"  ✓ Fetched {len(urls)} phishing URLs from Mitchell Krogza Database")
    except Exception as e:
        print(f"  ✗ Mitchell Krogza failed: {e}")
    return urls


def fetch_alienvault_otx():
    """Fetch phishing URLs from AlienVault OTX public pulses."""
    urls = []
    try:
        print("→ AlienVault OTX...")
        # Fetch recent phishing-related pulses (public API, no key needed for basic access)
        response = requests.get(
            'https://otx.alienvault.com/api/v1/pulses/subscribed?limit=50&page=1',
            headers={'Accept': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            for pulse in data.get('results', []):
                for indicator in pulse.get('indicators', []):
                    if indicator.get('type') == 'URL':
                        url = indicator.get('indicator', '')
                        if url.startswith('http'):
                            urls.append({'url': url, 'label': 'phishing', 'source': 'alienvault_otx'})
            print(f"  ✓ Fetched {len(urls)} URLs from AlienVault OTX")
        else:
            print(f"  ⚠ AlienVault OTX returned status {response.status_code}")
    except Exception as e:
        print(f"  ✗ AlienVault OTX failed: {e}")
    return urls


def generate_legitimate_urls(count: int):
    """Generate legitimate URLs from known-good domains."""
    import random

    legit_domains = [
        # Top websites
        'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
        'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'netflix.com',
        'microsoft.com', 'apple.com', 'adobe.com', 'zoom.us', 'dropbox.com',

        # News & Media
        'cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'reuters.com',
        'bloomberg.com', 'forbes.com', 'washingtonpost.com', 'wsj.com', 'espn.com',
        'usatoday.com', 'time.com', 'nationalgeographic.com', 'wired.com', 'techcrunch.com',
        'npr.org', 'politico.com', 'economist.com', 'ft.com', 'latimes.com',
        'cbsnews.com', 'nbcnews.com', 'abcnews.go.com', 'foxnews.com', 'msnbc.com',
        'apnews.com', 'huffpost.com', 'axios.com', 'vox.com', 'buzzfeed.com',
        'businessinsider.com', 'variety.com', 'hollywoodreporter.com', 'deadline.com', 'ign.com',

        # Tech & Development
        'github.com', 'stackoverflow.com', 'medium.com', 'dev.to', 'gitlab.com',
        'bitbucket.org', 'npmjs.com', 'pypi.org', 'docker.com', 'kubernetes.io',
        'aws.amazon.com', 'cloud.google.com', 'azure.microsoft.com', 'heroku.com', 'vercel.com',
        'digitalocean.com', 'linode.com', 'cloudflare.com', 'netlify.com', 'mongodb.com',
        'postgres.org', 'redis.io', 'elastic.co', 'jenkins.io', 'circleci.com',
        'travis-ci.com', 'codepen.io', 'replit.com', 'glitch.com', 'codesandbox.io',
        'hackernews.com', 'techradar.com', 'arstechnica.com', 'theverge.com', 'engadget.com',
        'cnet.com', 'zdnet.com', 'venturebeat.com', 'slashdot.org', 'pcmag.com',

        # E-commerce
        'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
        'aliexpress.com', 'shopify.com', 'wayfair.com', 'homedepot.com', 'lowes.com',
        'costco.com', 'macys.com', 'nordstrom.com', 'zappos.com', 'overstock.com',
        'kohls.com', 'jcpenney.com', 'sephora.com', 'ulta.com', 'nike.com',
        'adidas.com', 'gap.com', 'zara.com', 'hm.com', 'uniqlo.com',
        'ikea.com', 'crateandbarrel.com', 'williams-sonoma.com', 'bedbathandbeyond.com', 'chewy.com',
        'newegg.com', 'bhphotovideo.com', 'rei.com', 'dickssportinggoods.com', 'gamestop.com',

        # Financial Services
        'paypal.com', 'stripe.com', 'chase.com', 'bankofamerica.com', 'wellsfargo.com',
        'venmo.com', 'squareup.com', 'citibank.com', 'usbank.com', 'capitalone.com',
        'americanexpress.com', 'discover.com', 'fidelity.com', 'schwab.com', 'vanguard.com',
        'etrade.com', 'tdameritrade.com', 'robinhood.com', 'coinbase.com', 'kraken.com',
        'mint.com', 'creditkarma.com', 'nerdwallet.com', 'experian.com', 'equifax.com',
        'transunion.com', 'truist.com', 'pnc.com', 'regions.com', 'ally.com',

        # Email & Communication
        'gmail.com', 'outlook.com', 'yahoo.com', 'icloud.com', 'protonmail.com',
        'zoho.com', 'aol.com', 'mail.com', 'gmx.com', 'tutanota.com',
        'fastmail.com', 'yandex.com', 'disroot.org', 'mailfence.com', 'runbox.com',

        # Education
        'coursera.org', 'udemy.com', 'khanacademy.org', 'edx.org', 'mit.edu',
        'stanford.edu', 'harvard.edu', 'berkeley.edu', 'yale.edu', 'oxford.ac.uk',
        'cambridge.ac.uk', 'princeton.edu', 'columbia.edu', 'cornell.edu', 'upenn.edu',
        'caltech.edu', 'uchicago.edu', 'northwestern.edu', 'duke.edu', 'jhu.edu',
        'umich.edu', 'ucla.edu', 'ucsd.edu', 'ucsb.edu', 'uci.edu',
        'uw.edu', 'utexas.edu', 'wisc.edu', 'uiuc.edu', 'umn.edu',
        'codecademy.com', 'pluralsight.com', 'skillshare.com', 'udacity.com', 'datacamp.com',
        'brilliant.org', 'duolingo.com', 'memrise.com', 'babbel.com', 'rosettastone.com',

        # Entertainment & Streaming
        'spotify.com', 'twitch.tv', 'tiktok.com', 'vimeo.com', 'soundcloud.com',
        'hulu.com', 'disneyplus.com', 'hbo.com', 'primevideo.com', 'crunchyroll.com',
        'pandora.com', 'imgur.com', 'deviantart.com', 'behance.net', 'artstation.com',
        'pinterest.com', 'flickr.com', 'unsplash.com', 'pexels.com', 'giphy.com',
        'dailymotion.com', 'metacafe.com', 'rumble.com', 'odysee.com', 'dtube.video',
        'imdb.com', 'rottentomatoes.com', 'letterboxd.com', 'allmovie.com', 'moviefone.com',

        # Cloud & SaaS
        'salesforce.com', 'slack.com', 'notion.so', 'trello.com', 'asana.com',
        'atlassian.com', 'monday.com', 'zendesk.com', 'hubspot.com', 'mailchimp.com',
        'airtable.com', 'figma.com', 'canva.com', 'miro.com', 'clickup.com',
        'freshdesk.com', 'intercom.com', 'drift.com', 'typeform.com', 'surveymonkey.com',
        'jotform.com', 'wufoo.com', 'formstack.com', 'cognito.com', 'calendly.com',
        'docusign.com', 'hellosign.com', 'adobe.io', 'smartsheet.com', 'workday.com',

        # Travel & Transportation
        'booking.com', 'airbnb.com', 'expedia.com', 'tripadvisor.com', 'kayak.com',
        'uber.com', 'lyft.com', 'delta.com', 'united.com', 'southwest.com',
        'americanairlines.com', 'jetblue.com', 'spirit.com', 'frontier.com', 'allegiant.com',
        'hotels.com', 'priceline.com', 'hotwire.com', 'orbitz.com', 'travelocity.com',
        'vrbo.com', 'homeaway.com', 'hostelworld.com', 'agoda.com', 'rentalcars.com',
        'skyscanner.com', 'momondo.com', 'kiwi.com', 'dohop.com', 'rome2rio.com',

        # Government & Public
        'usa.gov', 'irs.gov', 'uscis.gov', 'usps.com', 'whitehouse.gov',
        'congress.gov', 'senate.gov', 'house.gov', 'fda.gov', 'cdc.gov',
        'nih.gov', 'nasa.gov', 'noaa.gov', 'usgs.gov', 'nps.gov',
        'state.gov', 'defense.gov', 'va.gov', 'opm.gov', 'gsa.gov',
        'dmv.org', 'medicare.gov', 'socialsecurity.gov', 'weather.gov', 'ready.gov',

        # Health & Medical
        'mayoclinic.org', 'clevelandclinic.org', 'hopkinsmedicine.org', 'webmd.com', 'healthline.com',
        'medlineplus.gov', 'drugs.com', 'rxlist.com', 'cvs.com', 'walgreens.com',
        'rite-aid.com', 'express-scripts.com', 'goodrx.com', 'ro.co',
        'teladoc.com', 'mdlive.com', 'zocdoc.com', 'healthgrades.com', 'vitals.com',

        # Food & Delivery
        'doordash.com', 'ubereats.com', 'grubhub.com', 'postmates.com', 'seamless.com',
        'instacart.com', 'shipt.com', 'freshdirect.com', 'peapod.com',
        'dominos.com', 'pizzahut.com', 'papajohns.com', 'mcdonalds.com', 'starbucks.com',
        'chipotle.com', 'subway.com', 'wendys.com', 'tacobell.com', 'kfc.com',

        # Social & Community
        'discord.com', 'telegram.org', 'signal.org', 'whatsapp.com', 'snapchat.com',
        'tumblr.com', 'meetup.com', 'nextdoor.com', 'yelp.com', 'foursquare.com',
        'quora.com', 'answers.com', 'ask.com',

        # Gaming
        'steam.com', 'epicgames.com', 'gog.com', 'origin.com', 'ubisoft.com',
        'ea.com', 'blizzard.com', 'riotgames.com', 'valvesoftware.com', 'minecraft.net',
        'playstation.com', 'xbox.com', 'nintendo.com', 'roblox.com', 'fortnite.com'
    ]

    print(f"→ Generating {count} legitimate URLs...")

    urls = []
    random.seed(int(datetime.now().timestamp()))
    shuffled_domains = random.sample(legit_domains, len(legit_domains))
    common_paths = ['', '/about', '/contact', '/help', '/support', '/faq', '/terms', '/privacy', '/login', '/signup']

    for domain in shuffled_domains:
        if len(urls) >= count:
            break

        # Add root URL
        urls.append({'url': f'https://{domain}', 'label': 'legitimate', 'source': 'known_good'})

        # Add www variant
        if len(urls) < count:
            urls.append({'url': f'https://www.{domain}', 'label': 'legitimate', 'source': 'known_good'})

        # Add some path variations
        for path in random.sample(common_paths, min(3, len(common_paths))):
            if len(urls) >= count:
                break
            if path:
                urls.append({'url': f'https://{domain}{path}', 'label': 'legitimate', 'source': 'known_good'})

    print(f"  ✓ Generated {len(urls)} legitimate URLs")
    return urls


def fetch_urls(output_file: str, target_count: int = 1000):
    """
    Fetch URLs from ALL public sources and deduplicate against master.

    Args:
        output_file: Path to save fetched URLs
        target_count: Target number of NEW URLs to fetch (default: 1000)

    Returns:
        Number of URLs fetched
    """
    start_time = time.time()

    print("=" * 80)
    print("PHISHING URL COLLECTION - MAXIMUM DATA HARVEST")
    print("=" * 80)
    print(f"Target: {target_count} NEW URLs (balanced 50:50)")
    print(f"Strategy: Fetch ALL available URLs from ALL sources")
    print("=" * 80)
    print()

    # Load existing URLs to avoid duplicates
    existing_urls = set()
    master_file = "data/processed/phishing_features_complete.csv"
    if os.path.exists(master_file):
        try:
            df_existing = pd.read_csv(master_file)
            if 'url' in df_existing.columns:
                existing_urls = set(df_existing['url'].tolist())
                print(f"📂 Found {len(existing_urls)} existing URLs in master dataset")
                print(f"   Will fetch only NEW urls to avoid duplicates\n")
        except Exception as e:
            print(f"⚠️  Could not load existing URLs: {e}\n")

    # Fetch from ALL sources (maximize data collection)
    print("=" * 80)
    print("FETCHING FROM ALL THREAT INTELLIGENCE SOURCES")
    print("=" * 80)

    all_phishing_urls = []

    # Call each fetcher and collect URLs
    fetchers = [
        ("PhishTank", fetch_phishtank),
        ("OpenPhish", fetch_openphish),
        ("URLhaus", fetch_urlhaus),
        ("PhishStats", fetch_phishstats),
        ("Phishing Army", fetch_phishing_army),
        ("URLAbuse", fetch_urlabuse),
        ("ThreatView", fetch_threatview),
        ("DigitalSide", fetch_digitalside),
        ("Mitchell Krogza DB", fetch_mitchellkrogza),
        ("Malware Domain List", fetch_malwaredomainlist),
        ("VXVault", fetch_vxvault),
        ("Cybercrime Tracker", fetch_cybercrime_tracker),
        ("AlienVault OTX", fetch_alienvault_otx),
    ]

    source_stats = {}
    for name, fetcher in fetchers:
        try:
            urls = fetcher()
            all_phishing_urls.extend(urls)
            source_stats[name] = len(urls)
        except Exception as e:
            print(f"  ✗ {name} failed with error: {e}")
            source_stats[name] = 0
        time.sleep(0.5)  # Small delay between requests

    # Create DataFrame and deduplicate
    print()
    print("=" * 80)
    print("DEDUPLICATION AND FILTERING")
    print("=" * 80)

    df_phishing = pd.DataFrame(all_phishing_urls)
    total_fetched = len(df_phishing)

    # Remove duplicates within fetched data
    df_phishing = df_phishing.drop_duplicates(subset=['url'])
    after_internal_dedup = len(df_phishing)

    print(f"Total URLs fetched: {total_fetched}")
    print(f"After internal dedup: {after_internal_dedup} (removed {total_fetched - after_internal_dedup})")

    # Remove URLs that exist in master
    df_phishing_new = df_phishing[~df_phishing['url'].isin(existing_urls)]
    after_master_dedup = len(df_phishing_new)

    print(f"After master dedup: {after_master_dedup} (removed {after_internal_dedup - after_master_dedup} duplicates)")

    # Generate legitimate URLs
    print()
    legit_needed = target_count // 2
    legit_urls = generate_legitimate_urls(legit_needed)
    df_legit = pd.DataFrame(legit_urls)
    df_legit = df_legit.drop_duplicates(subset=['url'])

    # Balance classes - aim for 50:50 split
    n_per_class = target_count // 2

    df_phish_sample = df_phishing_new.sample(
        n=min(len(df_phishing_new), n_per_class),
        random_state=42
    ) if len(df_phishing_new) > 0 else df_phishing_new

    df_legit_sample = df_legit.sample(
        n=min(len(df_legit), n_per_class),
        random_state=42
    ) if len(df_legit) > 0 else df_legit

    # Combine
    df_final = pd.concat([df_phish_sample, df_legit_sample], ignore_index=True)

    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df_final.to_csv(output_file, index=False)

    elapsed = time.time() - start_time

    # Print summary
    print()
    print("=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print()
    print("Source Statistics:")
    for source, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"  {source:25} {count:>6} URLs")
    print()
    print(f"Total phishing URLs fetched: {total_fetched}")
    print(f"Unique phishing URLs: {after_internal_dedup}")
    print(f"NEW phishing URLs (not in master): {after_master_dedup}")
    print()
    print("FINAL DATASET:")
    print(f"  Phishing URLs: {len(df_phish_sample)}")
    print(f"  Legitimate URLs: {len(df_legit_sample)}")
    print(f"  Total: {len(df_final)}")
    print(f"  Balance: {len(df_legit_sample)/len(df_final)*100:.1f}% legitimate" if len(df_final) > 0 else "")
    print()
    print(f"Saved to: {output_file}")
    print("=" * 80)

    return len(df_final)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fetch_urls.py <output_file> [target_count]")
        sys.exit(1)

    output_file = sys.argv[1]
    target_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = fetch_urls(output_file, target_count)
    sys.exit(0 if count > 0 else 1)
