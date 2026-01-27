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

    This function implements a retry loop to ensure we get exactly target_count
    NEW URLs (not counting duplicates). If initial fetch returns fewer new URLs
    due to duplicates, it will retry with increased fetch counts.

    Args:
        output_file: Path to save fetched URLs
        target_count: Target number of NEW URLs to fetch (default: 1000)

    Returns:
        Number of URLs fetched
    """
    print(f"Fetching URLs from public sources...")
    print(f"  Target: {target_count} NEW URLs")
    print()

    # Load existing URLs to avoid duplicates across runs
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

    # Retry loop: keep fetching until we have enough NEW URLs
    max_attempts = 5
    attempt = 1
    fetch_multiplier = 1.0
    collected_new_urls = []

    while len(collected_new_urls) < target_count and attempt <= max_attempts:
        if attempt > 1:
            print(f"\n{'='*80}")
            print(f"🔄 RETRY ATTEMPT {attempt}/{max_attempts}")
            print(f"   Current: {len(collected_new_urls)} new URLs")
            print(f"   Needed: {target_count - len(collected_new_urls)} more URLs")
            print(f"   Fetch multiplier: {fetch_multiplier:.1f}x")
            print(f"{'='*80}\n")

        all_urls = []

        # Calculate how many more we need
        remaining_needed = target_count - len(collected_new_urls)
        adjusted_target = int(remaining_needed * fetch_multiplier)

        # 1. PhishTank (phishing URLs) - Fetch with offset to get different URLs
        try:
            print("→ PhishTank...")
            response = requests.get(
                'http://data.phishtank.com/data/online-valid.csv',
                timeout=60
            )
            if response.status_code == 200:
                lines = response.text.split('\n')[1:]  # Skip header
                phishtank_count = 0

                # On retries, skip URLs we've already seen
                start_offset = (attempt - 1) * adjusted_target
                end_offset = start_offset + adjusted_target

                for i, line in enumerate(lines):
                    if i < start_offset:
                        continue
                    if i >= end_offset:
                        break

                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            url = parts[1].strip('"')
                            all_urls.append({'url': url, 'label': 'phishing', 'source': 'phishtank'})
                            phishtank_count += 1
                print(f"  ✓ Fetched {phishtank_count} phishing URLs from PhishTank (offset: {start_offset})")
        except Exception as e:
            print(f"  ✗ PhishTank failed: {e}")

        # 2. OpenPhish (phishing URLs) - Fetch with offset
        try:
            print("→ OpenPhish...")
            response = requests.get(
                'https://openphish.com/feed.txt',
                timeout=60
            )
            if response.status_code == 200:
                openphish_count = 0
                urls = response.text.split('\n')

                start_offset = (attempt - 1) * adjusted_target
                end_offset = start_offset + adjusted_target

                for i, url in enumerate(urls):
                    if i < start_offset:
                        continue
                    if i >= end_offset:
                        break

                    if url.strip():
                        all_urls.append({'url': url.strip(), 'label': 'phishing', 'source': 'openphish'})
                        openphish_count += 1
                print(f"  ✓ Fetched {openphish_count} phishing URLs from OpenPhish (offset: {start_offset})")
        except Exception as e:
            print(f"  ✗ OpenPhish failed: {e}")

        # 3. URLhaus (malware/phishing URLs) - Fetch with offset
        try:
            print("→ URLhaus...")
            response = requests.get(
                'https://urlhaus.abuse.ch/downloads/csv_recent/',
                timeout=60
            )
            if response.status_code == 200:
                lines = response.text.split('\n')
                urlhaus_count = 0

                start_offset = (attempt - 1) * adjusted_target
                end_offset = start_offset + adjusted_target

                line_idx = 0
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        if line_idx < start_offset:
                            line_idx += 1
                            continue
                        if line_idx >= end_offset:
                            break

                        parts = line.split(',')
                        if len(parts) >= 3:
                            url = parts[2].strip('"')
                            if url.startswith('http'):
                                all_urls.append({'url': url, 'label': 'phishing', 'source': 'urlhaus'})
                                urlhaus_count += 1
                                line_idx += 1
                print(f"  ✓ Fetched {urlhaus_count} malicious URLs from URLhaus (offset: {start_offset})")
        except Exception as e:
            print(f"  ✗ URLhaus failed: {e}")

        # 4. PhishStats (phishing URLs) - Fetch with offset
        try:
            print("→ PhishStats...")
            response = requests.get(
                'https://phishstats.info/phish_score.csv',
                timeout=60
            )
            if response.status_code == 200:
                lines = response.text.split('\n')[1:]  # Skip header
                phishstats_count = 0

                start_offset = (attempt - 1) * adjusted_target
                end_offset = start_offset + adjusted_target

                for i, line in enumerate(lines):
                    if i < start_offset:
                        continue
                    if i >= end_offset:
                        break

                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            url = parts[1].strip('"')
                            if url.startswith('http'):
                                all_urls.append({'url': url, 'label': 'phishing', 'source': 'phishstats'})
                                phishstats_count += 1
                print(f"  ✓ Fetched {phishstats_count} phishing URLs from PhishStats (offset: {start_offset})")
        except Exception as e:
            print(f"  ✗ PhishStats failed: {e}")

        # Remove duplicates before counting
        df_phishing = pd.DataFrame(all_urls)
        df_phishing = df_phishing.drop_duplicates(subset=['url'])
        phishing_total = len(df_phishing)
        print()
        print(f"📊 Total unique phishing URLs fetched this round: {phishing_total}")

        # 5. Legitimate URLs - Generate proportional amount for 50:50 balance
        legit_needed = remaining_needed // 2  # 50% legitimate for balanced dataset
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
            'rite-aid.com', 'express-scripts.com', 'goodrx.com', 'blink health.com', 'ro.co',
            'teladoc.com', 'mdlive.com', 'zocdoc.com', 'healthgrades.com', 'vitals.com',

            # Food & Delivery
            'doordash.com', 'ubereats.com', 'grubhub.com', 'postmates.com', 'seamless.com',
            'instacart.com', 'shipt.com', 'freshdirect.com', 'peapod.com', 'amazon fresh.com',
            'dominos.com', 'pizzahut.com', 'papajohns.com', 'mcdonalds.com', 'starbucks.com',
            'chipotle.com', 'subway.com', 'wendys.com', 'tacobell.com', 'kfc.com',

            # Social & Community
            'discord.com', 'telegram.org', 'signal.org', 'whatsapp.com', 'snapchat.com',
            'tumblr.com', 'meetup.com', 'nextdoor.com', 'yelp.com', 'foursquare.com',
            'quora.com', 'answers.com', 'ask.com', 'yahoo.answers.com', 'reddit.com',

            # Gaming
            'steam.com', 'epicgames.com', 'gog.com', 'origin.com', 'ubisoft.com',
            'ea.com', 'blizzard.com', 'riotgames.com', 'valvesoftware.com', 'minecraft.net',
            'playstation.com', 'xbox.com', 'nintendo.com', 'roblox.com', 'fortnite.com'
        ]

        legit_urls = []
        import random

        # Seed for reproducibility within run but variation across runs
        random.seed(int(datetime.now().timestamp()) + attempt)

        # Shuffle domains for variety
        shuffled_domains = random.sample(legit_domains, len(legit_domains))

        # Common paths to add variety
        common_paths = ['', '/about', '/contact', '/help', '/support', '/faq', '/terms', '/privacy']

        for domain in shuffled_domains:
            if len(legit_urls) >= legit_needed:
                break

            # Add root URL
            legit_urls.append({'url': f'https://{domain}', 'label': 'legitimate', 'source': 'known_good'})

            # Add www variant
            if len(legit_urls) < legit_needed:
                legit_urls.append({'url': f'https://www.{domain}', 'label': 'legitimate', 'source': 'known_good'})

            # Add some path variations
            for path in random.sample(common_paths, min(3, len(common_paths))):
                if len(legit_urls) >= legit_needed:
                    break
                if path:  # Skip empty path (already added root)
                    legit_urls.append({'url': f'https://{domain}{path}', 'label': 'legitimate', 'source': 'known_good'})

        print(f"  ✓ Generated {len(legit_urls)} legitimate URLs")

        # IMPORTANT: Only deduplicate PHISHING URLs against master
        # Legitimate URLs can repeat across batches (they're known-good domains)
        # This prevents the class imbalance bug where all legit URLs get filtered

        # 1. Filter phishing URLs only (remove duplicates from master)
        existing_and_collected = existing_urls | set([u['url'] for u in collected_new_urls])
        phishing_before = len(df_phishing)
        df_phishing_new = df_phishing[~df_phishing['url'].isin(existing_and_collected)]
        phishing_removed = phishing_before - len(df_phishing_new)

        # 2. For legitimate URLs, only dedupe within this batch (not against master)
        df_legit = pd.DataFrame(legit_urls)
        df_legit = df_legit.drop_duplicates(subset=['url'])

        # 3. Combine: new phishing + all legitimate
        df_round = pd.concat([df_phishing_new, df_legit], ignore_index=True)

        print()
        print(f"🔄 Removed {phishing_removed} duplicate phishing URLs")
        print(f"   {len(df_phishing_new)} NEW phishing urls + {len(df_legit)} legitimate urls = {len(df_round)} total")

        # Add new URLs to collected list
        collected_new_urls.extend(df_round.to_dict('records'))

        print(f"\n📊 Progress: {len(collected_new_urls)}/{target_count} NEW URLs collected")

        # Check if we need to retry
        if len(collected_new_urls) >= target_count:
            print(f"\n✅ Target reached! Collected {len(collected_new_urls)} NEW URLs")
            break
        elif attempt < max_attempts:
            # Calculate new multiplier based on duplicate rate
            duplicate_rate = removed_count / before_count if before_count > 0 else 0.5
            fetch_multiplier = 1.0 + duplicate_rate + 0.5  # Increase by duplicate rate + buffer
            print(f"\n⚠️  Need more URLs. Duplicate rate: {duplicate_rate:.1%}")
            print(f"   Increasing fetch multiplier to {fetch_multiplier:.1f}x for next attempt")
        else:
            print(f"\n⚠️  Max attempts reached. Collected {len(collected_new_urls)} URLs (target: {target_count})")

        attempt += 1

    # Create final dataframe from collected URLs
    df_all_collected = pd.DataFrame(collected_new_urls)

    # Balance classes - aim for 50:50 split
    df_phish = df_all_collected[df_all_collected['label'] == 'phishing']
    df_legit = df_all_collected[df_all_collected['label'] == 'legitimate']

    # Calculate target split: 50% each class
    n_per_class = target_count // 2

    # Sample from each class
    df_phish_sample = df_phish.sample(n=min(len(df_phish), n_per_class), random_state=42) if len(df_phish) > 0 else df_phish
    df_legit_sample = df_legit.sample(n=min(len(df_legit), n_per_class), random_state=42) if len(df_legit) > 0 else df_legit

    # If one class is short, take more from the other to reach target
    total_sampled = len(df_phish_sample) + len(df_legit_sample)
    if total_sampled < target_count:
        shortage = target_count - total_sampled
        if len(df_phish_sample) < n_per_class and len(df_phish) > len(df_phish_sample):
            # Take more phishing if available
            additional = min(shortage, len(df_phish) - len(df_phish_sample))
            df_phish_sample = df_phish.sample(n=len(df_phish_sample) + additional, random_state=42)
        elif len(df_legit_sample) < n_per_class and len(df_legit) > len(df_legit_sample):
            # Take more legitimate if available
            additional = min(shortage, len(df_legit) - len(df_legit_sample))
            df_legit_sample = df_legit.sample(n=len(df_legit_sample) + additional, random_state=42)

    # Combine
    df_final = pd.concat([df_phish_sample, df_legit_sample], ignore_index=True)

    print(f"\n{'='*80}")
    print("FINAL DATASET COMPOSITION")
    print(f"{'='*80}")
    print(f"Phishing URLs: {len(df_phish_sample)}")
    print(f"Legitimate URLs: {len(df_legit_sample)}")
    print(f"Balance: {len(df_legit_sample)/len(df_final)*100:.1f}% legitimate" if len(df_final) > 0 else "")

    # Shuffle for randomness
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df_final.to_csv(output_file, index=False)

    print()
    print(f"✅ Total NEW URLs collected: {len(df_final)}")
    print(f"   Phishing: {len(df_phish_sample)}")
    print(f"   Legitimate: {len(df_legit_sample)}")
    print(f"   Saved to: {output_file}")

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
