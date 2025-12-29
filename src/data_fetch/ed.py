# ===============================================================
# extract_domains_from_urls.py  (STEP BEFORE WHOIS + DNS)
# ===============================================================
# Input  : data/raw/final_urls_balanced.csv
# Output : data/processed/full_domain_list.csv
# ---------------------------------------------------------------

import os
import pandas as pd
import tldextract

INP = "data/raw/final_urls_balanced.csv"
OUT = "data/processed/full_domain_list.csv"

os.makedirs("data/processed", exist_ok=True)

print(f"[INFO] Loading: {INP}")
df = pd.read_csv(INP)

def extract(url):
    try:
        t = tldextract.extract(url)
        if t.domain and t.suffix:
            return f"{t.domain}.{t.suffix}".lower()
        return None
    except:
        return None

df['domain'] = df['url'].astype(str).apply(extract)
df = df.dropna(subset=['domain']).drop_duplicates(subset=['domain'])

df[['domain']].to_csv(OUT, index=False)
print(f"✅ Saved: {OUT} ({df.shape[0]} unique domains)")