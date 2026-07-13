-- PhishNet feature tables. Idempotent — safe to re-run.
--
-- These replace the S3 master CSVs. Daemon upserts one batch (~3000 rows)
-- per run instead of downloading, concatenating, and re-uploading the full
-- master, which was OOMing t3.micro at ~500 MB masters.
--
-- Column types are permissive TEXT/DOUBLE PRECISION/BOOLEAN to match what
-- pandas would infer from the source CSVs. `updated_at` is added so we can
-- see when each row last changed.

CREATE TABLE IF NOT EXISTS url_features (
    url                                   TEXT PRIMARY KEY,
    label                                 TEXT,
    source                                TEXT,
    url_length                            INTEGER,
    hostname_length                       INTEGER,
    path_length                           INTEGER,
    num_subdomains                        INTEGER,
    num_dots                              INTEGER,
    num_special_chars                     INTEGER,
    num_digits                            INTEGER,
    num_uppercase_chars                   INTEGER,
    has_at_symbol                         TEXT,
    has_double_slash_redirect             INTEGER,
    has_dash_in_domain                    TEXT,
    is_ip_address                         INTEGER,
    ip_category                           TEXT,
    has_encoded_chars                     TEXT,
    has_non_ascii_chars                   TEXT,
    url_entropy                           DOUBLE PRECISION,
    hostname_entropy                      DOUBLE PRECISION,
    digit_to_letter_ratio                 DOUBLE PRECISION,
    domain_quality                        TEXT,
    tld_length                            INTEGER,
    subdomain_entropy                     DOUBLE PRECISION,
    subdomain_length                      INTEGER,
    has_login_keyword                     INTEGER,
    has_suspicious_words                  INTEGER,
    has_brand_mismatch                    INTEGER,
    file_type                             TEXT,
    is_file_download                      INTEGER,
    is_script_file                        INTEGER,
    is_shortened                          INTEGER,
    num_fragments                         INTEGER,
    num_query_params                      INTEGER,
    num_directories                       INTEGER,
    port                                  INTEGER,
    is_risky_port                         INTEGER,
    protocol_mismatch                     INTEGER,
    is_unknown_port                       INTEGER,
    contains_hex_encoding                 INTEGER,
    starts_with_https_but_contains_http   INTEGER,
    missing_hostname_flag                 INTEGER,
    collected_at                          TIMESTAMPTZ,
    updated_at                            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dns_features (
    domain               TEXT PRIMARY KEY,
    has_A                DOUBLE PRECISION,
    num_A                DOUBLE PRECISION,
    has_AAAA             DOUBLE PRECISION,
    num_AAAA             DOUBLE PRECISION,
    has_MX               DOUBLE PRECISION,
    num_MX               DOUBLE PRECISION,
    has_NS               DOUBLE PRECISION,
    num_NS               DOUBLE PRECISION,
    has_TXT              DOUBLE PRECISION,
    num_TXT              DOUBLE PRECISION,
    has_CNAME            DOUBLE PRECISION,
    cname_chain_length   DOUBLE PRECISION,
    has_SOA              DOUBLE PRECISION,
    ttl_min              DOUBLE PRECISION,
    ttl_max              DOUBLE PRECISION,
    ttl_mean             DOUBLE PRECISION,
    ttl_var              DOUBLE PRECISION,
    mx_priority_min      DOUBLE PRECISION,
    mx_priority_max      DOUBLE PRECISION,
    num_distinct_ips     DOUBLE PRECISION,
    txt_entropy          DOUBLE PRECISION,
    has_SPF              DOUBLE PRECISION,
    has_DKIM             DOUBLE PRECISION,
    has_DMARC            DOUBLE PRECISION,
    has_wildcard_dns     DOUBLE PRECISION,
    dnssec_enabled       DOUBLE PRECISION,
    asn_list             TEXT,
    asn_org_list         TEXT,
    asn_country_list     TEXT,
    cidr_list            TEXT,
    error_type           DOUBLE PRECISION,
    url                  TEXT,
    label                TEXT,
    collected_at         TIMESTAMPTZ,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);
-- URL index so joining phishing_features on dns.url is fast.
CREATE INDEX IF NOT EXISTS dns_features_url_idx ON dns_features (url);

CREATE TABLE IF NOT EXISTS whois_features (
    url                        TEXT PRIMARY KEY,
    registrar                  TEXT,
    whois_server               TEXT,
    creation_date              TEXT,
    expiration_date            TEXT,
    updated_date               TEXT,
    domain_age_days            DOUBLE PRECISION,
    registration_length_days   DOUBLE PRECISION,
    status                     TEXT,
    registrant_country         TEXT,
    has_privacy_protection     DOUBLE PRECISION,
    whois_success              DOUBLE PRECISION,
    error_msg                  TEXT,
    label                      TEXT,
    collected_at               TIMESTAMPTZ,
    updated_at                 TIMESTAMPTZ DEFAULT NOW()
);

-- Joined view. This replaces the phishing_features_master.csv join that the
-- daemon used to materialize by hand. Postgres runs the join lazily; training
-- reads it as a table.
CREATE OR REPLACE VIEW phishing_features AS
SELECT
    u.*,
    d.has_A, d.num_A, d.has_AAAA, d.num_AAAA, d.has_MX, d.num_MX,
    d.has_NS, d.num_NS, d.has_TXT, d.num_TXT, d.has_CNAME, d.cname_chain_length,
    d.has_SOA, d.ttl_min, d.ttl_max, d.ttl_mean, d.ttl_var,
    d.mx_priority_min, d.mx_priority_max, d.num_distinct_ips, d.txt_entropy,
    d.has_SPF, d.has_DKIM, d.has_DMARC, d.has_wildcard_dns, d.dnssec_enabled,
    d.asn_list, d.asn_org_list, d.asn_country_list, d.cidr_list, d.error_type,
    w.registrar, w.whois_server, w.creation_date, w.expiration_date,
    w.updated_date, w.domain_age_days, w.registration_length_days,
    w.status AS whois_status, w.registrant_country, w.has_privacy_protection,
    w.whois_success, w.error_msg AS whois_error_msg
FROM url_features u
LEFT JOIN dns_features   d ON d.url = u.url
LEFT JOIN whois_features w ON w.url = u.url;
