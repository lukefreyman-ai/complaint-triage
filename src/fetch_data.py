"""
fetch_data.py — Pull consumer complaints from the CFPB public API.

The CFPB (Consumer Financial Protection Bureau) publishes every complaint
it receives, including the consumer's free-text narrative. That narrative
is what we'll send to the AI for classification, and `product` is the
ground-truth label we'll measure against.

No API key is needed — this is fully public data.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import requests

from taxonomy import VALID_PRODUCTS

# ── Paths ───────────────────────────────────────────────────────────────────
# Path(__file__) is the location of *this* file (src/fetch_data.py).
# .parent goes up to src/, .parent again goes up to the project root.
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
OUT_FILE      = DATA_DIR / "raw_complaints.json"
BALANCED_FILE = DATA_DIR / "balanced_complaints.json"

# ── API config ───────────────────────────────────────────────────────────────
# The CFPB search API accepts standard query params and returns JSON.
# Docs: https://cfpb.github.io/ccdb5-api/
CFPB_URL  = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

PARAMS = {
    "has_narrative": "true",   # only include complaints with free-text narratives
    "size": 600,               # number of results to request
    "no_aggs": "true",         # skip aggregation buckets (faster response)
    "date_received_min": "2024-01-01",
    # NOTE: the docs list a "format=json" param, but the live API currently
    # 404s if it's included — JSON is already the default response format,
    # so we just omit it.
}

# Fields we actually need — everything else gets dropped to keep the file small.
KEEP_FIELDS = {
    "complaint_id",
    "complaint_what_happened",  # the free-text narrative — this is the AI's input
    "product",                  # CFPB's label — this is the ground truth
    "sub_product",
    "issue",
    "company",
    "date_received",
}

MIN_NARRATIVE_LEN = 100  # complaints shorter than this are too vague to be useful

# ── Stratified (balanced) fetch config ────────────────────────────────────────
# The original fetch_complaints() above pulls one random batch of 600 recent
# complaints. Because ~84% of *all* CFPB complaints are "Credit reporting or
# other personal consumer reports", a random sample ends up almost entirely
# that one category — which makes "overall accuracy" a near-meaningless
# number (a model that always guesses "Credit reporting" would already score
# ~84% on a random sample, without learning anything).
#
# fetch_balanced_complaints() below fixes this by querying the API once per
# category (using the API's `product` filter) and capping each category at
# TARGET_PER_CATEGORY, so every category gets a fair shot in the dataset.
TARGET_PER_CATEGORY = 60

# We ask for more than TARGET_PER_CATEGORY per category because some hits
# will get dropped by the narrative-length filter below — this buffer makes
# it likely we still hit the target after filtering.
FETCH_SIZE_PER_CATEGORY = 150


def fetch_complaints() -> list[dict]:
    """Download complaints from the CFPB API and return a filtered list."""

    print(f"Fetching up to {PARAMS['size']} complaints from CFPB API...")
    print(f"URL: {CFPB_URL}")

    try:
        # timeout=(connect_seconds, read_seconds) — the CFPB API can be slow
        response = requests.get(CFPB_URL, params=PARAMS, timeout=(10, 60))
        response.raise_for_status()  # raises HTTPError for 4xx/5xx responses
    except requests.exceptions.ConnectionError:
        sys.exit("ERROR: Could not connect to the CFPB API. Check your internet connection.")
    except requests.exceptions.Timeout:
        sys.exit("ERROR: The CFPB API timed out. Try again in a few minutes.")
    except requests.exceptions.HTTPError as e:
        sys.exit(f"ERROR: CFPB API returned an error: {e}")

    data = response.json()

    # The API wraps results in {"hits": {"hits": [...]}}
    # Each hit has a "_source" key containing the actual complaint fields.
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        sys.exit("ERROR: API returned no results. The response structure may have changed.")

    print(f"  Raw results from API: {len(hits)}")

    # Filter and slim down each record
    records = []
    for hit in hits:
        source = hit.get("_source", {})

        narrative = source.get("complaint_what_happened", "")

        # Skip if narrative is missing or too short to be meaningful
        if not narrative or len(narrative) < MIN_NARRATIVE_LEN:
            continue

        # Skip if product label is missing — we need it as ground truth
        if not source.get("product"):
            continue

        # Keep only the fields we care about
        record = {k: source.get(k, "") for k in KEEP_FIELDS}
        records.append(record)

    return records


def fetch_balanced_complaints() -> list[dict]:
    """
    Download a STRATIFIED sample: one API call per product category, each
    capped at TARGET_PER_CATEGORY, instead of one random pull. This is what
    makes the eventual accuracy score mean something — see the comment above
    TARGET_PER_CATEGORY for why a random sample doesn't.

    A connection/timeout failure aborts the whole run (the network is down,
    retrying category-by-category won't help). An HTTP error for a single
    category (e.g. a malformed filter) is logged and that category is left
    at 0 rather than aborting everything else.
    """
    print(f"Fetching a balanced sample: up to {TARGET_PER_CATEGORY} complaints "
          f"per category, across {len(VALID_PRODUCTS)} categories...")

    all_records: list[dict] = []
    counts: dict[str, int] = {}

    for product in VALID_PRODUCTS:
        params = dict(PARAMS)  # copy so we don't mutate the shared PARAMS dict
        params["size"] = FETCH_SIZE_PER_CATEGORY
        params["product"] = product  # CFPB API supports filtering by exact product name

        try:
            response = requests.get(CFPB_URL, params=params, timeout=(10, 60))
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            sys.exit("ERROR: Could not connect to the CFPB API. Check your internet connection.")
        except requests.exceptions.Timeout:
            sys.exit("ERROR: The CFPB API timed out. Try again in a few minutes.")
        except requests.exceptions.HTTPError as e:
            print(f"  WARNING: request for {product!r} failed ({e}) — recording 0 for this category.")
            counts[product] = 0
            continue

        hits = response.json().get("hits", {}).get("hits", [])

        # Filter and slim down, same rules as fetch_complaints(), stopping
        # once we hit the per-category target.
        category_records = []
        for hit in hits:
            source = hit.get("_source", {})
            narrative = source.get("complaint_what_happened", "")

            if not narrative or len(narrative) < MIN_NARRATIVE_LEN:
                continue
            if not source.get("product"):
                continue

            category_records.append({k: source.get(k, "") for k in KEEP_FIELDS})
            if len(category_records) >= TARGET_PER_CATEGORY:
                break

        counts[product] = len(category_records)
        all_records.extend(category_records)
        print(f"  {product:<60} {len(category_records):>3}/{TARGET_PER_CATEGORY}"
              f"  (API had {len(hits)} raw hits with a narrative)")

    return all_records


def print_category_counts(records: list[dict]):
    """Print a table of how many records ended up in each product category."""
    counts = Counter(r["product"] for r in records)
    print("\n── Final category counts ───────────────────────────────────────")
    for product in VALID_PRODUCTS:
        n = counts.get(product, 0)
        flag = "  <-- below target" if n < TARGET_PER_CATEGORY else ""
        print(f"  {n:>4}  {product}{flag}")
    print(f"  {sum(counts.values()):>4}  TOTAL")
    print("───────────────────────────────────────────────────────────────")


def print_majority_baseline(records: list[dict]):
    """
    Print the accuracy you'd get by always guessing the single most common
    category in this dataset — i.e. the score a model gets for doing zero
    actual classification work. Any model's real accuracy has to be judged
    against this number, not against 0%.
    """
    counts = Counter(r["product"] for r in records)
    majority_product, majority_count = counts.most_common(1)[0]
    baseline = majority_count / len(records)

    print("\n── Majority-class baseline ─────────────────────────────────────")
    print(f"  Most common category: {majority_product} ({majority_count}/{len(records)})")
    print(f"  Baseline accuracy (always guess that category): {baseline:.1%}")
    print("  This is the number the model has to beat — anything close to")
    print("  this means the model isn't adding much signal over guessing.")
    print("───────────────────────────────────────────────────────────────")


def load_or_fetch_balanced() -> list[dict]:
    """
    Same caching pattern as load_or_fetch() below, but for the balanced
    (stratified) dataset. Deliberately a separate file (data/balanced_complaints.json)
    so the original random sample in data/raw_complaints.json is preserved
    for comparison.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if BALANCED_FILE.exists():
        with open(BALANCED_FILE) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from cache: {BALANCED_FILE}")
        return records

    records = fetch_balanced_complaints()

    if not records:
        sys.exit("ERROR: No records fetched for the balanced sample.")

    with open(BALANCED_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} records to {BALANCED_FILE}")
    return records


def load_or_fetch() -> list[dict]:
    """
    If the cache file already exists, load it. Otherwise fetch from the API.
    This is the function other scripts should call — it prevents us from
    hammering the API every time we run a test.
    """

    DATA_DIR.mkdir(exist_ok=True)

    if OUT_FILE.exists():
        # Cache hit — skip the network entirely
        with open(OUT_FILE) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from cache: {OUT_FILE}")
        return records

    # Cache miss — go to the network
    records = fetch_complaints()

    print(f"  After filtering (narrative >= {MIN_NARRATIVE_LEN} chars): {len(records)} records")

    if not records:
        sys.exit("ERROR: No records passed the filter. Try loosening MIN_NARRATIVE_LEN.")

    with open(OUT_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {OUT_FILE}")
    return records


# ── Run directly ─────────────────────────────────────────────────────────────
# `if __name__ == "__main__"` means this block only runs when you execute
# this file directly (python src/fetch_data.py). It does NOT run when another
# file imports fetch_data.
#
# Two modes:
#   python src/fetch_data.py              -> original random sample (unchanged)
#   python src/fetch_data.py --balanced   -> new stratified per-category sample
if __name__ == "__main__":
    if "--balanced" in sys.argv:
        records = load_or_fetch_balanced()
        print_category_counts(records)
        print_majority_baseline(records)
    else:
        records = load_or_fetch()

        # Print one sample so we can sanity-check the shape of the data
        sample = records[0]
        print("\n── Sample record ──────────────────────────────────────────────")
        print(f"  complaint_id : {sample['complaint_id']}")
        print(f"  date_received: {sample['date_received']}")
        print(f"  product      : {sample['product']}")
        print(f"  sub_product  : {sample['sub_product']}")
        print(f"  company      : {sample['company']}")
        print(f"  narrative    : {sample['complaint_what_happened'][:200]}...")
        print("───────────────────────────────────────────────────────────────")

        # Show the breakdown of product categories so we know what we're working with
        counts = Counter(r["product"] for r in records)
        print("\n── Product distribution ────────────────────────────────────────")
        for product, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {product}")
        print("───────────────────────────────────────────────────────────────")
