"""
classify.py — Send each complaint narrative to Claude and record its guess.

For every complaint we:
  1. Build a prompt containing the narrative and the closed VALID_PRODUCTS list.
  2. Ask Claude for a JSON object: predicted product, urgency, and a reason.
  3. Compare the prediction to the CFPB's real `product` label (ground truth).
  4. Append one row to data/results.csv.

Design choices explained inline below — look for comments starting with "Why".
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

from taxonomy import VALID_PRODUCTS, normalize_product
from fetch_data import load_or_fetch, load_or_fetch_balanced

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# Which complaint set to classify, and where its results go. Kept as separate
# files (rather than one results.csv with a "dataset" column) so the original
# random-sample results are never at risk of being overwritten by a balanced
# run, and so each file can be pointed at directly (e.g. by the Streamlit app).
DATASET_LOADERS = {
    "raw": load_or_fetch,
    "balanced": load_or_fetch_balanced,
}

# Keyed by (dataset, prompt_version). We only classify the balanced set with
# both prompts — the raw/skewed set already served its purpose in Phase 1/2
# and re-running prompt experiments on it would just re-inherit the same
# class-imbalance distortion.
RESULTS_FILES = {
    ("raw", 1): ROOT / "data" / "results.csv",
    ("balanced", 1): ROOT / "data" / "results_balanced.csv",
    ("balanced", 2): ROOT / "data" / "results_balanced_v2.csv",
}

# Model name as a constant, per the brief, so it's a one-line swap later.
# Haiku is Anthropic's fastest/cheapest model family — a good fit for a
# classification task like this where we're doing hundreds of short calls.
MODEL_NAME = "claude-haiku-4-5-20251001"

MAX_WORKERS = 5          # how many complaints we classify in parallel
MAX_RETRIES = 5          # retries per complaint on rate-limit errors
BASE_BACKOFF_SECONDS = 2 # doubles each retry: 2s, 4s, 8s, 16s, 32s

CSV_COLUMNS = [
    "complaint_id",
    "narrative",
    "true_product",
    "predicted_product",
    "urgency",
    "reason",
    "correct",
]


# ── Prompt v2: Debt collection / Credit reporting disambiguation ────────────
# Phase 2's evaluation showed the model's single biggest, highest-volume
# error is confusing "Debt collection" with "Credit reporting or other
# personal consumer reports" — 34 of 60 Debt collection complaints in the
# balanced set were misrouted there. Reading the actual narratives, the
# reason is structural: a huge fraction of debt collection complaints
# mention a credit report (because the collector reported the debt to a
# bureau), which pulls the model toward "Credit reporting" even though the
# complaint is really about the collector's conduct.
#
# DISAMBIGUATION_RULE gives the model an explicit tie-breaker for that
# specific overlap. FEW_SHOT_EXAMPLES then shows it three worked cases,
# including the exact boundary case (a collector furnishing bad info to a
# bureau) that the rule alone might still get argued around.
DISAMBIGUATION_RULE = """
DISAMBIGUATION RULE — Debt collection vs. Credit reporting:
These two categories overlap constantly because a debt collector's actions
often show up on a credit report, which makes it tempting to default to
"Credit reporting" whenever a report is mentioned. Use this rule instead:
- If the complaint is fundamentally about a debt collector's CONDUCT —
  including a collector furnishing inaccurate information to a credit
  bureau, repeated collection calls, disputing whether a debt is owed, or
  collection practices in general — classify as "Debt collection", even if
  a credit report is mentioned.
- Classify as "Credit reporting or other personal consumer reports" only
  when the complaint targets the credit BUREAU's own conduct (e.g. failing
  to investigate a dispute, mishandling personal data) or the general
  accuracy of the report itself, with no debt collector involved.
"""

FEW_SHOT_EXAMPLES = """
WORKED EXAMPLES:

Example 1:
Narrative: "A debt collection agency added a fraudulent account to my credit
report and has refused to remove it despite three dispute letters."
Correct classification: Debt collection
Why: The complaint is about the collector's conduct — furnishing false
information — even though the effect shows up on a credit report.

Example 2:
Narrative: "I disputed an error directly with Experian, but after 45 days
they still hadn't corrected it or told me the outcome of their
investigation."
Correct classification: Credit reporting or other personal consumer reports
Why: No debt collector is involved — this is about the bureau's own
investigation obligations.

Example 3:
Narrative: "A collection agency calls me multiple times a day about a debt
I already paid off, and it still shows as unpaid on my credit report."
Correct classification: Debt collection
Why: The core issue is the collector's behavior and inaccurate debt status,
not the bureau's handling of the report.
"""


def build_prompt(narrative: str, prompt_version: int = 1) -> str:
    """
    Construct the classification prompt.

    Why we spell out VALID_PRODUCTS explicitly: an LLM asked to freely name a
    "category" will paraphrase, invent subcategories, or use different
    capitalization every time. Giving it the exact closed list, and telling
    it to copy one verbatim, is what makes the accuracy score meaningful.

    Why we demand raw JSON with no markdown fences/preamble: it's much easier
    to parse programmatically. Models often wrap JSON in ```json ... ``` by
    habit, so classify.py also strips fences defensively (see parse_response).

    prompt_version: 1 = original baseline prompt. 2 = adds
    DISAMBIGUATION_RULE + FEW_SHOT_EXAMPLES targeting the Debt
    collection/Credit reporting confusion found in Phase 2. Kept as two
    separate, complete prompts (rather than one prompt with an if-branch
    buried inside it) so it's easy to read exactly what each version says,
    and to diff them.
    """
    product_list = "\n".join(f"- {p}" for p in VALID_PRODUCTS)

    extra_guidance = ""
    if prompt_version == 2:
        extra_guidance = DISAMBIGUATION_RULE + FEW_SHOT_EXAMPLES

    return f"""You are a consumer-finance complaint triage assistant.

Read the complaint narrative below and classify it.

VALID PRODUCT CATEGORIES (you must pick exactly one, copied verbatim):
{product_list}
{extra_guidance}
COMPLAINT NARRATIVE:
\"\"\"
{narrative}
\"\"\"

Respond with ONLY a raw JSON object (no markdown code fences, no preamble,
no explanation outside the JSON) with exactly these keys:
{{
  "product": "<one of the categories above, copied exactly>",
  "urgency": "<low, medium, or high>",
  "reason": "<one sentence explaining your classification>"
}}"""


def parse_response(text: str) -> dict:
    """
    Defensively parse the model's reply into a dict with keys
    product/urgency/reason. Raises ValueError on anything unparseable —
    callers catch this and record a PARSE_ERROR row instead of crashing.
    """
    cleaned = text.strip()

    # Strip ```json ... ``` or plain ``` ... ``` fences if the model added
    # them despite our instructions not to.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop the opening ``` or ```json line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop the closing ```
        cleaned = "\n".join(lines).strip()

    parsed = json.loads(cleaned)  # raises json.JSONDecodeError (a ValueError) if invalid

    for key in ("product", "urgency", "reason"):
        if key not in parsed:
            raise ValueError(f"missing key '{key}' in model response")

    return parsed


def classify_one(client: Anthropic, complaint: dict, prompt_version: int = 1) -> dict:
    """
    Classify a single complaint. Returns a dict matching CSV_COLUMNS.
    Never raises — any failure is captured as a PARSE_ERROR row so one bad
    complaint can't take down the whole batch run.
    """
    narrative = complaint["complaint_what_happened"]
    true_product = complaint["product"]

    prompt = build_prompt(narrative, prompt_version=prompt_version)

    # Retry loop with exponential backoff, specifically for rate-limit errors.
    # Other errors (bad request, auth, etc.) aren't retried — retrying those
    # would just fail the same way five times in a row.
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            break
        except RateLimitError as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            time.sleep(wait)
    else:
        # Loop exhausted all retries without breaking.
        return {
            "complaint_id": complaint["complaint_id"],
            "narrative": narrative,
            "true_product": true_product,
            "predicted_product": "PARSE_ERROR",
            "urgency": "PARSE_ERROR",
            "reason": f"rate-limited after {MAX_RETRIES} retries: {last_error}",
            "correct": False,
        }

    # Try to parse the model's JSON reply.
    try:
        parsed = parse_response(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        return {
            "complaint_id": complaint["complaint_id"],
            "narrative": narrative,
            "true_product": true_product,
            "predicted_product": "PARSE_ERROR",
            "urgency": "PARSE_ERROR",
            "reason": f"parse failure: {e}. raw={raw_text[:200]!r}",
            "correct": False,
        }

    # Normalize both sides through the same taxonomy so formatting quirks
    # (case, whitespace) don't cause false "incorrect" results.
    predicted_norm = normalize_product(parsed["product"]) or parsed["product"]
    true_norm = normalize_product(true_product) or true_product

    return {
        "complaint_id": complaint["complaint_id"],
        "narrative": narrative,
        "true_product": true_product,
        "predicted_product": predicted_norm,
        "urgency": parsed.get("urgency", ""),
        "reason": parsed.get("reason", ""),
        "correct": predicted_norm == true_norm,
    }


def load_already_classified_ids(results_file: Path) -> set[str]:
    """
    Read a results CSV (if it exists) and return the set of complaint_ids
    already present, so we can skip them. This is what makes the script
    safe to re-run after a crash — it picks up where it left off instead
    of re-classifying (and re-paying for) everything.
    """
    if not results_file.exists():
        return set()

    with open(results_file, newline="") as f:
        reader = csv.DictReader(f)
        return {row["complaint_id"] for row in reader}


def ensure_csv_header(results_file: Path):
    """Create a results CSV with a header row if it doesn't exist yet."""
    if not results_file.exists():
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def run(dataset: str = "raw", prompt_version: int = 1, limit: int | None = None):
    """
    Classify complaints and write results incrementally.

    dataset: "raw" (the original random sample) or "balanced" (the
             stratified per-category sample).
    prompt_version: 1 (baseline) or 2 (adds the Debt collection/Credit
             reporting disambiguation rule + few-shot examples — see
             build_prompt above). Only "balanced" has a v2 results file
             defined in RESULTS_FILES; raw+v2 isn't something we classify
             (see the comment above RESULTS_FILES).
    limit: if given, only classify the first N *unclassified* complaints.
    Used for the small test run (e.g. limit=20) before scaling up.
    """
    load_dotenv()  # reads .env into environment variables
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY not found. Copy .env.example to .env "
            "and add your key from https://console.anthropic.com/"
        )

    client = Anthropic(api_key=api_key)

    key = (dataset, prompt_version)
    if key not in RESULTS_FILES:
        sys.exit(
            f"ERROR: no results file configured for dataset={dataset!r}, "
            f"prompt_version={prompt_version}. Valid combinations: "
            f"{list(RESULTS_FILES.keys())}"
        )
    results_file = RESULTS_FILES[key]
    load_complaints = DATASET_LOADERS[dataset]

    all_complaints = load_complaints()
    already_done = load_already_classified_ids(results_file)

    # Filter out anything already in the results file, so re-running is cheap and safe.
    todo = [c for c in all_complaints if str(c["complaint_id"]) not in already_done]

    if limit is not None:
        todo = todo[:limit]

    if not todo:
        print("Nothing to do — all complaints already classified (or limit=0).")
        return

    print(f"Classifying {len(todo)} complaints from the '{dataset}' dataset "
          f"with prompt v{prompt_version} "
          f"({len(already_done)} already done, skipping those)...")
    print(f"Writing to {results_file}")

    ensure_csv_header(results_file)

    # We open the CSV once and append rows as each future completes, using a
    # ThreadPoolExecutor for concurrency. Writing incrementally (not batching
    # results in memory) is what protects us from losing progress on a crash.
    with open(results_file, "a", newline="") as f, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        futures = {
            pool.submit(classify_one, client, c, prompt_version): c for c in todo
        }

        parse_errors = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            result = future.result()  # classify_one never raises, so this is safe
            writer.writerow(result)
            f.flush()  # push to disk immediately so a crash doesn't lose this row
            if result["predicted_product"] == "PARSE_ERROR":
                parse_errors += 1

    print(f"Done. Wrote {len(todo)} rows to {results_file}")
    if parse_errors:
        print(f"  ({parse_errors} rows were PARSE_ERROR — see evaluate.py for details)")


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify CFPB complaints with Claude.")
    parser.add_argument(
        "limit", nargs="?", type=int, default=None,
        help="Only classify the first N unclassified complaints (e.g. `20` for a quick test).",
    )
    parser.add_argument(
        "--dataset", choices=["raw", "balanced"], default="raw",
        help="Which complaint set to classify: 'raw' (original random sample) "
             "or 'balanced' (stratified per-category sample). Default: raw.",
    )
    parser.add_argument(
        "--prompt-version", type=int, choices=[1, 2], default=1,
        help="1 = baseline prompt. 2 = adds the Debt collection/Credit "
             "reporting disambiguation rule + few-shot examples. Default: 1.",
    )
    args = parser.parse_args()

    run(dataset=args.dataset, prompt_version=args.prompt_version, limit=args.limit)
