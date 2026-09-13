"""
app.py — Streamlit UI for the complaint-triage project.

Three tabs:
  1. Try it     — classify any narrative you paste in, live.
  2. Results     — the accuracy scoreboard, confusion matrix, filterable table.
  3. Failures    — only the misclassified rows, to read where the model breaks.

Run with: streamlit run app.py

Styling: near-black/white, one accent color, no gradients/emoji, generous
whitespace, tabular numerals for metrics — all done via a small CSS block
injected once at the top (Streamlit doesn't expose these as native theme
options, so raw CSS is the simplest way to get exact control).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Make src/ importable so we can reuse the same classification + taxonomy
# logic the batch scripts use — we don't want the live "Try it" tab to
# duplicate that logic and risk drifting out of sync.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from taxonomy import VALID_PRODUCTS, normalize_product  # noqa: E402
from classify import build_prompt, parse_response, MODEL_NAME  # noqa: E402
from anthropic import Anthropic  # noqa: E402

# The three classification runs we can show, keyed by the label shown in the
# UI toggle. Mirrors DATASETS in src/evaluate.py — kept as a separate dict
# here (rather than importing evaluate.py's) since the app only needs the
# file paths, not evaluate.py's printing/charting logic.
DATASET_OPTIONS = {
    "Original (random sample)": {
        "results_file": ROOT / "data" / "results.csv",
        "chart_suffix": "original",
    },
    "Balanced — prompt v1": {
        "results_file": ROOT / "data" / "results_balanced.csv",
        "chart_suffix": "balanced",
    },
    "Balanced — prompt v2 (disambiguation)": {
        "results_file": ROOT / "data" / "results_balanced_v2.csv",
        "chart_suffix": "balanced_v2",
    },
}
CHARTS_DIR = ROOT / "charts"

ACCENT = "#2563eb"  # single accent color, used sparingly

st.set_page_config(page_title="Complaint Triage", layout="wide")

# ── Global styling ───────────────────────────────────────────────────────────
# Minimal, sharp: near-black/white palette, one accent, no gradients/emoji,
# generous whitespace, and tabular-nums so metric digits line up vertically.
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-variant-numeric: tabular-nums;
    }}
    .stApp {{
        background-color: #ffffff;
        color: #111111;
    }}
    h1, h2, h3 {{
        color: #111111;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #555555;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums;
        color: #111111;
    }}
    .stButton button {{
        background-color: {ACCENT};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton button:hover {{
        background-color: #1d4ed8;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(results_file_str: str) -> pd.DataFrame:
    """Load a results CSV once and cache it per file path across reruns
    (Streamlit reruns the whole script on every interaction, so caching
    avoids re-reading the CSV from disk each time). Takes a string rather
    than a Path because st.cache_data keys its cache off the argument value,
    and Path objects hash fine but a str is simplest/most explicit here."""
    results_file = Path(results_file_str)
    if not results_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(results_file)
    df["correct"] = df["correct"].astype(str).map({"True": True, "False": False})
    return df


def compute_baseline_and_lift(df: pd.DataFrame) -> tuple[float, float, float, str]:
    """
    Returns (accuracy, baseline, lift, majority_product) for one results
    dataframe. baseline = accuracy from always guessing the single most
    common true category — see the comment at the top of src/evaluate.py
    for why this matters more than raw accuracy on an imbalanced dataset.
    """
    scored = df[df["predicted_product"] != "PARSE_ERROR"]
    if scored.empty:
        return float("nan"), float("nan"), float("nan"), ""

    accuracy = scored["correct"].mean()
    true_counts = scored["true_product"].value_counts()
    majority_product = true_counts.index[0]
    baseline = true_counts.iloc[0] / len(scored)
    return accuracy, baseline, accuracy - baseline, majority_product


def get_client() -> Anthropic | None:
    """Load the API key and build a client. Returns None if no key is set,
    so the 'Try it' tab can show a friendly message instead of crashing."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


st.title("Complaint Triage")
st.caption(
    "Benchmarking an LLM's ability to classify consumer financial complaints "
    "against the CFPB's real, human-labeled categories."
)

# One dataset selector, shared by the Results and Failures tabs below. Defined
# once here (rather than as a widget inside each tab) because Streamlit
# renders every tab's content on every rerun — creating the same widget twice
# with the same key would raise a duplicate-widget-ID error.
dataset_choice = st.selectbox(
    "Dataset",
    options=list(DATASET_OPTIONS.keys()),
    index=0,
    help="Compare the three classification runs: the original class-imbalanced "
         "sample, the balanced sample, and the balanced sample with the v2 "
         "disambiguation prompt.",
)
selected = DATASET_OPTIONS[dataset_choice]

tab_try, tab_results, tab_failures = st.tabs(["Try it", "Results", "Failures"])


# ── Tab 1: Try it ────────────────────────────────────────────────────────────
with tab_try:
    st.subheader("Classify a complaint narrative")
    st.write(
        "Paste any consumer complaint text below. The model will pick one "
        f"of the {len(VALID_PRODUCTS)} CFPB product categories, rate urgency, "
        "and explain its reasoning."
    )

    narrative_input = st.text_area(
        "Complaint narrative",
        height=180,
        placeholder="e.g. \"I have been trying to dispute an incorrect late "
                    "payment on my credit report for three months...\"",
    )

    if st.button("Classify", type="primary"):
        if not narrative_input.strip():
            st.warning("Enter some complaint text first.")
        else:
            client = get_client()
            if client is None:
                st.error(
                    "No ANTHROPIC_API_KEY found. Copy .env.example to .env "
                    "and add your key."
                )
            else:
                with st.spinner(f"Classifying with {MODEL_NAME}..."):
                    prompt = build_prompt(narrative_input)
                    response = client.messages.create(
                        model=MODEL_NAME,
                        max_tokens=300,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_text = response.content[0].text
                    try:
                        parsed = parse_response(raw_text)
                        product = normalize_product(parsed["product"]) or parsed["product"]
                        urgency = parsed.get("urgency", "")
                        reason = parsed.get("reason", "")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted product", product)
                        with col2:
                            st.metric("Urgency", urgency.capitalize())
                        st.write("**Reasoning:**", reason)
                    except (ValueError, Exception) as e:
                        st.error(f"Could not parse model response: {e}")
                        st.code(raw_text)


# ── Tab 2: Results ───────────────────────────────────────────────────────────
with tab_results:
    df = load_results(str(selected["results_file"]))

    if df.empty:
        st.info(
            f"No results yet for '{dataset_choice}' — run "
            f"`python src/classify.py` (see README for the exact command "
            f"for this dataset) first."
        )
    else:
        parse_errors = (df["predicted_product"] == "PARSE_ERROR").sum()
        accuracy, baseline, lift, majority_product = compute_baseline_and_lift(df)

        # The headline comparison: accuracy is meaningless on its own on an
        # imbalanced dataset (see the Original run, where the majority-class
        # baseline is ~84%). st.metric's `delta` argument does exactly what
        # we want here — it shows accuracy with a colored +/- indicator
        # relative to the baseline, so the lift is impossible to miss.
        st.subheader("Accuracy vs. baseline")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Accuracy",
            f"{accuracy:.1%}",
            delta=f"{lift:+.1%} vs. baseline",
        )
        col2.metric(
            "Majority-class baseline",
            f"{baseline:.1%}",
            help=f"Accuracy from always guessing '{majority_product}' — "
                 f"the most common true category in this dataset.",
        )
        col3.metric("Complaints classified", f"{len(df):,}")
        if parse_errors:
            st.caption(f"{parse_errors} parse error rows excluded from scoring.")

        st.subheader("Confusion matrix")
        confusion_img = CHARTS_DIR / f"confusion_matrix_{selected['chart_suffix']}.png"
        if confusion_img.exists():
            st.image(str(confusion_img), use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate the confusion matrix.")

        st.subheader("Urgency distribution")
        urgency_img = CHARTS_DIR / f"urgency_distribution_{selected['chart_suffix']}.png"
        if urgency_img.exists():
            st.image(str(urgency_img))
        else:
            st.info("Run `python src/evaluate.py` to generate the urgency chart.")

        st.subheader("All classified complaints")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            product_filter = st.multiselect(
                "Filter by true product", sorted(df["true_product"].unique())
            )
        with col2:
            correctness_filter = st.selectbox(
                "Filter by result", ["All", "Correct only", "Incorrect only"]
            )

        filtered = df.copy()
        if product_filter:
            filtered = filtered[filtered["true_product"].isin(product_filter)]
        if correctness_filter == "Correct only":
            filtered = filtered[filtered["correct"] == True]
        elif correctness_filter == "Incorrect only":
            filtered = filtered[filtered["correct"] == False]

        def highlight_correctness(row):
            color = "#e6f4ea" if row["correct"] else "#fdecea"  # soft green / soft red
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            filtered.style.apply(highlight_correctness, axis=1),
            use_container_width=True,
            height=500,
        )


# ── Tab 3: Failures ──────────────────────────────────────────────────────────
with tab_failures:
    df = load_results(str(selected["results_file"]))

    if df.empty:
        st.info(f"No results yet for '{dataset_choice}' — run `python src/classify.py` first.")
    else:
        failures = df[df["correct"] == False]

        st.subheader(f"Misclassified complaints ({len(failures)})")
        st.write(
            "These are the complaints the model got wrong — read the "
            "narrative and reasoning to see where it breaks down."
        )

        if failures.empty:
            st.success("No misclassifications — the model got everything right.")
        else:
            for _, row in failures.iterrows():
                with st.expander(
                    f"#{row['complaint_id']} — true: {row['true_product']} "
                    f"→ predicted: {row['predicted_product']}"
                ):
                    st.write("**Narrative:**")
                    st.write(row["narrative"])
                    st.write("**Model's reasoning:**", row["reason"])
                    st.write("**Urgency assigned:**", row["urgency"])
