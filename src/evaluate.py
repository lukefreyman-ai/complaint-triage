"""
evaluate.py — Score classify.py's output against the CFPB ground truth.

For each results CSV that exists, this prints:
  - Overall accuracy
  - Majority-class baseline (accuracy from always guessing the most common
    category) and "lift" — how much better the model does than that baseline
  - Per-category accuracy, with sample sizes
  - The 10 most common misclassification pairs
  - Urgency rating distribution
  - Count of PARSE_ERROR rows

...and saves a confusion matrix PNG per dataset into charts/.

Why "lift over baseline" matters more than raw accuracy: on an imbalanced
dataset (like the original random sample, ~84% one category), a model that
always guesses the majority category scores ~84% while doing zero real
classification work. Raw accuracy alone can't tell the difference between
"the model is smart" and "the dataset is lopsided." Lift = accuracy minus
baseline is what's actually left over once you subtract out that free score.

Run it any time after classify.py has produced at least some rows — you
don't need a full dataset classified to get a useful read.
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — we're not showing plots interactively
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
CHARTS_DIR = ROOT / "charts"

# Every results file we know how to evaluate, and a short label used in
# printed output and chart filenames. classify.py writes these paths — see
# RESULTS_FILES in that file.
DATASETS = {
    "original": {
        "results_file": ROOT / "data" / "results.csv",
        "label": "Original (random sample, ~84% one category)",
        "chart_suffix": "original",
    },
    "balanced": {
        "results_file": ROOT / "data" / "results_balanced.csv",
        "label": "Balanced (stratified, ~60/category), prompt v1",
        "chart_suffix": "balanced",
    },
    "balanced_v2": {
        "results_file": ROOT / "data" / "results_balanced_v2.csv",
        "label": "Balanced (stratified, ~60/category), prompt v2 (disambiguation rule)",
        "chart_suffix": "balanced_v2",
    },
}


def load_results(results_file: Path) -> pd.DataFrame:
    if not results_file.exists():
        return None
    df = pd.read_csv(results_file)
    # "correct" was written as the string "True"/"False" by Python's csv
    # module (bool doesn't round-trip through CSV), so convert it back.
    df["correct"] = df["correct"].astype(str).map({"True": True, "False": False})
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute the headline numbers for one results dataframe: accuracy,
    majority-class baseline, and lift (accuracy - baseline). Excludes
    PARSE_ERROR rows from all of these — a parse failure isn't a
    classification decision, right or wrong.
    """
    parse_errors = int((df["predicted_product"] == "PARSE_ERROR").sum())
    scored = df[df["predicted_product"] != "PARSE_ERROR"]

    accuracy = scored["correct"].mean() if len(scored) else float("nan")

    # Baseline = accuracy from always guessing the single most common TRUE
    # category. This is computed on the same `scored` rows so it's an
    # apples-to-apples comparison with `accuracy`.
    true_counts = Counter(scored["true_product"])
    majority_product, majority_count = true_counts.most_common(1)[0]
    baseline = majority_count / len(scored) if len(scored) else float("nan")

    return {
        "n_total": len(df),
        "n_scored": len(scored),
        "parse_errors": parse_errors,
        "accuracy": accuracy,
        "majority_product": majority_product,
        "baseline": baseline,
        "lift": accuracy - baseline,
    }


def print_summary(summary: dict, label: str):
    print(f"── {label} ──")
    print(f"  Total rows:              {summary['n_total']}")
    print(f"  Parse errors:            {summary['parse_errors']}")
    print(f"  Scored rows:             {summary['n_scored']}")
    print(f"  Accuracy:                {summary['accuracy']:.1%}")
    print(f"  Majority-class baseline: {summary['baseline']:.1%}  "
          f"(always guess {summary['majority_product']!r})")
    lift_sign = "+" if summary["lift"] >= 0 else ""
    print(f"  >>> LIFT OVER BASELINE:  {lift_sign}{summary['lift']:.1%} <<<")
    print()


def print_per_category_accuracy(df: pd.DataFrame):
    scored = df[df["predicted_product"] != "PARSE_ERROR"]

    print("  Per-category accuracy (sorted worst to best):")
    grouped = scored.groupby("true_product")["correct"].agg(["mean", "count"])
    grouped = grouped.sort_values("mean")
    for product, row in grouped.iterrows():
        print(f"    {row['mean']:>6.1%}  (n={int(row['count']):>3})  {product}")
    print()


def print_top_misclassifications(df: pd.DataFrame, top_n: int = 10):
    wrong = df[(df["correct"] == False) & (df["predicted_product"] != "PARSE_ERROR")]
    pairs = Counter(zip(wrong["true_product"], wrong["predicted_product"]))

    print(f"  Top {top_n} misclassification pairs (true -> predicted):")
    for (true_p, pred_p), count in pairs.most_common(top_n):
        print(f"    {count:>3}x   {true_p}  ->  {pred_p}")
    print()


def print_urgency_distribution(df: pd.DataFrame):
    scored = df[df["predicted_product"] != "PARSE_ERROR"]
    print("  Urgency rating distribution:")
    counts = scored["urgency"].value_counts()
    for level, count in counts.items():
        print(f"    {count:>4}  {level}")
    print()


def save_confusion_matrix(df: pd.DataFrame, chart_suffix: str, title_label: str):
    scored = df[df["predicted_product"] != "PARSE_ERROR"]
    if scored.empty:
        print("No scored rows — skipping confusion matrix.")
        return

    # Labels = every product that appears as either a true or predicted value,
    # so the matrix is square and covers everything actually seen.
    labels = sorted(set(scored["true_product"]) | set(scored["predicted_product"]))

    cm = confusion_matrix(scored["true_product"], scored["predicted_product"], labels=labels)

    # Shorten labels for the plot — the full CFPB product names are long and
    # would make the axis ticks unreadable.
    short_labels = [l if len(l) <= 25 else l[:22] + "..." for l in labels]

    CHARTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    plt.xlabel("Predicted product")
    plt.ylabel("True product (CFPB label)")
    plt.title(f"Confusion matrix — {title_label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = CHARTS_DIR / f"confusion_matrix_{chart_suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  Saved confusion matrix to {out_path}")


def save_urgency_chart(df: pd.DataFrame, chart_suffix: str, title_label: str):
    scored = df[df["predicted_product"] != "PARSE_ERROR"]
    if scored.empty:
        return

    CHARTS_DIR.mkdir(exist_ok=True)

    order = ["low", "medium", "high"]
    counts = scored["urgency"].value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, color="#4C72B0")
    plt.xlabel("Urgency rating")
    plt.ylabel("Count")
    plt.title(f"Urgency ratings — {title_label}")
    plt.tight_layout()

    out_path = CHARTS_DIR / f"urgency_distribution_{chart_suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  Saved urgency chart to {out_path}")


def evaluate_dataset(key: str) -> dict | None:
    """Run the full evaluation for one dataset and return its summary dict
    (or None if that dataset's results file doesn't exist yet)."""
    config = DATASETS[key]
    df = load_results(config["results_file"])
    if df is None:
        print(f"── {config['label']} ──")
        print(f"  {config['results_file']} not found — skipping "
              f"(run classify.py for this dataset first).\n")
        return None

    summary = compute_summary(df)
    print_summary(summary, config["label"])
    print_per_category_accuracy(df)
    print_top_misclassifications(df)
    print_urgency_distribution(df)
    save_confusion_matrix(df, config["chart_suffix"], config["label"])
    save_urgency_chart(df, config["chart_suffix"], config["label"])
    print()

    summary["key"] = key
    return summary


def print_side_by_side(summaries: dict[str, dict]):
    """Print a compact side-by-side comparison table across every dataset
    that was actually evaluated (skips ones with no results file yet)."""
    if len(summaries) < 2:
        return  # nothing to compare

    print("═" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("═" * 70)

    header = f"  {'metric':<26}" + "".join(
        f"{DATASETS[key]['chart_suffix']:>20}" for key in summaries
    )
    print(header)

    def row(label, values):
        print(f"  {label:<26}" + "".join(f"{v:>20}" for v in values))

    row("accuracy", [f"{s['accuracy']:.1%}" for s in summaries.values()])
    row("majority baseline", [f"{s['baseline']:.1%}" for s in summaries.values()])
    row(">>> LIFT OVER BASELINE <<<", [f"{s['lift']:+.1%}" for s in summaries.values()])
    row("scored rows", [str(s["n_scored"]) for s in summaries.values()])
    row("parse errors", [str(s["parse_errors"]) for s in summaries.values()])
    print("═" * 70)
    print()


def print_prompt_comparison():
    """
    Phase 3's headline output: does the disambiguation prompt (v2) actually
    fix the Debt collection <-> Credit reporting confusion found in Phase 2?
    Compares "balanced" (prompt v1) against "balanced_v2" (prompt v2) on the
    exact same underlying complaints, per category, with the accuracy delta
    highlighted so the effect (or lack of one) is immediately visible.
    """
    df_v1 = load_results(DATASETS["balanced"]["results_file"])
    df_v2 = load_results(DATASETS["balanced_v2"]["results_file"])
    if df_v1 is None or df_v2 is None:
        return  # one or both prompt runs haven't been done yet

    scored_v1 = df_v1[df_v1["predicted_product"] != "PARSE_ERROR"]
    scored_v2 = df_v2[df_v2["predicted_product"] != "PARSE_ERROR"]

    acc_v1 = scored_v1.groupby("true_product")["correct"].mean()
    acc_v2 = scored_v2.groupby("true_product")["correct"].mean()
    n_v1 = scored_v1.groupby("true_product")["correct"].count()

    print("═" * 78)
    print("PROMPT EXPERIMENT: v1 (baseline) vs. v2 (disambiguation rule + few-shot)")
    print("Same balanced dataset, same complaints — only the prompt changed.")
    print("═" * 78)

    categories = sorted(acc_v1.index, key=lambda c: acc_v2.get(c, 0) - acc_v1.get(c, 0))
    print(f"  {'category':<55}{'v1':>8}{'v2':>8}{'delta':>9}")
    for cat in categories:
        v1 = acc_v1.get(cat, float("nan"))
        v2 = acc_v2.get(cat, float("nan"))
        delta = v2 - v1
        marker = "  <-- target of the fix" if cat == "Debt collection" else ""
        print(f"  {cat:<55}{v1:>7.1%}{v2:>8.1%}{delta:>+8.1%}{marker}")

    overall_v1 = scored_v1["correct"].mean()
    overall_v2 = scored_v2["correct"].mean()
    print(f"\n  {'OVERALL':<55}{overall_v1:>7.1%}{overall_v2:>8.1%}"
          f"{overall_v2 - overall_v1:>+8.1%}")
    print("═" * 78)
    print()


def run():
    summaries = {}
    for key in DATASETS:
        summary = evaluate_dataset(key)
        if summary is not None:
            summaries[key] = summary

    print_side_by_side(summaries)
    print_prompt_comparison()


if __name__ == "__main__":
    run()
