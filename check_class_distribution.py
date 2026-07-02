"""
check_class_distribution.py
─────────────────────────────
Reads trace_annotation_log.xlsx ONLY (the authoritative, most up-to-date
source) and reports:
  1. Class distribution using `Verified Label` — the human-confirmed,
     authoritative label used for DeBERTa training.
  2. Class distribution using `Original Label` — the original auto-labeller
     output, for comparison.
  3. A reclassification breakdown: how many traces changed label during
     human review, and exactly which original->verified transitions
     happened. This quantifies your auto-labeller's real-world error rate.

Usage:
    python check_class_distribution.py
    python check_class_distribution.py --excel /full/path/to/trace_annotation_log.xlsx

Requires: pandas, openpyxl
    pip install pandas openpyxl
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

LOW_SUPPORT_THRESHOLD = 10

LABEL_COL_CANDIDATES = ["Original Label", "label", "Label", "auto_label"]
FINAL_LABEL_COL_CANDIDATES = ["Verified Label", "final_label", "Final Label", "label_human"]


def find_column(df, candidates, name):
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower().strip(): c for c in df.columns}
    for col in candidates:
        if col.lower().strip() in lower_map:
            return lower_map[col.lower().strip()]
    raise ValueError(
        f"Could not find a '{name}' column.\n"
        f"Columns actually found: {list(df.columns)}\n"
        f"Add your real column name to the candidates list at the top of this script."
    )


def clean_labels(series):
    return series.astype(str).str.strip().str.upper().replace({"NAN": None}).dropna()


def print_table(title, counts: Counter):
    total = sum(counts.values())
    print(f"\n  {title}")
    print("  " + "─" * 44)
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        flag = "  ⚠ LOW SUPPORT" if 0 < count < LOW_SUPPORT_THRESHOLD else ""
        print(f"    {cls:<20}{count:>8}{flag}")
    print("  " + "─" * 44)
    print(f"    {'TOTAL':<20}{total:>8}")


def main():
    parser = argparse.ArgumentParser(description="Report class distribution from trace_annotation_log.xlsx only.")
    parser.add_argument("--excel", type=str, default="trace_annotation_log.xlsx",
                         help="Path to trace_annotation_log.xlsx")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"[!] File not found: {excel_path.resolve()}")
        print(f"    Run this from the folder containing trace_annotation_log.xlsx, "
              f"or pass --excel /full/path/to/trace_annotation_log.xlsx")
        return

    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df)} rows from {excel_path.resolve()}")

    label_col = find_column(df, LABEL_COL_CANDIDATES, "Original Label")
    final_col = find_column(df, FINAL_LABEL_COL_CANDIDATES, "Verified Label")
    print(f"  Using '{label_col}' as original auto-label column")
    print(f"  Using '{final_col}' as authoritative final-label column")

    auto_counts = Counter(clean_labels(df[label_col]))
    final_counts = Counter(clean_labels(df[final_col]))

    print("\n" + "═" * 50)
    print("  CLASS DISTRIBUTION — trace_annotation_log.xlsx")
    print("═" * 50)

    print_table("AUTHORITATIVE (Verified Label) — use this for training/reporting", final_counts)
    print_table("ORIGINAL AUTO-LABEL (Original Label) — for comparison only", auto_counts)

    both = df[[label_col, final_col]].copy()
    both[label_col] = clean_labels(both[label_col])
    both[final_col] = clean_labels(both[final_col])
    both = both.dropna()
    changed = both[both[label_col] != both[final_col]]

    print("\n" + "═" * 50)
    print("  RECLASSIFICATION SUMMARY")
    print("═" * 50)
    print(f"  Total rows compared: {len(both)}")
    if len(both):
        print(f"  Reclassified during human review: {len(changed)} ({len(changed) / len(both) * 100:.1f}%)")
    else:
        print("  Reclassified: 0")

    if len(changed) > 0:
        print("\n  Breakdown of changes (Original Label -> Verified Label):")
        change_pairs = Counter(zip(changed[label_col], changed[final_col]))
        for (orig, new), count in sorted(change_pairs.items(), key=lambda x: -x[1]):
            print(f"    {orig:<15} -> {new:<15} : {count}")

    low_support = {c: n for c, n in final_counts.items() if 0 < n < LOW_SUPPORT_THRESHOLD}
    if low_support:
        print("\n" + "═" * 50)
        print("  ⚠ SEVERE CLASS IMBALANCE (based on Verified Label)")
        print("═" * 50)
        for cls, count in sorted(low_support.items(), key=lambda x: x[1]):
            print(f"    {cls}: {count} example(s) — target 25-30+ before DeBERTa fine-tuning")

    print()


if __name__ == "__main__":
    main()