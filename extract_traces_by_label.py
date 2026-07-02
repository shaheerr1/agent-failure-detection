"""
extract_traces_by_label.py
────────────────────────────
Pulls full trace details from trace_annotation_log.xlsx for specific labels,
so you can inspect the actual task/steps/reasoning behind rare or
reclassified traces without manually scrolling the spreadsheet.

Usage examples:
    # All traces currently verified as GOAL_DRIFT
    python extract_traces_by_label.py --verified GOAL_DRIFT

    # All traces that were ORIGINALLY labelled TOOL_MISUSE by the auto-labeller,
    # regardless of what they got reclassified to (this is what you need to
    # see WHY your 2 TOOL_MISUSE traces didn't survive review)
    python extract_traces_by_label.py --original TOOL_MISUSE

    # Traces that were originally X but got reclassified to Y
    python extract_traces_by_label.py --original TOOL_MISUSE --verified LOOP

    # Save to a text file instead of printing to console
    python extract_traces_by_label.py --original TOOL_MISUSE --out tool_misuse_review.txt

Requires: pandas, openpyxl
    pip install pandas openpyxl
"""

import argparse
from pathlib import Path
import pandas as pd

ORIGINAL_COL = "Original Label"
VERIFIED_COL = "Verified Label"


def main():
    parser = argparse.ArgumentParser(description="Extract full trace rows by label from the Excel log.")
    parser.add_argument("--excel", type=str, default="trace_annotation_log.xlsx")
    parser.add_argument("--original", type=str, default=None,
                         help="Filter by Original Label (e.g. TOOL_MISUSE)")
    parser.add_argument("--verified", type=str, default=None,
                         help="Filter by Verified Label (e.g. GOAL_DRIFT)")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional: write output to this text file instead of printing")
    args = parser.parse_args()

    if not args.original and not args.verified:
        print("[!] Provide at least --original or --verified to filter by.")
        return

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"[!] File not found: {excel_path.resolve()}")
        return

    df = pd.read_excel(excel_path)

    mask = pd.Series([True] * len(df))
    if args.original:
        mask &= df[ORIGINAL_COL].astype(str).str.strip().str.upper() == args.original.upper()
    if args.verified:
        mask &= df[VERIFIED_COL].astype(str).str.strip().str.upper() == args.verified.upper()

    matches = df[mask]

    if matches.empty:
        print("No matching rows found.")
        return

    lines = []
    lines.append(f"Found {len(matches)} matching trace(s).\n")

    for i, (_, row) in enumerate(matches.iterrows(), 1):
        lines.append("═" * 90)
        lines.append(f"[{i}/{len(matches)}]  Trace ID: {row.get('Trace ID', 'N/A')}")
        lines.append(f"Original Label: {row.get('Original Label', 'N/A')}   "
                      f"Verified Label: {row.get('Verified Label', 'N/A')}   "
                      f"Confidence: {row.get('Confidence', 'N/A')}")
        lines.append("─" * 90)
        lines.append(f"KEY EVIDENCE:\n{row.get('Key Evidence', '')}\n")
        lines.append(f"FAILURE PATTERN:\n{row.get('Failure Pattern', '')}\n")
        lines.append(f"EVAL NOTES:\n{row.get('Eval Notes', '')}\n")
        lines.append(f"TRACE CONTENT:\n{row.get('Trace Content', '')}\n")

    output = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {len(matches)} trace(s) to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()