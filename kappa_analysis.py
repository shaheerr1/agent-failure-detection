
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    sys.exit("Missing scikit-learn. Install with:  pip install scikit-learn matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["SUCCESS", "HALLUCINATION", "LOOP", "UNSAFE_EXECUTION",
           "TOOL_MISUSE", "GOAL_DRIFT", "DISPUTED"]


def norm(label):
    if label is None:
        return None
    s = str(label).strip().upper().replace(" ", "_").replace("-", "_")
    return s if s and s not in {"NONE", "NULL", "NA", "N/A", ""} else None


def load_traces(root: Path):
    files = sorted(root.rglob("*.json"))
    if not files:
        sys.exit(f"No .json files found under {root}")
    rows, skipped = [], 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            skipped += 1
            continue
        if not isinstance(d, dict) or "trace_id" not in d:
            skipped += 1
            continue
        rows.append({
            "trace_id": d.get("trace_id"),
            "claude":   norm(d.get("label_claude")),
            "gpt":      norm(d.get("label_gpt4o") or d.get("label_gpt")),
            "human":    norm(d.get("label_human")),
            "final":    norm(d.get("final_label")),
            "reviewed": bool(d.get("reviewed", False)),
            "folder":   fp.parent.name,
            "file":     fp.name,
        })
    return rows, skipped


def kappa_for(rows, key_a, key_b):
    pairs = [(r[key_a], r[key_b]) for r in rows if r[key_a] and r[key_b]]
    if len(pairs) < 2:
        return None, 0, [], []
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return None, len(pairs), a, b
    return cohen_kappa_score(a, b, labels=labels), len(pairs), a, b


def raw_agreement(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0


def interpret(k):
    if k is None: return "undefined"
    if k < 0.00:  return "poor (worse than chance)"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial"
    return "almost perfect"


def per_class_agreement(rows, key_a, key_b):
    out = {}
    present = sorted({r[key_a] for r in rows if r[key_a]} |
                     {r[key_b] for r in rows if r[key_b]})
    for c in present:
        both = sum(1 for r in rows if r[key_a] == c and r[key_b] == c)
        either = sum(1 for r in rows if c in (r[key_a], r[key_b]))
        out[c] = (both, either, both / either if either else 0.0)
    return out


def plot_confusion(a, b, path, title):
    labels = [c for c in CLASSES if c in set(a) | set(b)]
    labels += sorted((set(a) | set(b)) - set(labels))
    cm = confusion_matrix(a, b, labels=labels)
    fig, ax = plt.subplots(figsize=(1.35 * len(labels) + 2.5,
                                    1.15 * len(labels) + 2.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_xlabel("GPT-4o annotator", fontsize=11)
    ax.set_ylabel("Claude annotator", fontsize=11)
    ax.set_title(title, fontsize=12, pad=14)
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "#1a1a1a")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Traces")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Folder containing labelled trace JSON files")
    ap.add_argument("--out", default="kappa_out", help="Output folder")
    ap.add_argument("--exclude", nargs="*", default=["TOOL_MISUSE", "GOAL_DRIFT"],
                    help="Final labels to exclude from the analysis")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    excluded_labels = {norm(x) for x in args.exclude}

    all_rows, skipped = load_traces(root)
    rows = [r for r in all_rows if r["final"] not in excluded_labels]
    dropped = [r for r in all_rows if r["final"] in excluded_labels]
    reviewed = [r for r in rows if r["reviewed"] and r["human"]]

    L = []
    L.append("=" * 68)
    L.append("INTER-ANNOTATOR AGREEMENT ANALYSIS")
    L.append("=" * 68)
    L.append(f"Source folder      : {root}")
    L.append(f"Trace files parsed : {len(all_rows)}   (skipped/unreadable: {skipped})")
    L.append(f"Excluded by label  : {len(dropped)}  {sorted(excluded_labels)}")
    L.append(f"Traces analysed    : {len(rows)}")
    L.append(f"Human-reviewed     : {len(reviewed)}")
    L.append("")

    if dropped:
        L.append("Excluded traces by source folder")
        L.append("-" * 68)
        for f, n in Counter(r["folder"] for r in dropped).most_common():
            L.append(f"  {f:<22} {n:>5}")
        L.append("")

    L.append("Final label distribution (analysed set)")
    L.append("-" * 68)
    total = len(rows) or 1
    for lab, n in Counter(r["final"] for r in rows if r["final"]).most_common():
        L.append(f"  {lab:<22} {n:>5}   ({n/total*100:5.1f}%)")
    L.append("")

    L.append("Source folder vs final label (relabelling audit)")
    L.append("-" * 68)
    moved = [r for r in all_rows if r["final"] and r["folder"]
             and r["folder"].upper() != r["final"]]
    L.append(f"  Traces whose final label differs from their folder: {len(moved)}")
    for (f, fin), n in Counter((r["folder"], r["final"]) for r in moved).most_common(15):
        L.append(f"    {f:<20} -> {fin:<20} {n:>4}")
    L.append("")

    results = {}

    k, n, a, b = kappa_for(rows, "claude", "gpt")
    if n:
        ra = raw_agreement(a, b)
        results["Claude vs GPT-4o (all analysed traces)"] = (k, ra, n)
        L.append("PRIMARY MEASURE - Claude vs GPT-4o")
        L.append("-" * 68)
        L.append(f"  Traces compared  : {n}")
        L.append(f"  Raw agreement    : {ra:.3f}  ({int(round(ra*n))} of {n})")
        L.append(f"  Cohen's kappa    : {k:.3f}" if k is not None
                 else "  Cohen's kappa    : undefined")
        L.append(f"  Interpretation   : {interpret(k)}  (Landis & Koch, 1977)")
        L.append("")
        L.append("  Per-class concordance (both annotators chose this class):")
        for c, (both, either, frac) in per_class_agreement(rows, "claude", "gpt").items():
            L.append(f"    {c:<22} {both:>4} / {either:<4}  = {frac:.3f}")
        L.append("")
        plot_confusion(a, b, outdir / "confusion_claude_gpt.png",
                       "Annotator concordance: Claude vs GPT-4o")

    for key, name in (("claude", "Claude"), ("gpt", "GPT-4o")):
        k2, n2, a2, b2 = kappa_for(reviewed, key, "human")
        if n2:
            ra2 = raw_agreement(a2, b2)
            results[f"{name} vs human (reviewed subset)"] = (k2, ra2, n2)
            L.append(f"SECONDARY - {name} vs human-adjudicated gold (reviewed only)")
            L.append("-" * 68)
            L.append(f"  Traces compared  : {n2}")
            L.append(f"  Raw agreement    : {ra2:.3f}")
            L.append(f"  Cohen's kappa    : {k2:.3f}" if k2 is not None
                     else "  Cohen's kappa    : undefined")
            L.append(f"  Interpretation   : {interpret(k2)}")
            L.append("")

    L.append("!! CAVEAT ON THE SECONDARY FIGURES")
    L.append("-" * 68)
    L.append("  The reviewed subset is enriched with traces the two automated")
    L.append("  annotators disagreed on, i.e. the hardest cases. Kappa against")
    L.append("  human gold on this subset is a LOWER BOUND, not an estimate of")
    L.append("  pipeline accuracy over the whole dataset. Report it as such.")
    L.append("  The Claude-vs-GPT figure is the clean, unbiased measure of")
    L.append("  taxonomy reliability and should be the headline.")
    L.append("")

    changed = [r for r in reviewed if r["human"]
               and r["human"] != r["claude"] and r["human"] != r["gpt"]]
    L.append("Adjudication impact")
    L.append("-" * 68)
    L.append(f"  Reviewed traces where the human label differs from BOTH")
    L.append(f"  automated annotators: {len(changed)}")
    L.append("  (Evidence the human stage did substantive work.)")
    L.append("")
    L.append("=" * 68)

    report = "\n".join(L)
    print(report)
    (outdir / "kappa_summary.txt").write_text(report, encoding="utf-8")

    with open(outdir / "agreement_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trace_id", "claude", "gpt", "human",
                                           "final", "reviewed", "folder", "file"])
        w.writeheader()
        w.writerows(all_rows)

    tex = [r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{Inter-annotator agreement for the trace annotation pipeline.}",
           r"\label{tab:kappa}",
           r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
           r"\textbf{Annotator pair} & \textbf{$n$} & \textbf{Raw agr.} "
           r"& \textbf{Cohen's $\kappa$} \\", r"\midrule"]
    for name, (k_, ra_, n_) in results.items():
        kk = f"{k_:.3f}" if k_ is not None else "--"
        tex.append(name.replace("&", r"\&") + f" & {n_} & {ra_:.3f} & {kk}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "kappa_table.tex").write_text("\n".join(tex), encoding="utf-8")

    print(f"\nWritten to {outdir.resolve()}/")
    for f in ["kappa_summary.txt", "agreement_pairs.csv",
              "confusion_claude_gpt.png", "kappa_table.tex"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()