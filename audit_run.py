"""
audit_run.py - shortcut ablations and runtime false-positive rate.

Runs against the EXISTING trained checkpoint. No training, no new data.

Usage (from the repo root):
    python audit_run.py verify     # baseline test-set scores + bootstrap CIs
    python audit_run.py ablate     # baseline vs 4 ablation variants
    python audit_run.py fpr        # runtime false positive rate on SUCCESS traces
    python audit_run.py all        # all three

Options:
    --model  classifier/final_model      path to checkpoint
    --test   data_splits/test.xlsx       path to test split
    --out    results                     output directory

Writes results/*.json and results/summary.md. Paste summary.md back into the chat.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512

# --------------------------------------------------------------------------
# text transforms (the ablations)
# --------------------------------------------------------------------------

TOOL_ALIASES = {
    "purchase_mock":    "action_tool_a",
    "send_email_mock":  "action_tool_b",
    "delete_file_mock": "action_tool_c",
    "get_weather_mock": "info_tool_b",
}

MOCK_MARKERS = ["[MOCK ACTION LOG]", "[MOCK WEATHER]", "[MOCK]"]


def t_identity(text):
    return text


def t_mask_tools(text):
    """V1: rename the action tools so the label is not spelled out in the input."""
    for real, alias in TOOL_ALIASES.items():
        text = text.replace(real, alias)
    return text


def t_strip_mock_markers(text):
    """V2: remove the synthetic [MOCK...] markers the environment injects."""
    for m in MOCK_MARKERS:
        text = text.replace(m, "")
    return text


def t_normalise_final(text):
    """V3: drop the FINAL: line from every class, so truncation carries no signal."""
    i = text.find("\nFINAL:")
    if i == -1:
        i = text.find("FINAL:")
        if i == 0:
            return ""
        if i == -1:
            return text
    return text[:i].rstrip()


def t_all(text):
    return t_normalise_final(t_strip_mock_markers(t_mask_tools(text)))


VARIANTS = [
    ("baseline",              t_identity,            "unmodified trace text"),
    ("V1 tool names masked",  t_mask_tools,          "*_mock tool names aliased"),
    ("V2 mock markers gone",  t_strip_mock_markers,  "[MOCK] markers removed"),
    ("V3 FINAL: normalised",  t_normalise_final,     "FINAL: line dropped for ALL classes"),
    ("V4 all three",          t_all,                 "V1 + V2 + V3 combined"),
]

# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------


def head_tail_truncate(text, tokenizer, max_length=MAX_LEN):
    """Same head-and-tail strategy used in training."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    return tokenizer.decode(ids[:head] + ids[-(budget - head):])


def load_model(model_dir):
    print(f"[load] {model_dir}  (device={DEVICE})")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE)
    mdl.eval()
    return tok, mdl


def resolve_labels(model, df):
    """Prefer the checkpoint's id2label; fall back to the data's label_id mapping."""
    id2label = dict(model.config.id2label)
    if all(str(v).startswith("LABEL_") for v in id2label.values()):
        if "label_id" in df.columns:
            pairs = df[["label_id", "Label"]].drop_duplicates()
            id2label = {int(r.label_id): str(r.Label) for r in pairs.itertuples()}
            print("[labels] config had generic names; derived from data:", id2label)
    return {int(k): str(v) for k, v in id2label.items()}


@torch.no_grad()
def predict_batch(texts, tok, mdl, batch_size=16):
    """Returns (pred_ids, probs) for a list of raw texts."""
    preds, probs_all = [], []
    for i in range(0, len(texts), batch_size):
        chunk = [head_tail_truncate(t, tok) for t in texts[i:i + batch_size]]
        enc = tok(chunk, truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(DEVICE)
        logits = mdl(**enc).logits
        p = F.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(p)
        preds.extend(p.argmax(axis=1).tolist())
    return np.array(preds), np.vstack(probs_all)


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------


def score(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for i, lab in enumerate(labels):
        rows.append({"label": lab, "precision": round(float(p[i]), 3),
                     "recall": round(float(r[i]), 3), "f1": round(float(f[i]), 3),
                     "support": int(s[i])})
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return {"per_class": rows, "macro_f1": round(macro, 3), "accuracy": round(acc, 3)}


def bootstrap_macro_f1(y_true, y_pred, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_obs = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=labels,
                             average="macro", zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def fmt_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i])
                                     for i, c in enumerate(row)) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# command: verify
# --------------------------------------------------------------------------


def cmd_verify(df, tok, mdl, id2label, md):
    labels = [id2label[i] for i in sorted(id2label)]
    y_true = df["Label"].tolist()

    texts_raw = df["Trace Content"].astype(str).tolist()
    pred_ids, _ = predict_batch(texts_raw, tok, mdl)
    y_pred = [id2label[int(i)] for i in pred_ids]
    res = score(y_true, y_pred, labels)
    lo, hi = bootstrap_macro_f1(y_true, y_pred, labels)

    md.append("## 1. Baseline verification\n")
    md.append(f"Test set: {len(df)} traces. Device: {DEVICE}.\n")
    md.append(fmt_table(
        ["class", "precision", "recall", "f1", "support"],
        [[r["label"], r["precision"], r["recall"], r["f1"], r["support"]]
         for r in res["per_class"]]))
    md.append(f"\n**macro F1 = {res['macro_f1']}**  (95% bootstrap CI {lo} to {hi})  "
              f"| accuracy = {res['accuracy']}")
    md.append(f"\nDissertation reported macro F1 0.820, accuracy 0.816.\n")

    # fidelity check: does the pre-truncated column give the same answer?
    if "text_truncated" in df.columns:
        p2, _ = predict_batch(df["text_truncated"].astype(str).tolist(), tok, mdl)
        y2 = [id2label[int(i)] for i in p2]
        r2 = score(y_true, y2, labels)
        agree = float((np.array(y_pred) == np.array(y2)).mean())
        md.append(f"Pipeline fidelity check: scoring the stored `text_truncated` column "
                  f"gives macro F1 {r2['macro_f1']} and agrees with re-truncated raw text "
                  f"on {agree:.1%} of traces.\n")

    md.append("\nConfusion matrix (rows true, cols predicted), order "
              + ", ".join(labels) + ":\n```")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for lab, row in zip(labels, cm):
        md.append(f"{lab:18s} " + " ".join(f"{v:4d}" for v in row))
    md.append("```\n")
    return {"baseline": res, "macro_f1_ci": [lo, hi]}


# --------------------------------------------------------------------------
# command: ablate
# --------------------------------------------------------------------------


def cmd_ablate(df, tok, mdl, id2label, md):
    labels = [id2label[i] for i in sorted(id2label)]
    y_true = df["Label"].tolist()
    raw = df["Trace Content"].astype(str).tolist()

    results = {}
    per_class_rows = []
    macro_rows = []
    base_macro = None
    base_f1 = {}

    for name, fn, desc in VARIANTS:
        texts = [fn(t) for t in raw]
        n_changed = sum(1 for a, b in zip(raw, texts) if a != b)
        pred_ids, _ = predict_batch(texts, tok, mdl)
        y_pred = [id2label[int(i)] for i in pred_ids]
        res = score(y_true, y_pred, labels)
        results[name] = {"desc": desc, "traces_modified": n_changed, **res}

        f1s = {r["label"]: r["f1"] for r in res["per_class"]}
        if name == "baseline":
            base_macro = res["macro_f1"]
            base_f1 = f1s
            macro_rows.append([name, res["macro_f1"], "-", res["accuracy"], n_changed])
        else:
            d = round(res["macro_f1"] - base_macro, 3)
            macro_rows.append([name, res["macro_f1"], f"{d:+.3f}", res["accuracy"], n_changed])

        row = [name]
        for lab in labels:
            if name == "baseline":
                row.append(f"{f1s[lab]:.3f}")
            else:
                row.append(f"{f1s[lab]:.3f} ({f1s[lab]-base_f1[lab]:+.3f})")
        per_class_rows.append(row)

    md.append("## 2. Shortcut ablations\n")
    md.append("Same checkpoint, same test set, input text transformed.\n")
    md.append(fmt_table(["variant", "macro F1", "delta", "accuracy", "traces changed"],
                        macro_rows))
    md.append("\nPer-class F1, delta against baseline in brackets:\n")
    md.append(fmt_table(["variant"] + labels, per_class_rows))
    md.append("""
**How to read this.** A large negative delta on UNSAFE_EXECUTION under V1 means the
model was reading tool names rather than behaviour. A large negative delta on LOOP
under V3 means it was reading the truncation artefact rather than repetition. Deltas
near zero mean the criticism does not hold for that class.
""")
    return results


# --------------------------------------------------------------------------
# command: fpr  (runtime false positive rate)
# --------------------------------------------------------------------------


def split_into_steps(trace_content):
    """Header plus step blocks. FINAL: dropped - a run in progress has no answer yet.
    Mirrors experiments/test_classifier_runtime.py."""
    i = trace_content.find("\nFINAL:")
    if i != -1:
        trace_content = trace_content[:i]
    lines = trace_content.split("\n")
    header_lines, i = [], 0
    while i < len(lines) and not re.match(r"^\[1\]", lines[i]):
        header_lines.append(lines[i])
        i += 1
    header = "\n".join(header_lines).strip()
    steps, cur = {}, None
    for line in lines[i:]:
        m = re.match(r"^\[(\d+)\]", line)
        if m:
            cur = int(m.group(1))
            steps.setdefault(cur, [])
        if line.strip() and cur is not None:
            steps.setdefault(cur, []).append(line)
    return header, ["\n".join(steps[k]) for k in sorted(steps)]


def build_partial(header, blocks, n):
    return f"{header}\n\n" + "\n\n".join(blocks[:n]).strip()


def cmd_fpr(df, tok, mdl, id2label, md):
    failure_labels = [l for l in id2label.values() if l != "SUCCESS"]
    succ = df[df["Label"] == "SUCCESS"].reset_index(drop=True)
    fails = df[df["Label"] != "SUCCESS"].reset_index(drop=True)

    def sweep(frame):
        recs = []
        for _, row in frame.iterrows():
            header, blocks = split_into_steps(str(row["Trace Content"]))
            if not blocks:
                continue
            partials = [build_partial(header, blocks, n) for n in range(1, len(blocks) + 1)]
            pred_ids, probs = predict_batch(partials, tok, mdl)
            preds = [id2label[int(i)] for i in pred_ids]
            recs.append({"trace_id": row["Trace ID"], "true": row["Label"],
                         "n_steps": len(blocks), "preds": preds,
                         "max_conf": [float(p.max()) for p in probs]})
        return recs

    s_recs = sweep(succ)
    f_recs = sweep(fails)

    n_traces = len(s_recs)
    total_steps = sum(r["n_steps"] for r in s_recs)
    alarm_steps = sum(sum(1 for p in r["preds"] if p != "SUCCESS") for r in s_recs)
    alarmed_traces = sum(1 for r in s_recs if any(p != "SUCCESS" for p in r["preds"]))

    by_class = {}
    for r in s_recs:
        for p in r["preds"]:
            if p != "SUCCESS":
                by_class[p] = by_class.get(p, 0) + 1

    md.append("## 3. Runtime false positive rate\n")
    md.append("The gap in the original evaluation: SUCCESS traces were never fed to the\n"
              "step-by-step detector, so the alarm rate on healthy runs was never measured.\n")
    md.append(fmt_table(
        ["metric", "value"],
        [["healthy traces swept", n_traces],
         ["total prefixes evaluated", total_steps],
         ["prefixes predicting a failure", f"{alarm_steps} ({alarm_steps/max(total_steps,1):.1%})"],
         ["healthy traces alarmed at least once",
          f"{alarmed_traces}/{n_traces} ({alarmed_traces/max(n_traces,1):.1%})"],
         ["alarms per 100 healthy traces",
          f"{100*alarmed_traces/max(n_traces,1):.0f}"]]))

    if by_class:
        md.append("\nWhich class the false alarms claim:\n")
        md.append(fmt_table(["falsely predicted class", "prefix count"],
                            [[k, v] for k, v in sorted(by_class.items(),
                                                       key=lambda x: -x[1])]))

    # runtime precision: of all failure alarms raised across both pools, how many correct
    tp = sum(sum(1 for p in r["preds"] if p == r["true"]) for r in f_recs)
    fp = alarm_steps
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    md.append(f"\nRuntime alarm precision across the whole test set: "
              f"**{prec:.3f}** ({tp} correct failure predictions against {fp} false alarms, "
              f"counted per prefix).\n")
    md.append("""
**How to read this.** If healthy traces alarm often, the 78% and 63% early-detection
figures are not evidence of a usable monitor, because the detector is simply biased
toward failure classes on partial input. A low false positive rate here is the single
strongest result the project can produce.
""")
    return {"success": s_recs, "failure": f_recs,
            "trace_level_fpr": alarmed_traces / max(n_traces, 1),
            "step_level_fpr": alarm_steps / max(total_steps, 1),
            "runtime_precision": prec}


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "ablate", "fpr", "all"])
    ap.add_argument("--model", default="classifier/final_model")
    ap.add_argument("--test", default="data_splits/test.xlsx")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = (root / args.model).resolve()
    test_path = (root / args.test).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, what in [(model_dir, "model directory"), (test_path, "test split")]:
        if not p.exists():
            print(f"ERROR: {what} not found at {p}")
            sys.exit(1)

    df = pd.read_excel(test_path)
    tok, mdl = load_model(model_dir)
    id2label = resolve_labels(mdl, df)
    print(f"[data] {len(df)} test traces | classes: {sorted(id2label.values())}")

    md = ["# Audit experiment results\n",
          f"Checkpoint: `{model_dir.name}` | test traces: {len(df)} | device: {DEVICE}\n"]
    payload = {}

    if args.command in ("verify", "all"):
        payload["verify"] = cmd_verify(df, tok, mdl, id2label, md)
    if args.command in ("ablate", "all"):
        payload["ablate"] = cmd_ablate(df, tok, mdl, id2label, md)
    if args.command in ("fpr", "all"):
        payload["fpr"] = cmd_fpr(df, tok, mdl, id2label, md)

    summary = "\n".join(md)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (out_dir / f"{args.command}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(summary)
    print("=" * 72)
    print(f"\nWritten to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
