"""
classifier/evaluate.py

Scores an already-trained checkpoint on a frozen split. No training.

Usage, from the repo root:

    python classifier/evaluate.py
    python classifier/evaluate.py --model classifier/final_model_roberta
    python classifier/evaluate.py --split data_splits/val.xlsx

Reports per-class precision/recall/F1, macro F1 with a bootstrap 95% confidence
interval, the confusion matrix, and a provenance breakdown showing how much of
each class score rests on REAL versus SYNTHETIC traces.

For the shortcut ablations and the runtime false-alarm rate, see audit_run.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 512


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def head_tail_truncate(text, tokenizer, max_length=MAX_LEN):
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    return tokenizer.decode(ids[:head] + ids[-(budget - head):])


@torch.no_grad()
def predict(texts, tokenizer, model, device, batch_size=16):
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        chunk = [head_tail_truncate(str(t), tokenizer) for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        probs = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        preds.extend(probs.argmax(axis=1).tolist())
        confs.extend(probs.max(axis=1).tolist())
    return np.array(preds), np.array(confs)


def bootstrap_ci(y_true, y_pred, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=labels,
                             average="macro", zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="classifier/final_model")
    p.add_argument("--split", default="data_splits/test.xlsx")
    p.add_argument("--out", default="results/evaluate.json")
    args = p.parse_args()

    model_dir = ROOT / args.model
    split_path = ROOT / args.split
    for path, what in [(model_dir, "checkpoint"), (split_path, "split")]:
        if not path.exists():
            raise SystemExit(f"ERROR: {what} not found at {path}")

    device = get_device()
    print(f"[load] {model_dir}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    df = pd.read_excel(split_path)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    if all(v.startswith("LABEL_") for v in id2label.values()) and "label_id" in df.columns:
        pairs = df[["label_id", "Label"]].drop_duplicates()
        id2label = {int(r.label_id): str(r.Label) for r in pairs.itertuples()}

    names = [id2label[i] for i in sorted(id2label)]
    label2id = {v: k for k, v in id2label.items()}
    print(f"[data] {len(df)} traces from {split_path.name} | classes {names}")

    pred_ids, confs = predict(df["Trace Content"].tolist(), tokenizer, model, device)
    gold_ids = df["Label"].map(label2id).values
    label_ids = sorted(id2label)

    macro = float(f1_score(gold_ids, pred_ids, average="macro", zero_division=0))
    acc = float(accuracy_score(gold_ids, pred_ids))
    lo, hi = bootstrap_ci(gold_ids, pred_ids, label_ids)

    print("\n" + classification_report(gold_ids, pred_ids, labels=label_ids,
                                       target_names=names, digits=3, zero_division=0))
    print(f"macro F1 {macro:.3f}  (95% bootstrap CI {lo:.3f} to {hi:.3f})  acc {acc:.3f}")
    print("\nconfusion matrix (rows true, cols pred), order " + ", ".join(names))
    cm = confusion_matrix(gold_ids, pred_ids, labels=label_ids)
    for n, row in zip(names, cm):
        print(f"  {n:18s} " + " ".join(f"{v:4d}" for v in row))

    prov = None
    if "Source" in df.columns:
        print("\nprovenance of the held out set (a score resting on 0 REAL traces "
              "does not generalise):")
        tab = pd.crosstab(df["Label"], df["Source"])
        print(tab.to_string())
        correct = pd.Series(pred_ids == gold_ids, index=df.index)
        print("\naccuracy by class and provenance:")
        acc_tab = df.assign(correct=correct).groupby(
            ["Label", "Source"])["correct"].agg(["mean", "size"]).round(3)
        print(acc_tab.to_string())
        prov = {"composition": tab.to_dict(),
                "accuracy": acc_tab.reset_index().to_dict(orient="records")}

    errs = df.loc[pred_ids != gold_ids, ["Trace ID", "Label"]].copy()
    if len(errs):
        errs["predicted"] = [id2label[i] for i in pred_ids[pred_ids != gold_ids]]
        errs["confidence"] = confs[pred_ids != gold_ids].round(3)
        print(f"\n{len(errs)} errors:")
        print(errs.to_string(index=False))
        hi_conf = (errs["confidence"] > 0.8).sum()
        print(f"\n{hi_conf} of {len(errs)} errors are above 0.8 confidence. "
              f"Confidence cannot gate these.")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": str(args.model), "split": str(args.split), "device": device,
        "macro_f1": round(macro, 4), "accuracy": round(acc, 4),
        "macro_f1_ci95": [lo, hi],
        "report": classification_report(gold_ids, pred_ids, labels=label_ids,
                                        target_names=names, output_dict=True,
                                        zero_division=0),
        "confusion_matrix": cm.tolist(),
        "provenance": prov,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
