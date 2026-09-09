"""
classifier/train.py

Fine-tunes DeBERTa-v3 (or any HF encoder) on the frozen splits in data_splits/.
Extracted from train_classifier.ipynb so runs are reproducible and seed sweeps
are a single command.

Usage, from the repo root:

    python classifier/train.py                          # single run, seed 42
    python classifier/train.py --seed 43
    python classifier/train.py --seeds 42,43,44,45,46   # sweep, reports mean +/- std
    python classifier/train.py --model roberta-base --out-dir classifier/final_model_roberta

Each run writes classifier/runs/<model>_seed<N>/ containing config.json,
metrics.json and predictions.csv. A sweep also writes classifier/runs/sweep_<model>.json.

The dataset build (diversity sampling, true-group construction) still lives in
train_classifier.ipynb. This script consumes its frozen output.
"""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parent.parent


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def head_tail_truncate(text, tokenizer, max_length=512):
    """Keep the head and the tail; the task line and the final answer both survive."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    return tokenizer.decode(ids[:head] + ids[-(budget - head):])


def load_splits(tokenizer, splits_dir):
    frames = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.xlsx"
        if not path.exists():
            sys.exit(f"ERROR: missing split {path}. Build them in train_classifier.ipynb first.")
        frames[name] = pd.read_excel(path)

    labels = sorted(frames["train"]["Label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    for df in frames.values():
        df["label_id"] = df["Label"].map(label2id)
        if "text_truncated" not in df.columns or df["text_truncated"].isna().any():
            df["text_truncated"] = df["Trace Content"].apply(
                lambda t: head_tail_truncate(str(t), tokenizer))
    return frames, label2id, id2label


def to_hf(df, tokenizer):
    ds = Dataset.from_pandas(
        df[["text_truncated", "label_id"]].rename(
            columns={"text_truncated": "text", "label_id": "label"}))
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=512,
                                      padding=False), batched=True)


def make_metrics_fn():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0)}
    return compute_metrics


def bootstrap_ci(y_true, y_pred, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    vals = [f1_score(y_true[i], y_pred[i], labels=labels, average="macro", zero_division=0)
            for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def run_one(args, seed, device):
    print(f"\n{'=' * 70}\n[train] model={args.model} seed={seed} device={device}\n{'=' * 70}")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    frames, label2id, id2label = load_splits(tokenizer, ROOT / args.splits)
    num_labels = len(label2id)
    train_df, val_df, test_df = frames["train"], frames["val"], frames["test"]
    print(f"[data] train {len(train_df)} / val {len(val_df)} / test {len(test_df)} "
          f"| classes {list(label2id)}")

    train_ds = to_hf(train_df, tokenizer)
    val_ds = to_hf(val_df, tokenizer)
    test_ds = to_hf(test_df, tokenizer)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_labels),
                                         y=train_df["label_id"].values)
    # device-aware: the notebook hardcoded .to("cuda") and broke on anything else
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("[weights]", dict(zip(label2id, class_weights.round(3))))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weights_t)(
                outputs.logits.view(-1, num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, id2label=id2label, label2id=label2id,
        dtype=torch.float32)

    run_dir = ROOT / "classifier" / "runs" / f"{Path(args.model).name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        logging_steps=10,
        fp16=(device == "cuda"),   # fp16 is CUDA only; MPS and CPU need fp32
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_metrics_fn(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    trainer.train()

    out = trainer.predict(test_ds)
    preds = np.argmax(out.predictions, axis=-1)
    gold = out.label_ids
    label_ids = list(range(num_labels))
    names = [id2label[i] for i in label_ids]

    macro = float(f1_score(gold, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(gold, preds))
    lo, hi = bootstrap_ci(gold, preds, label_ids)

    print("\n" + classification_report(gold, preds, target_names=names, digits=3,
                                       zero_division=0))
    print(f"macro F1 {macro:.3f}  (95% bootstrap CI {lo:.3f} to {hi:.3f})  acc {acc:.3f}")
    print("confusion matrix (rows true, cols pred), order " + ", ".join(names))
    print(confusion_matrix(gold, preds, labels=label_ids))

    per_class = classification_report(gold, preds, target_names=names, digits=4,
                                      output_dict=True, zero_division=0)
    metrics = {"seed": seed, "model": args.model, "device": device,
               "macro_f1": round(macro, 4), "accuracy": round(acc, 4),
               "macro_f1_ci95": [lo, hi],
               "per_class": {n: {k: round(v, 4) for k, v in per_class[n].items()}
                             for n in names},
               "confusion_matrix": confusion_matrix(gold, preds, labels=label_ids).tolist()}

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        **vars(args), "seed": seed, "device": device,
        "python": platform.python_version(), "torch": torch.__version__,
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"trace_id": test_df["Trace ID"], "true": [id2label[i] for i in gold],
                  "pred": [id2label[i] for i in preds]}).to_csv(
        run_dir / "predictions.csv", index=False)

    if args.save_model:
        dest = ROOT / args.out_dir
        trainer.save_model(str(dest))
        tokenizer.save_pretrained(str(dest))
        print(f"[save] model -> {dest}")

    if not args.keep_checkpoints:
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    print(f"[save] metrics -> {run_dir / 'metrics.json'}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--splits", default="data_splits")
    p.add_argument("--out-dir", default="classifier/final_model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None,
                   help="comma separated list, e.g. 42,43,44,45,46 (overrides --seed)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=26)
    p.add_argument("--save-model", action="store_true",
                   help="write the trained weights to --out-dir (single-seed runs)")
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()

    device = get_device()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    if len(seeds) > 1 and args.save_model:
        print("[warn] --save-model with a sweep would keep only the last seed; ignoring it.")
        args.save_model = False

    results = [run_one(args, s, device) for s in seeds]

    if len(results) > 1:
        macros = np.array([r["macro_f1"] for r in results])
        accs = np.array([r["accuracy"] for r in results])
        names = list(results[0]["per_class"])
        print(f"\n{'=' * 70}\nSEED SWEEP: {args.model}, seeds {seeds}\n{'=' * 70}")
        print(f"{'metric':20s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
        print(f"{'macro F1':20s} {macros.mean():8.4f} {macros.std(ddof=1):8.4f} "
              f"{macros.min():8.4f} {macros.max():8.4f}")
        print(f"{'accuracy':20s} {accs.mean():8.4f} {accs.std(ddof=1):8.4f} "
              f"{accs.min():8.4f} {accs.max():8.4f}")
        print("\nper class F1:")
        summary = {"model": args.model, "seeds": seeds,
                   "macro_f1": {"mean": round(float(macros.mean()), 4),
                                "std": round(float(macros.std(ddof=1)), 4)},
                   "accuracy": {"mean": round(float(accs.mean()), 4),
                                "std": round(float(accs.std(ddof=1)), 4)},
                   "per_class": {}, "runs": results}
        for n in names:
            v = np.array([r["per_class"][n]["f1-score"] for r in results])
            print(f"  {n:20s} {v.mean():8.4f} {v.std(ddof=1):8.4f} "
                  f"{v.min():8.4f} {v.max():8.4f}")
            summary["per_class"][n] = {"mean": round(float(v.mean()), 4),
                                       "std": round(float(v.std(ddof=1)), 4)}
        out = ROOT / "classifier" / "runs" / f"sweep_{Path(args.model).name}.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[save] sweep summary -> {out}")
        print("\nReport mean and std, not the best seed.")


if __name__ == "__main__":
    main()
