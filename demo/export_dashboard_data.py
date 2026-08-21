#!/usr/bin/env python3
"""
export_dashboard_data.py
------------------------
Builds demo/dashboard_data.js from the project's frozen evaluation artefacts.

The dashboard is a static viewer: it runs no inference of its own. This script
is the only place the model is loaded, and it is run once, offline, to produce
the data file the dashboard reads.

Usage (from the project root, with the venv active):

    python demo/export_dashboard_data.py

Outputs:
    demo/dashboard_data.js      window.DASHBOARD_DATA = {...}

A .js file is emitted rather than .json so the dashboard can be opened directly
from the filesystem: browsers block fetch() of local .json under file://, but a
<script src> tag loads fine.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "classifier" / "final_model"
TEST_XLSX = ROOT / "data_splits" / "test.xlsx"
CLEAN_XLSX = ROOT / "data_splits" / "dataset_clean_433.xlsx"
RUNTIME_XLSX = ROOT / "experiments" / "runtime_results.xlsx"
OUT_JS = ROOT / "demo" / "dashboard_data.js"

CLASSES = ["SUCCESS", "HALLUCINATION", "LOOP", "UNSAFE_EXECUTION"]

# How many traces to make available in the replay panel, per class.
REPLAY_PER_CLASS = 4


# ----------------------------------------------------------------------------
# Trace parsing — mirrors experiments/inspect_runtime.py exactly so that the
# dashboard's step boundaries match those used in the runtime experiment.
# ----------------------------------------------------------------------------
def split_into_steps(trace_content: str):
    """Split a trace into (header, [step blocks]), dropping the FINAL answer."""
    fin = trace_content.find("\nFINAL:")
    final_answer = ""
    if fin != -1:
        final_answer = trace_content[fin:].replace("\nFINAL:", "", 1).strip()
        trace_content = trace_content[:fin]

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
        if cur is not None and line.strip():
            steps.setdefault(cur, []).append(line)

    ordered = ["\n".join(steps[k]) for k in sorted(steps)]
    return header, ordered, final_answer


def build_partial(header: str, step_blocks: list[str], n: int) -> str:
    return f"{header}\n\n" + "\n\n".join(step_blocks[:n])


def parse_step_block(block: str) -> dict:
    """Pull THOUGHT / ACTION / INPUT / OBS out of one step block for display."""
    out = {"thought": "", "action": "", "input": "", "obs": ""}
    current = None
    for line in block.split("\n"):
        m = re.match(r"^\[\d+\]\s*(THOUGHT|ACTION|INPUT|OBS):\s*(.*)$", line)
        if m:
            current = m.group(1).lower()
            out[current] = m.group(2)
        elif current:
            out[current] += "\n" + line
    return {k: v.strip() for k, v in out.items()}


# ----------------------------------------------------------------------------
def load_model():
    """Load the fine-tuned classifier. Returns None if unavailable."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("  ! torch/transformers not available — exporting without per-step "
              "probabilities.", file=sys.stderr)
        return None

    if not MODEL_DIR.exists():
        print(f"  ! {MODEL_DIR} not found — exporting without per-step "
              "probabilities.", file=sys.stderr)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    mdl.eval()
    print(f"  · model loaded on {device}")
    return {"torch": torch, "F": F, "tok": tok, "mdl": mdl,
            "device": device, "id2label": mdl.config.id2label}


def head_tail_truncate(text: str, tok, max_length: int = 512) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    return tok.decode(ids[:head] + ids[-(budget - head):])


def classify(text: str, m: dict) -> dict:
    text = head_tail_truncate(text, m["tok"])
    enc = m["tok"](text, truncation=True, max_length=512,
                   return_tensors="pt").to(m["device"])
    with m["torch"].no_grad():
        probs = m["F"].softmax(m["mdl"](**enc).logits, dim=-1)[0]
    return {m["id2label"][i]: round(float(probs[i]), 4) for i in range(len(probs))}


# ----------------------------------------------------------------------------
def main() -> None:
    print("Building dashboard data …")

    for p in (TEST_XLSX, CLEAN_XLSX, RUNTIME_XLSX):
        if not p.exists():
            sys.exit(f"Missing required artefact: {p}")

    test = pd.read_excel(TEST_XLSX)
    clean = pd.read_excel(CLEAN_XLSX)
    runtime = pd.read_excel(RUNTIME_XLSX)
    print(f"  · test {len(test)} · dataset {len(clean)} · runtime {len(runtime)}")

    # ---- dataset panel ----------------------------------------------------
    counts = clean["Label"].value_counts()
    sources = (clean["Source"].value_counts()
               if "Source" in clean.columns else pd.Series(dtype=int))

    dataset = {
        "total": int(len(clean)),
        "splits": {"train": 259, "val": 87, "test": 87},
        "classes": [{"name": c, "count": int(counts.get(c, 0))} for c in CLASSES],
        "sources": [{"name": str(k), "count": int(v)} for k, v in sources.items()],
    }

    # ---- offline results --------------------------------------------------
    # Frozen values from the evaluation run recorded in the training notebook.
    offline = {
        "macroF1": 0.820,
        "accuracy": 0.816,
        "perClass": [
            {"name": "SUCCESS",          "precision": 0.783, "recall": 0.692,
             "f1": 0.735, "support": 26},
            {"name": "HALLUCINATION",    "precision": 0.667, "recall": 0.727,
             "f1": 0.696, "support": 22},
            {"name": "LOOP",             "precision": 0.824, "recall": 0.875,
             "f1": 0.848, "support": 16},
            {"name": "UNSAFE_EXECUTION", "precision": 1.000, "recall": 1.000,
             "f1": 1.000, "support": 23},
        ],
        # rows = true, cols = predicted, in CONFUSION_ORDER
        "confusionOrder": ["HALLUCINATION", "LOOP", "SUCCESS", "UNSAFE_EXECUTION"],
        "confusion": [
            [16, 1, 5, 0],
            [2, 14, 0, 0],
            [6, 2, 18, 0],
            [0, 0, 0, 23],
        ],
    }

    # ---- baseline comparison ---------------------------------------------
    baselines = {
        "models": [
            {"name": "Majority class",   "short": "Majority",  "macroF1": 0.115,
             "kind": "floor"},
            {"name": "MiniLM + LogReg",  "short": "MiniLM",    "macroF1": 0.464,
             "kind": "baseline"},
            {"name": "TF-IDF + LogReg",  "short": "TF-IDF",    "macroF1": 0.664,
             "kind": "baseline"},
            {"name": "RoBERTa-base",     "short": "RoBERTa",   "macroF1": 0.756,
             "kind": "baseline"},
            {"name": "DeBERTa-v3-base",  "short": "DeBERTa-v3", "macroF1": 0.820,
             "kind": "model"},
        ],
        "perClass": {
            "HALLUCINATION":    [0.000, 0.586, 0.678, 0.703, 0.696],
            "LOOP":             [0.000, 0.353, 0.476, 0.970, 0.848],
            "SUCCESS":          [0.460, 0.359, 0.619, 0.667, 0.735],
            "UNSAFE_EXECUTION": [0.000, 0.558, 0.885, 0.686, 1.000],
        },
    }

    # ---- runtime detection ------------------------------------------------
    by_type = []
    for label in ["UNSAFE_EXECUTION", "LOOP", "HALLUCINATION"]:
        sub = runtime[runtime["label"] == label]
        detected = sub["detected_at"].notna()
        # "Detected (any step)" — the failure was named at some prefix length,
        # including the full trace. "Before final step" is the runtime measure:
        # the failure was named strictly earlier than the last step.
        det_any = int(detected.sum())
        det_before = int((detected & (sub["detected_at"] < sub["n_steps"])).sum())
        # Average detection point is taken over every detected trace, matching
        # the runtime experiment. It is therefore conditional on detection and
        # must be read alongside the rate, not independently of it.
        frac = sub["detected_frac"].dropna()
        by_type.append({
            "name": label,
            "n": int(len(sub)),
            "detectedAny": det_any,
            "detectedBefore": det_before,
            "rate": round(det_before / len(sub), 3) if len(sub) else 0.0,
            "avgPoint": round(float(frac.mean()), 3) if len(frac) else None,
        })

    runtime_traces = [
        {
            "traceId": str(r.trace_id),
            "label": str(r.label),
            "nSteps": int(r.n_steps),
            "detectedAt": (int(r.detected_at) if pd.notna(r.detected_at) else None),
            "detectedFrac": (round(float(r.detected_frac), 3)
                             if pd.notna(r.detected_frac) else None),
            "fullTracePred": str(r.full_trace_pred),
        }
        for r in runtime.itertuples()
    ]

    # ---- replay panel -----------------------------------------------------
    m = load_model()
    det_lookup = {str(r.trace_id): r for r in runtime.itertuples()}

    replay = []
    for label in ["UNSAFE_EXECUTION", "LOOP", "HALLUCINATION", "SUCCESS"]:
        sub = test[test["Label"] == label]
        # Prefer REAL traces, and prefer ones the runtime experiment caught early.
        if "Source" in sub.columns:
            sub = pd.concat([sub[sub["Source"] == "REAL"],
                             sub[sub["Source"] != "REAL"]])
        picked = 0
        for row in sub.itertuples():
            if picked >= REPLAY_PER_CLASS:
                break
            header, blocks, final_answer = split_into_steps(row._2)  # Trace Content
            if len(blocks) < 2:
                continue

            steps = []
            for n in range(1, len(blocks) + 1):
                item = {"n": n, **parse_step_block(blocks[n - 1])}
                if m is not None:
                    partial = build_partial(header, blocks, n)
                    probs = classify(partial, m)
                    item["probs"] = probs
                    item["pred"] = max(probs, key=probs.get)
                steps.append(item)

            d = det_lookup.get(str(row._1))
            replay.append({
                "traceId": str(row._1),
                "label": label,
                "source": str(getattr(row, "Source", "REAL")),
                "task": header.replace("TASK:", "", 1).strip(),
                "finalAnswer": final_answer,
                "nSteps": len(blocks),
                "detectedAt": (int(d.detected_at)
                               if d is not None and pd.notna(d.detected_at) else None),
                "steps": steps,
            })
            picked += 1
            print(f"  · replay {label:17s} {row._1}  ({len(blocks)} steps)")

    payload = {
        "meta": {
            "project": "Runtime Detection of Failure Modes in LLM Agents",
            "model": "DeBERTa-v3-base",
            "agent": "Llama 4 Scout (Groq) — LangGraph ReAct",
            "classes": CLASSES,
            "hasProbabilities": m is not None,
        },
        "dataset": dataset,
        "offline": offline,
        "baselines": baselines,
        "runtime": {"byType": by_type, "traces": runtime_traces},
        "replay": replay,
    }

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text(
        "// Generated by demo/export_dashboard_data.py — do not edit by hand.\n"
        "window.DASHBOARD_DATA = "
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    kb = OUT_JS.stat().st_size / 1024
    print(f"\nWrote {OUT_JS.relative_to(ROOT)}  ({kb:.0f} KB, "
          f"{len(replay)} replay traces, "
          f"probabilities={'yes' if m else 'NO'})")
    if m is None:
        print("\n  To include per-step probabilities, run this again with the "
              "project venv active so torch/transformers and "
              "classifier/final_model are available.")


if __name__ == "__main__":
    main()
