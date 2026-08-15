# experiments/test_classifier_runtime.py
# RUNTIME (step-by-step) detection test.
# For each labelled test trace, feed the classifier the trace GROWING one
# step at a time (step 1, then 1-2, then 1-2-3 ...) WITHOUT the final answer,
# and record the first step at which it correctly predicts the true failure.
# This measures runtime detection: does it catch the failure before completion?

import re
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "classifier" / "final_model"
TEST_XLSX = ROOT / "data_splits" / "test.xlsx"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Split a full Trace Content string into (header, [step_blocks]) ─────────────
def split_into_steps(trace_content: str):
    """
    Returns (task_header, list_of_step_blocks).
    A step block starts at a line '[N] THOUGHT:' or '[N] ACTION:' and runs
    until the next step marker. The FINAL: line (if any) is dropped, since a
    trace 'in progress' has no final answer yet.
    """
    # drop everything from FINAL: onwards
    fin = trace_content.find("\nFINAL:")
    if fin != -1:
        trace_content = trace_content[:fin]

    lines = trace_content.split("\n")

    # header = everything before the first [1] marker
    header_lines, i = [], 0
    while i < len(lines) and not re.match(r"^\[1\]", lines[i]):
        header_lines.append(lines[i])
        i += 1
    header = "\n".join(header_lines).strip()

    # group remaining lines by step number
    steps = {}
    for line in lines[i:]:
        m = re.match(r"^\[(\d+)\]", line)
        if m:
            cur = int(m.group(1))
            steps.setdefault(cur, [])
        if line.strip() and 'cur' in dir():
            steps.setdefault(cur, []).append(line)

    ordered = [ "\n".join(steps[k]) for k in sorted(steps) ]
    return header, ordered

def build_partial(header, step_blocks, n):
    """Rebuild trace text using only the first n steps (no FINAL)."""
    body = "\n\n".join(step_blocks[:n])
    return f"{header}\n\n{body}".strip()

# ── truncation (same as training) ─────────────────────────────────────────────
def head_tail_truncate(text, tokenizer, max_length=512):
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    return tokenizer.decode(ids[:head] + ids[-(budget - head):])

def classify(text, tokenizer, model, id2label):
    text = head_tail_truncate(text, tokenizer)
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)[0]
    pid = int(probs.argmax())
    return id2label[pid], float(probs[pid])

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[runtime] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
model.eval()
id2label = model.config.id2label

df = pd.read_excel(TEST_XLSX)
# focus on FAILURE traces only (SUCCESS has no failure step to 'catch')
FAILURE_LABELS = ["LOOP", "HALLUCINATION", "UNSAFE_EXECUTION"]
fails = df[df["Label"].isin(FAILURE_LABELS)].reset_index(drop=True)
print(f"[runtime] {len(fails)} failure traces to test\n")

# ── Run step-by-step ──────────────────────────────────────────────────────────
records = []
for _, row in fails.iterrows():
    true_label = row["Label"]
    header, steps = split_into_steps(row["Trace Content"])
    n_steps = len(steps)
    if n_steps == 0:
        continue

    detected_at = None
    for n in range(1, n_steps + 1):
        partial = build_partial(header, steps, n)
        pred, conf = classify(partial, tokenizer, model, id2label)
        if pred == true_label and detected_at is None:
            detected_at = n            # first correct detection step
            det_conf = conf

    # also classify the FULL trace (post-hoc reference)
    full_pred, full_conf = classify(build_partial(header, steps, n_steps),
                                    tokenizer, model, id2label)

    records.append({
        "trace_id": row["Trace ID"],
        "label": true_label,
        "n_steps": n_steps,
        "detected_at": detected_at,                       # None = never caught early
        "detected_frac": round(detected_at / n_steps, 2) if detected_at else None,
        "full_trace_pred": full_pred,
    })

res = pd.DataFrame(records)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 80)
print("PER-TRACE RESULTS")
print("=" * 80)
for _, r in res.iterrows():
    if pd.notna(r["detected_at"]):
        frac = int(r["detected_frac"] * 100)
        msg = f"caught at step {int(r['detected_at'])}/{r['n_steps']} ({frac}% through)"
    else:
        msg = f"NEVER caught early (full-trace pred: {r['full_trace_pred']})"
    print(f"  {r['label']:16s} | {msg}")

print("\n" + "=" * 80)
print("SUMMARY BY FAILURE TYPE")
print("=" * 80)
for label in FAILURE_LABELS:
    sub = res[res["label"] == label]
    if len(sub) == 0:
        continue
    caught = int(sub["detected_at"].notna().sum())
    total = len(sub)
    # "early" = caught strictly before the final step
    early = int(((sub["detected_frac"] < 1.0) & sub["detected_frac"].notna()).sum())
    avg_frac = sub["detected_frac"].dropna().mean()
    print(f"\n{label}:")
    print(f"  detected (at any step):      {caught}/{total}")
    print(f"  detected BEFORE final step:  {early}/{total}")
    if pd.notna(avg_frac):
        print(f"  avg detection point:         {avg_frac*100:.0f}% through the trace")

res.to_excel(ROOT / "experiments" / "runtime_results.xlsx", index=False)
print(f"\n[runtime] Saved detailed results to experiments/runtime_results.xlsx")