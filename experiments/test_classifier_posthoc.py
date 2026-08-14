# experiments/test_classifier_posthoc.py
# OPTION A: post-hoc classification test.
# Loads the trained DeBERTa model, formats each experiment trace into the
# EXACT training text format, predicts, and reports per-trace results.
# This tests the classifier on fresh gpt-oss-120b traces (a cross-model check).

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "classifier" / "final_model"
TRACE_DIR  = Path(__file__).parent / "traces"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Trace -> training text format ─────────────────────────────────────────────
def format_trace(trace: dict) -> str:
    """
    Rebuild the EXACT 'Trace Content' string the classifier trained on.
    Rules read directly from training_dataset.xlsx:
      - "TASK: {task}" then blank line
      - per step: optional "[N] THOUGHT:" (only if real thought),
        then "[N] ACTION:", "[N] INPUT:", "[N] OBS:"
      - blank line between steps
      - end with "FINAL: {final_answer}" if present
    """
    lines = [f"TASK: {trace['task']}", ""]

    for idx, step in enumerate(trace["steps"], 1):
        thought = step.get("thought", "").strip()
        # training format OMITS placeholder thoughts
        if thought and thought != "[no explicit reasoning]":
            lines.append(f"[{idx}] THOUGHT: {thought}")
        lines.append(f"[{idx}] ACTION: {step['action']}")
        lines.append(f"[{idx}] INPUT: {step['action_input']}")
        lines.append(f"[{idx}] OBS: {step['observation']}")
        lines.append("")   # blank line between steps

    if trace.get("final_answer"):
        lines.append(f"FINAL: {trace['final_answer']}")

    return "\n".join(lines).strip()

# ── Head+tail truncation (same as training) ───────────────────────────────────
def head_tail_truncate(text, tokenizer, max_length=512):
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget:
        return text
    head = budget // 2
    tail = budget - head
    return tokenizer.decode(ids[:head] + ids[-tail:])

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[test] Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
model.eval()

id2label = model.config.id2label
print(f"[test] Classes: {id2label}")
print(f"[test] Device: {DEVICE}\n")

# ── Classify each trace ───────────────────────────────────────────────────────
trace_files = sorted(TRACE_DIR.glob("trace_*.json"))
print(f"[test] Found {len(trace_files)} traces\n")
print("=" * 90)

for tf in trace_files:
    trace = json.loads(tf.read_text(encoding="utf-8"))

    text = format_trace(trace)
    text = head_tail_truncate(text, tokenizer)

    enc = tokenizer(text, truncation=True, max_length=512,
                    return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        pred = id2label[pred_id]
        conf = float(probs[pred_id])

    task_short = trace["task"][:55]
    print(f"TASK : {task_short}")
    print(f"PRED : {pred}  (confidence {conf:.3f})  |  {trace['step_count']} steps")
    # show full probability spread
    spread = "  ".join(f"{id2label[i]}={probs[i]:.2f}" for i in range(len(probs)))
    print(f"PROBS: {spread}")
    print("-" * 90)