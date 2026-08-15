# experiments/inspect_runtime.py
# Transparency tool: pick one trace per failure type, and watch the classifier's
# prediction evolve step by step as the partial trace grows.
# Shows the FULL trace text at each prefix + the prediction + all 4 class probs,


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

def split_into_steps(trace_content: str):
    fin = trace_content.find("\nFINAL:")
    if fin != -1:
        trace_content = trace_content[:fin]
    lines = trace_content.split("\n")
    header_lines, i = [], 0
    while i < len(lines) and not re.match(r"^\[1\]", lines[i]):
        header_lines.append(lines[i]); i += 1
    header = "\n".join(header_lines).strip()
    steps, cur = {}, None
    for line in lines[i:]:
        m = re.match(r"^\[(\d+)\]", line)
        if m:
            cur = int(m.group(1)); steps.setdefault(cur, [])
        if cur is not None and line.strip():
            steps.setdefault(cur, []).append(line)
    ordered = ["\n".join(steps[k]) for k in sorted(steps)]
    return header, ordered

def build_partial(header, step_blocks, n):
    return f"{header}\n\n" + "\n\n".join(step_blocks[:n])

def head_tail_truncate(text, tokenizer, max_length=512):
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - 2
    if len(ids) <= budget: return text
    head = budget // 2
    return tokenizer.decode(ids[:head] + ids[-(budget - head):])

def classify(text, tokenizer, model, id2label):
    text = head_tail_truncate(text, tokenizer)
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)[0]
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

# ── load ──────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
model.eval()
id2label = model.config.id2label

df = pd.read_excel(TEST_XLSX)

# pick ONE trace per failure type (first of each)
picks = {}
for label in ["UNSAFE_EXECUTION", "LOOP", "HALLUCINATION"]:
    sub = df[df["Label"] == label]
    if len(sub):
        picks[label] = sub.iloc[0]

# ── inspect each ──────────────────────────────────────────────────────────────
for label, row in picks.items():
    header, steps = split_into_steps(row["Trace Content"])
    print("\n\n" + "#" * 100)
    print(f"# TRUE LABEL: {label}   |   Trace ID: {row['Trace ID']}   |   {len(steps)} steps")
    print("#" * 100)
    print(f"\n{header}\n")

    for n in range(1, len(steps) + 1):
        # show the step we just added
        print("." * 100)
        print(f"--- After STEP {n}/{len(steps)} — newly added step content: ---")
        print(steps[n-1])

        partial = build_partial(header, steps, n)
        probs = classify(partial, tokenizer, model, id2label)
        pred = max(probs, key=probs.get)

        # visual bar for each class prob
        print(f"\n  >>> CLASSIFIER PREDICTION: {pred}" +
              ("  <-- MATCHES TRUE LABEL" if pred == label else ""))
        for cls in sorted(probs, key=probs.get, reverse=True):
            bar = "█" * int(probs[cls] * 30)
            print(f"      {cls:16s} {probs[cls]:.3f} {bar}")
        print()

    input(f"\n[Press ENTER to see the next failure type...]")

print("\nDone.")