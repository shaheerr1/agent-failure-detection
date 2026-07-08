import json
from pathlib import Path

trace_id = "7d3a13e8"
matches = list(Path("data").rglob(f"*{trace_id}*.json"))

if not matches:
    print(f"No file found matching '{trace_id}'")
else:
    for path in matches:
        with open(path, encoding="utf-8") as f:
            trace = json.load(f)
        print(f"File: {path}")
        print(f"  Current folder: {path.parent.name}")
        print(f"  final_label: {trace.get('final_label')}")
        print(f"  label_human: {trace.get('label_human')}")
        print(f"  reviewed: {trace.get('reviewed')}")