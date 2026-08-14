# experiment/generate_test_traces.py
# Generates fresh traces for the runtime-detection experiment.
# Same agent as run_pipeline, but reads from experiment/test_tasks.txt
# and saves traces into experiment/traces/ (isolated from data/raw).
# No auto-labelling — raw traces are fed to the trained classifier directly.

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "agent"))

from agent import run_agent

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent
TASKS_FILE = ROOT_DIR / "agent" / "tasks" / "test_tasks.txt"
TRACE_OUT_DIR  = EXPERIMENT_DIR / "traces"
TRACE_OUT_DIR.mkdir(exist_ok=True)

# ── Task reader (same logic as run_pipeline) ──────────────────────────────────
def read_tasks(task_file: Path) -> list:
    """Read one task per line. Skip blank lines."""
    with open(task_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ── Generate ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TASKS_FILE.exists():
        print(f"[generate] ERROR — task file not found: {TASKS_FILE}")
        sys.exit(1)

    tasks = read_tasks(TASKS_FILE)

    print(f"\n{'=' * 70}")
    print(f"[generate] Task file  : {TASKS_FILE.name}")
    print(f"[generate] Tasks      : {len(tasks)}")
    print(f"[generate] Output dir : {TRACE_OUT_DIR}")
    print(f"{'=' * 70}\n")

    saved   = 0
    skipped = 0
    summary = []

    for i, task in enumerate(tasks, 1):
        print(f"\n[generate] ── Task {i}/{len(tasks)} ──")
        trace = run_agent(
            task=task,
            category="EXP",
            trace_output_dir=str(TRACE_OUT_DIR),
            min_steps=1,
            timeout_seconds=90,
        )

        discarded = trace.get("discarded", False)
        if discarded:
            skipped += 1
        else:
            saved += 1

        summary.append({
            "n":         i,
            "steps":     trace["step_count"],
            "answer":    trace["final_answer"] is not None,
            "discarded": discarded,
            "task":      task[:55],
        })

        if i < len(tasks):
            print(f"[generate] Waiting 15s...")
            time.sleep(15)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[generate] COMPLETE")
    print(f"{'=' * 70}")
    for r in summary:
        status = "SKIP" if r["discarded"] else ("OK  " if r["answer"] else "WARN")
        print(f"  {r['n']:2d}. {status} | {r['steps']} steps | {r['task']}")
    print(f"\n  Saved: {saved}  |  Skipped: {skipped}")
    print(f"  Traces in: {TRACE_OUT_DIR}")