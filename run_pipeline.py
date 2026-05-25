# run_pipeline.py
# Full pipeline: reads tasks → runs agent → auto-labels → routes to folders
#
# Usage:
#   python run_pipeline.py
#
# Change task_file and category at the bottom to switch between A, B, C.

import sys
import time
from pathlib import Path

# Add agent/ to path so we can import agent.py and tools.py
sys.path.insert(0, str(Path(__file__).parent / "agent"))

from agent import run_agent
from annotation.auto_labeller import label_all_traces

ROOT_DIR = Path(__file__).parent

# ── Task reader ───────────────────────────────────────────────────────────────
def read_tasks(task_file: Path) -> list:
    """Read one task per line from a text file. Skip blank lines."""
    with open(task_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(
    task_file: str,
    category: str,
    sleep_between: int = 15,
    auto_label: bool = True
):
    task_path = ROOT_DIR / task_file

    if not task_path.exists():
        print(f"[Pipeline] ERROR — task file not found: {task_path}")
        return

    tasks = read_tasks(task_path)

    print(f"\n{'=' * 70}")
    print(f"[Pipeline] Task file : {task_file}")
    print(f"[Pipeline] Category  : {category}")
    print(f"[Pipeline] Tasks     : {len(tasks)}")
    print(f"[Pipeline] Sleep     : {sleep_between}s between tasks")
    print(f"[Pipeline] Auto-label: {auto_label}")
    print(f"{'=' * 70}\n")

    saved   = 0
    skipped = 0

    for i, task in enumerate(tasks, 1):
        print(f"\n[Pipeline] ── Task {i}/{len(tasks)} ──")
        trace = run_agent(task=task, category=category)

        if trace.get("discarded"):
            skipped += 1
            print(f"[Pipeline] Discarded — {trace.get('discard_reason', 'unknown')}")
        else:
            saved += 1

        if i < len(tasks):
            print(f"[Pipeline] Waiting {sleep_between}s...")
            time.sleep(sleep_between)

    # ── Generation summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[Pipeline] GENERATION COMPLETE")
    print(f"  Tasks run : {len(tasks)}")
    print(f"  Saved     : {saved}")
    print(f"  Skipped   : {skipped}")
    print(f"  Discard % : {skipped / len(tasks) * 100:.1f}%")
    print(f"{'=' * 70}\n")

    # ── Auto-labeller ─────────────────────────────────────────────────────────
    if auto_label and saved > 0:
        print(f"[Pipeline] Starting auto-labeller on {saved} new traces...\n")
        label_all_traces()
    elif saved == 0:
        print(f"[Pipeline] No traces to label — skipping auto-labeller.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        task_file="agent/tasks/category_c.txt",
        category="C",
        sleep_between=20,
        auto_label=True
    )