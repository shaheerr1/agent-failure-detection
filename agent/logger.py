# logger.py
# Builds and saves a structured JSON trace from a LangGraph agent run.

import os
import json
import uuid
import datetime
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


class TraceLogger:
    """
    Parses a LangGraph message history into a structured trace
    and saves it to disk as a JSON file.
    """

    def __init__(self, task: str, category: str = "A", output_dir: str = "data/raw"):
        """
        Args:
            task:       The original task string given to the agent.
            category:   Task category — "A", "B", or "C".
            output_dir: Directory where trace JSON files are saved.
        """
        self.task = task
        self.category = category
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.trace = {
            "trace_id":          str(uuid.uuid4()),
            "timestamp":         datetime.datetime.now().isoformat(),
            "task":              task,
            "category":          category,
            "steps":             [],
            "final_answer":      None,
            "step_count":        0,
            "label_claude":      None,
            "label_gpt4o":       None,
            "label_human":       None,
            "final_label":       None,
            "confidence_claude": None,
            "confidence_gpt4o":  None,
            "agreement":         None,
        }

    def build_from_messages(self, messages: list) -> dict:
        """
        Parse the LangGraph message list into the trace format.

        LangGraph message order:
          HumanMessage  -> the original task
          AIMessage     -> agent thought + tool call (repeats per step)
          ToolMessage   -> tool output (one per tool call)
          AIMessage     -> final answer (no tool calls attached)
        """
        steps = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            # ── Agent thought + tool call ──────────────────────────────────
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:

                    # Look ahead for the matching ToolMessage by tool_call_id
                    observation = ""
                    for j in range(i + 1, len(messages)):
                        if (
                            isinstance(messages[j], ToolMessage)
                            and messages[j].tool_call_id == tool_call["id"]
                        ):
                            observation = messages[j].content
                            break

                    # Extract thought — fix empty thought HERE before appending
                    thought = msg.content if isinstance(msg.content, str) else ""
                    if not thought.strip():
                        thought = "[no explicit reasoning]"

                    steps.append({
                        "thought":      thought,
                        "action":       tool_call["name"],
                        "action_input": str(tool_call["args"]),
                        "observation":  observation.strip()
                    })

            # ── Final answer ───────────────────────────────────────────────
            # Must be elif — connected to the tool_calls check above
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip():
                    self.trace["final_answer"] = content.strip()

            # HumanMessage and ToolMessage are intentionally ignored here.
            # ToolMessages are handled by the lookahead above.

            i += 1

        self.trace["steps"] = steps
        self.trace["step_count"] = len(steps)
        return self.trace

    def save(self) -> str:
        """
        Write the trace to a JSON file in the output directory.
        Returns the file path of the saved trace.
        """
        filename = f"trace_{self.trace['trace_id']}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)

        print(f"\n[TraceLogger] Saved  -> {filepath}")
        print(f"[TraceLogger] Steps  : {self.trace['step_count']}")
        print(f"[TraceLogger] Answer : {str(self.trace['final_answer'])[:120]}")

        return filepath