# logger.py
# Builds and saves a structured JSON trace from a LangGraph agent run.
#
# LangGraph returns a message history after each invoke() call.
# This module parses that message history into the trace format used
# throughout the dissertation:
#
# {
#   "trace_id":     "<uuid>",
#   "timestamp":    "<iso datetime>",
#   "task":         "<original task>",
#   "steps":        [
#                     {
#                       "thought":      "<agent reasoning text>",
#                       "action":       "<tool name>",
#                       "action_input": "<tool input>",
#                       "observation":  "<tool output>"
#                     },
#                     ...
#                   ],
#   "final_answer": "<agent final answer or null>",
#   "label":        null    <-- filled in manually during annotation
# }

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

    def __init__(self, task: str, output_dir: str = "data/traces"):
        """
        Args:
            task:       The original task string given to the agent.
            output_dir: Directory where trace JSON files are saved.
        """
        self.task = task
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialise the trace skeleton
        self.trace = {
            "trace_id":     str(uuid.uuid4()),
            "timestamp":    datetime.datetime.now().isoformat(),
            "task":         task,
            "steps":        [],
            "final_answer": None,
            "label":        None   # filled in manually during annotation
        }

    def build_from_messages(self, messages: list) -> dict:
        """
        Parse the LangGraph message list into the trace format.

        LangGraph returns messages in this order:
          HumanMessage  -> the task
          AIMessage     -> agent thought + tool call (may repeat)
          ToolMessage   -> tool output (one per tool call)
          AIMessage     -> final answer (no tool calls)

        Args:
            messages: The message list from agent.invoke()["messages"]

        Returns:
            The completed trace dictionary.
        """
        steps = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            # --- Agent thought + tool call ---
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Look ahead for the matching ToolMessage
                    observation = ""
                    for j in range(i + 1, len(messages)):
                        if (
                            isinstance(messages[j], ToolMessage)
                            and messages[j].tool_call_id == tool_call["id"]
                        ):
                            observation = messages[j].content
                            break

                    # Thought text sits in the AIMessage content
                    thought = msg.content if isinstance(msg.content, str) else ""

                    steps.append({
                        "thought":      thought.strip(),
                        "action":       tool_call["name"],
                        "action_input": str(tool_call["args"]),
                        "observation":  observation.strip()
                    })

            # --- Final answer ---
            # An AIMessage with no tool calls is the final answer
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip():
                    self.trace["final_answer"] = content.strip()

            i += 1

        self.trace["steps"] = steps
        return self.trace

    def save(self) -> str:
        """
        Write the trace to a JSON file in the output directory.

        Returns:
            The file path of the saved trace.
        """
        filename = f"trace_{self.trace['trace_id']}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)

        print(f"\n[TraceLogger] Saved -> {filepath}")
        print(f"[TraceLogger] Steps: {len(self.trace['steps'])}")
        print(f"[TraceLogger] Answer: {str(self.trace['final_answer'])[:120]}")

        return filepath
