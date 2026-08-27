#!/usr/bin/env python3
"""Add a greedy-tuned 128k thinking variant to Pi models.json."""
import json
from pathlib import Path

path = Path.home() / ".pi" / "agent" / "models.json"
data = json.loads(path.read_text())

freetoken_models = data["providers"]["freetoken"]["models"]

new_entry = {
    "id": "qwen3.6-35b-a3b-128k-thinking-greedy",
    "name": "Qwen3.6 35B-A3B 128k thinking greedy (tool-optimized)",
    "input": ["text"],
    "contextWindow": 131072,
    "maxTokens": 8192,
    "reasoning": True,
    "thinkingLevelMap": {
        "minimal": "low", "low": "low", "medium": "medium",
        "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
    },
    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": True},
    "samplingParams": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
}

ids = [m["id"] for m in freetoken_models]
if new_entry["id"] not in ids:
    freetoken_models.append(new_entry)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Added {new_entry['id']}")
else:
    print(f"{new_entry['id']} already present")

print(f"Total freetoken models: {len(freetoken_models)}")