#!/usr/bin/env python3
"""Sampling sweep harness — caches every generation to CSV for plotting.

Usage:
    python3 sweep.py                     # run all phases
    python3 sweep.py --phases 1          # temperature sweep only
    python3 sweep.py --list              # show phases
    python3 sweep.py --summary results/sweep.csv   # summarize a CSV

Resumable: skips (config, prompt, sample_i) keys already present in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

API = os.environ.get("FT_URL", "http://127.0.0.1:1919/v1/chat/completions")
MODEL = os.environ.get("FT_MODEL", "qwen3.6-35b-a3b.gguf")
RESULT_DIR = Path(__file__).resolve().parent / "results"

CSV_FIELDS = [
    "phase",
    "config_id",
    "temperature",
    "top_k",
    "top_p",
    "max_tokens",
    "prompt_key",
    "prompt",
    "sample_i",
    "started_at",
    "elapsed_s",
    "finish_reason",
    "success",
    "tool_name",
    "tool_args",
    "format_class",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_len",
    "content",
    "reasoning",
]

EXPLICIT = "Use the bash tool to list files in /tmp"
SOFT = "What files are in the current directory? Use bash."

QUALITY_PROMPTS = {
    "water_cycle": "Explain the water cycle in exactly 5 sentences.",
    "poem": "Write a short poem about autumn. 4 lines.",
    "pi_cmp": "Is 355/113 greater than pi? Answer with only yes or no.",
    "tcp_udp": "Explain the difference between TCP and UDP in 3 sentences.",
}

# Realistic agent tool set (Pi-style) so the model sees tool definitions in the
# chat template. Without a ``tools`` array the model just answers in prose/code
# fences instead of emitting structured tool calls.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write text content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "Text to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


@dataclass
class Config:
    temperature: float
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 600
    label: str | None = None

    @property
    def id(self) -> str:
        return self.label or f"t{self.temperature:.2f}_k{self.top_k}_p{self.top_p:.2f}"


def build_grid_phase(n: int = 20) -> dict:
    """2D temperature x top_p grid (k=20) around the best region."""
    ts = [0.90 + i * 0.01 for i in range(5)]  # 0.90..0.94
    ps = [0.95 + i * 0.01 for i in range(6)]  # 0.95..1.00
    return {
        "name": "phase6_tp_grid",
        "items": [
            {"config": Config(round(t, 2), 20, round(p, 2)), "prompt": EXPLICIT, "n": n}
            for t in ts
            for p in ps
        ],
    }


def build_phases(n_primary: int = 10, n_soft: int = 8) -> list[dict]:
    temps = [0.0, 0.5, 0.7, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    explicit = [
        {"config": Config(t, 20, 0.95), "prompt": EXPLICIT, "n": n_primary}
        for t in temps
    ]

    soft = (
        {"config": Config(t, 20, 0.95), "prompt": SOFT, "n": n_soft}
        for t in (0.95, 0.98, 1.0)
    )

    topk_sweep = (
        {"config": Config(1.0, k, 0.95), "prompt": EXPLICIT, "n": n_soft}
        for k in (20, 60, 200)
    )

    topp_sweep = (
        {"config": Config(1.0, 20, p), "prompt": EXPLICIT, "n": n_soft}
        for p in (0.90, 0.95, 1.00)
    )

    quality = [
        {
            "config": Config(t, k, p, max_tokens=800, label=f"qual_t{t}_k{k}_p{p}"),
            "prompt": prompt,
            "n": 2,
        }
        for t, k, p in ((0.0, -1, 1.0), (0.3, 20, 0.95), (1.0, 20, 0.95))
        for prompt in QUALITY_PROMPTS.values()
    ]

    return [
        {"name": "phase1_temp_primary", "items": explicit},
        {"name": "phase2_temp_soft", "items": list(soft)},
        {"name": "phase3_topk", "items": list(topk_sweep)},
        {"name": "phase4_topp", "items": list(topp_sweep)},
        {"name": "phase5_quality", "items": quality},
        build_grid_phase(),
    ]


def classify(content: str | None, tool_calls: list) -> str:
    if tool_calls:
        return "parsed_tool_calls"
    c = content or ""
    if "standard_tool_calling" in c:
        return "std_meta_tag"
    if "<function=" in c:
        return "function_tag"
    if "<tool_call>" in c:
        return "tool_call_tag"
    if re.search(r"<parameter\b", c):
        return "parameter_tags"
    if re.search(r"<(\w+)\b", c):
        return "bare_tag"
    if c.strip():
        return "text"
    return "empty"


def call_one(cfg: Config, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "stream": False,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    elapsed = time.monotonic() - t0

    ch = d["choices"][0]
    msg = ch["message"]
    tc = msg.get("tool_calls") or []
    tool_name = tc[0]["function"]["name"] if tc else ""
    tool_args = tc[0]["function"]["arguments"] if tc else ""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = d.get("usage", {})
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": f"{elapsed:.1f}",
        "finish_reason": ch.get("finish_reason", ""),
        "success": 1 if tc else 0,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "format_class": classify(content, tc),
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "reasoning_len": len(reasoning),
        "content": content,
        "reasoning": reasoning,
    }


def existing_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add(dkey(row))
    return keys


def dkey(row: dict) -> tuple:
    return (
        row.get("phase", ""),
        row.get("config_id", ""),
        row.get("prompt_key", ""),
        row.get("sample_i", ""),
    )


def run_phase(phase: dict, out: Path, keys: set, tag: str) -> None:
    items = phase["items"]
    total = sum(i["n"] for i in items)
    done = 0
    print(f"\n[{tag}] {phase['name']}: {len(items)} configs, {total} samples")
    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for item in items:
            cfg = item["config"]
            prompt = item["prompt"]
            prompt_key = "explicit" if prompt == EXPLICIT else (
                "soft" if prompt == SOFT else f"qual:{prompt.split()[0][:12]}"
            )
            for i in range(item["n"]):
                row = {
                    "phase": phase["name"],
                    "config_id": cfg.id,
                    "temperature": cfg.temperature,
                    "top_k": cfg.top_k,
                    "top_p": cfg.top_p,
                    "max_tokens": cfg.max_tokens,
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    "sample_i": i,
                }
                if dkey(row) in keys:
                    done += 1
                    continue
                try:
                    res = call_one(cfg, prompt)
                except Exception as e:
                    print(f"  ERR {cfg.id} {prompt_key}#{i}: {e!r}")
                    continue
                row.update(res)
                writer.writerow(row)
                f.flush()
                done += 1
                stat = {
                    "parsed_tool_calls": 1,
                    "std_meta_tag": "S",
                    "function_tag": "F",
                    "tool_call_tag": "T",
                    "parameter_tags": "P",
                    "bare_tag": "B",
                    "text": "X",
                    "empty": "E",
                }.get(res["format_class"], "?")
                tok = res.get("completion_tokens", "?")
                print(
                    f"  [{done}/{total}] {cfg.id:18s} {prompt_key:8s} "
                    f"#{i} {stat} fr={res['finish_reason']:8s} tok={tok}"
                )
    print(f"  [{tag}] phase done: {done} samples")


def summarize(path: Path) -> None:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        print("empty")
        return
    groups: dict = {}
    for r in rows:
        groups.setdefault(
            (r["phase"], r["config_id"], r["prompt_key"]), []
        ).append(r)
    print(f"\n{'─'*100}")
    print(f"{'Phase':<22}{'Config':<20}{'Prompt':<10}{'N':>4}{'OK':>5}"
          f"{'%':>6}  {'finish_reasons':<30}{'classes'}")
    print(f"{'─'*100}")
    for (phase, cid, pkey), grp in sorted(groups.items()):
        n = len(grp)
        ok = sum(int(r["success"]) for r in grp)
        pct = f"{100*ok/n:.0f}%" if n else "0%"
        frs = ",".join(sorted({r["finish_reason"] for r in grp}))
        cls = ",".join(sorted({r["format_class"] for r in grp}))
        print(f"{phase:<22}{cid:<20}{pkey:<10}{n:>4}{ok:>5}{pct:>6}  {frs:<30}{cls}")
    print(f"{'─'*100}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", help="comma list of phase indices (1-5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-primary", type=int, default=10)
    ap.add_argument("--n-soft", type=int, default=8)
    ap.add_argument("--summary", metavar="CSV")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize(Path(args.summary))
        return

    phases = build_phases(args.n_primary, args.n_soft)
    if args.list:
        for i, p in enumerate(phases, 1):
            n = sum(x["n"] for x in p["items"])
            print(f"  {i}. {p['name']:<22} {len(p['items']):>3} configs, {n} samples")
        return

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"sweep_tools_{datetime.now():%Y%m%d}.csv"
    keys = existing_keys(out)
    tag = f"sweep_{datetime.now():%Y%m%d_%H%M%S}"

    if args.phases:
        idxs = [int(x) for x in args.phases.split(",")]
    else:
        idxs = range(1, len(phases) + 1)

    for i in idxs:
        run_phase(phases[i - 1], out, keys, tag)

    print(f"\nAll done → {out}")
    summarize(out)


if __name__ == "__main__":
    main()