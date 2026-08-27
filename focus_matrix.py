#!/usr/bin/env python3
"""Focus matrix: official Qwen3.6-35B-A3B sampling presets vs our tuned best,
measured under the fixed tool-call parser (non-streaming)."""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
from sweep import call_one, Config, EXPLICIT, SOFT


def run(label, cfg, prompt, n, out, w):
    pk = "explicit" if prompt == EXPLICIT else "soft"
    ok = 0
    for i in range(n):
        try:
            r = call_one(cfg, prompt)
        except Exception as e:
            print(f"  ERR {label}#{i}: {e!r}")
            continue
        row = {
            "label": label, "config_id": cfg.id, "temperature": cfg.temperature,
            "top_k": cfg.top_k, "top_p": cfg.top_p, "prompt_key": pk, "sample_i": i,
            "finish_reason": r["finish_reason"], "success": r["success"],
            "format_class": r["format_class"], "tool_name": r["tool_name"],
            "tool_args": r["tool_args"], "completion_tokens": r["completion_tokens"],
            "content": r["content"][:200],
        }
        w.writerow(row)
        ok += int(r["success"])
        print(f"  {label:32s}#{i} {'OK' if r['success']=='1' else '--'} fr={r['finish_reason']:9s}"
              f" cls={r['format_class']:16s} tok={r['completion_tokens']}")
        time.sleep(0.2)
    return ok, n


def main():
    out = f"results/focus_official_{datetime.now():%Y%m%d_%H%M%S}.csv"
    cols = ["label","config_id","temperature","top_k","top_p","prompt_key","sample_i",
            "finish_reason","success","format_class","tool_name","tool_args","completion_tokens","content"]
    N = 30
    batches = [
        ("OFFICIAL general (t1.0/p0.95)", Config(1.00, 20, 0.95), EXPLICIT),
        ("OFFICIAL coding (t0.6/p0.95)",  Config(0.60, 20, 0.95), EXPLICIT),
        ("TUNED best (t0.91/p0.98)",      Config(0.91, 20, 0.98), EXPLICIT),
        ("OFFICIAL general soft prompt",  Config(1.00, 20, 0.95), SOFT),
        ("OFFICIAL coding soft prompt",   Config(0.60, 20, 0.95), SOFT),
        ("TUNED best soft prompt",        Config(0.91, 20, 0.98), SOFT),
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        summary = []
        for label, cfg, prompt in batches:
            ok, n = run(label, cfg, prompt, N, out, w)
            summary.append((label, ok, n))
            print(f"\n  === {label}: {ok}/{n} ({100*ok/n:.1f}%) ===\n")
    print("\n===== SUMMARY =====")
    for label, ok, n in summary:
        mark = " <== BEST" if ok == max(s[1] for s in summary) else ""
        print(f"  {label:34s} {ok:3d}/{n} ({100*ok/n:5.1f}%){mark}")
    print("\nCSV:", out)


if __name__ == "__main__":
    main()