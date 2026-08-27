#!/usr/bin/env python3
"""Write focus-matrix results (markdown + bar plot) to results/."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/focus_official_20260828_004018.csv"
PLOT = "results/focus_official_matrix.png"
MD = "results/focus_official_results.md"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(r)

order = [
    "OFFICIAL general (t1.0/p0.95)",
    "OFFICIAL coding (t0.6/p0.95)",
    "TUNED best (t0.91/p0.98)",
    "OFFICIAL general soft prompt",
    "OFFICIAL coding soft prompt",
    "TUNED best soft prompt",
]

def success(r):
    try:
        args = json.loads(r["tool_args"] or "{}")
        return bool(args.get("command") or args.get("path"))
    except Exception:
        return False

summary = []
for lab in order:
    grp = by[lab]
    n = len(grp)
    ok = sum(int(r["success"] == "1") for r in grp)
    valid = sum(success(r) for r in grp)
    classes = Counter(r["format_class"] for r in grp)
    fr = Counter(r["finish_reason"] for r in grp)
    summary.append(dict(label=lab, n=n, ok=ok, pct=100 * ok / n,
                        valid=valid, classes=classes, fr=fr))

lines = []
lines.append("# Focus Matrix Results — FreeToken qwen3.6-35b-a3b (fixed parser)\n")
lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
lines.append("Measured with the tool-call parser fix live (non-streaming). "
             "`presence_penalty`/`min_p` are parsed by the API but are **not applied** "
             "by the FreeToken sampler yet (no-op).\n")
lines.append("| Config | N | Success | Valid args | Classes | Finish |")
lines.append("|---|---|---:|---:|---|---|")
for s in summary:
    cls = ", ".join(f"{k}:{v}" for k, v in sorted(s["classes"].items()))
    frs = ", ".join(f"{k}:{v}" for k, v in sorted(s["fr"].items()))
    lines.append(f"| {s['label']} | {s['n']} | {s['ok']}/{s['n']} ({s['pct']:.1f}%) | "
                 f"{s['valid']}/{s['n']} | {cls} | {frs} |")

with open(MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", MD)

# Plot
fig, ax = plt.subplots(figsize=(10, 5.5))
labels = [s["label"].replace(" soft prompt", "\n(soft prompt)") for s in summary]
pcts = [s["pct"] for s in summary]
colors = []
for s in summary:
    if s["pct"] == 100:
        colors.append("#2e7d32")   # green
    elif s["pct"] >= 90:
        colors.append("#f9a825")   # amber
    else:
        colors.append("#c62828")   # red
bars = ax.bar(labels, pcts, color=colors)
ax.axhline(50, color="gray", ls="--", lw=0.8, label="original baseline")
ax.set_ylabel("Tool-call success (%)")
ax.set_ylim(0, 105)
ax.set_title("Focus Matrix: Official Qwen3.6 presets vs Tuned best (fixed parser)")
for b, s in zip(bars, summary):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"{s['ok']}/{s['n']}", ha="center", va="bottom", fontsize=9)
ax.legend(loc="upper left")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(PLOT, dpi=130)
print("wrote", PLOT)
