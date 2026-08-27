#!/usr/bin/env python3
"""Analyze sweep CSVs and produce plots + a consolidated summary table.

Usage:
    python3 analysis.py results/sweep_tools_20260827.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, newline="")))


def pct(rows) -> tuple[int, int, float]:
    n = len(rows)
    ok = sum(1 for r in rows if r["success"] == "1")
    return ok, n, (100 * ok / n if n else 0.0)


def series(rows, key: str, transform=lambda v: float(v)) -> tuple[list[float], list[float]]:
    groups = defaultdict(list)
    for r in rows:
        groups[transform(r[key])].append(r)
    xs = sorted(groups)
    return xs, [pct(groups[x])[2] for x in xs]


def plot_series(title, xlabel, series_data, filename) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (xs, ys) in series_data:
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tool-call success %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if series_data:
        ax.legend(fontsize=9)
    ax.text(
        0.99, 0.01,
        "thinking ON (server default), tools passed in request",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "results" / filename
    fig.savefig(out, dpi=150)
    print(f"  plot -> {out}")


def plot_heatmap(rows, filename, title) -> None:
    if plt is None:
        return
    cells: dict[tuple[float, float], tuple[int, int]] = {}
    for r in rows:
        key = (round(float(r["temperature"]), 2), round(float(r["top_p"]), 2))
        ok, n = cells.get(key, (0, 0))
        cells[key] = (ok + int(r["success"]), n + 1)
    ts = sorted({k[0] for k in cells})
    ps = sorted({k[1] for k in cells})
    import numpy as np

    Z = np.full((len(ps), len(ts)), np.nan)
    for (t, p), (ok, n) in cells.items():
        Z[ps.index(p), ts.index(t)] = 100 * ok / n
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(ts)), [f"{t:.2f}" for t in ts])
    ax.set_yticks(range(len(ps)), [f"{p:.2f}" for p in ps])
    ax.set_xlabel("temperature")
    ax.set_ylabel("top_p")
    ax.set_title(title)
    for i in range(len(ps)):
        for j in range(len(ts)):
            v = Z[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="tool-call success %")
    ax.text(
        0.99, 0.01,
        "thinking ON (server default), tools passed in request",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="white",
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "results" / filename
    fig.savefig(out, dpi=150)
    print(f"  plot -> {out}")


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_tools_20260827.csv"
    path = Path(__file__).resolve().parent / src
    rows = load(path)
    if not rows:
        print("no rows")
        return

    p1 = [r for r in rows if r["phase"] == "phase1_temp_primary" and r["prompt_key"] == "explicit"]
    p3 = [r for r in rows if r["phase"] == "phase3_topk"]
    p4 = [r for r in rows if r["phase"] == "phase4_topp"]
    p5 = [r for r in rows if r["phase"] == "phase5_quality"]
    g6 = [r for r in rows if r["phase"] == "phase6_tp_grid"]

    print(f"Source: {path}  ({len(rows)} samples)")

    print("\n## Tool-call success by temperature  (explicit prompt, k=20, p=0.95)")
    xs, ys = series(p1, "temperature", lambda v: round(float(v), 2))
    for x, y in zip(xs, ys):
        ok, n, _ = pct([r for r in p1 if abs(float(r["temperature"]) - x) < 1e-9])
        print(f"  t={x:5.2f}  {ok:>2}/{n:<2} = {y:5.1f}%")

    print("\n## By top_k  (t=1.0, p=0.95)")
    for x, y in zip(*series(p3, "top_k", lambda v: int(v))):
        ok, n, _ = pct([r for r in p3 if int(r["top_k"]) == int(x)])
        print(f"  k={int(x):>4}  {ok:>2}/{n:<2} = {y:5.1f}%")

    print("\n## By top_p  (t=1.0, k=20)")
    for x, y in zip(*series(p4, "top_p", lambda v: round(float(v), 2))):
        ok, n, _ = pct([r for r in p4 if abs(float(r["top_p"]) - x) < 1e-9])
        print(f"  p={x:5.2f}  {ok:>2}/{n:<2} = {y:5.1f}%")

    print("\n## Quality (non-tool) prompt output shape by temp")
    for cid in sorted({r["config_id"] for r in p5}):
        grp = [r for r in p5 if r["config_id"] == cid]
        classes = defaultdict(int)
        for r in grp:
            classes[r["format_class"]] += 1
        avg_len = sum(len(r["content"]) for r in grp) / len(grp) if grp else 0
        print(f"  {cid:<24} {len(grp):>2} samples  classes={dict(classes)}  avg_content_len={avg_len:.0f}")

    print("\n## Finish reasons (all tool-prompt samples)")
    fr = defaultdict(int)
    for r in rows:
        if r["prompt_key"].startswith("qual"):
            continue
        fr[r["finish_reason"]] += 1
    print(f"  {dict(fr)}")

    if g6:
        print("\n## t x p grid  (k=20, explicit prompt, n=20)")
        best = (None, -1)
        for cid in sorted({r["config_id"] for r in g6}):
            grp = [r for r in g6 if r["config_id"] == cid]
            ok, n, _ = pct(grp)
            mark = " <-- best" if ok > best[1] else ""
            if ok > best[1]:
                best = (cid, ok)
            print(f"  {cid:<20} {ok:>2}/{n:<2} = {100*ok/n:5.1f}%{mark}")
        if best[0]:
            print(f"  best: {best[0]} ({best[1]} successes)")

    if plt:
        plot_series(
            "Tool-call success vs temperature (explicit prompt)",
            "temperature",
            [("k=20, p=0.95", series(p1, "temperature", lambda v: round(float(v), 2)))],
            "temp.png",
        )
        plot_series(
            "Tool-call success vs top_k (t=1.0, p=0.95)",
            "top_k",
            [("k", series(p3, "top_k", lambda v: int(v)))],
            "topk.png",
        )
        plot_series(
            "Tool-call success vs top_p (t=1.0, k=20)",
            "top_p",
            [("p", series(p4, "top_p", lambda v: round(float(v), 2)))],
            "topp.png",
        )
        if g6:
            plot_heatmap(
                g6,
                "tp_grid.png",
                "Tool-call success % vs temperature x top_p (k=20)",
            )
    else:
        print("\n(matplotlib not available; plots skipped)")


if __name__ == "__main__":
    main()