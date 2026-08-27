#!/usr/bin/env python3
"""FreeToken monitor — poll /v1/stats into CSV, then show stats and plot.

Usage:
    python monitor.py                  # poll → samples_<date>.csv
    python monitor.py -o my_run.csv    # poll → custom filename
    python monitor.py --view run.csv   # load CSV, print stats, show plot
    python monitor.py --view run.csv --save-plot run.png  # save plot to file
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

EMA_ALPHA = 0.05  # ~60-sample half-life
WINDOW = 60  # seconds for moving average

API = os.environ.get("FT_STATS_URL", "http://127.0.0.1:1919/v1/stats")
POLL_INTERVAL = 1.0

CSV_FIELDS = [
    "timestamp",
    "decode_tps",
    "prefill_tps",
    "vram_bytes",
    "ram_available_bytes",
    "active",
    "completed",
    "prompt_tokens_total",
    "completion_tokens_total",
    "ttft_mean_ms",
    "p95_ms",
]


def fetch_stats() -> dict | None:
    try:
        with urlopen(Request(API, headers={"Accept": "application/json"}), timeout=2) as r:
            return json.loads(r.read())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def read_ram() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


def _is_full_zero_row(row: dict) -> bool:
    return (
        float(row["decode_tps"]) == 0.0
        and float(row["prefill_tps"]) == 0.0
        and float(row["ttft_mean_ms"]) == 0.0
        and float(row["p95_ms"]) == 0.0
        and int(row["active"]) == 0
    )


def row_from_stats(d: dict) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decode_tps": f"{d['throughput']['decode_tps']:.2f}",
        "prefill_tps": f"{d['throughput']['prefill_tps']:.2f}",
        "vram_bytes": d["vram_bytes"],
        "ram_available_bytes": read_ram(),
        "active": d["requests"]["active"],
        "completed": d["requests"]["completed"],
        "prompt_tokens_total": d["requests"]["prompt_tokens_total"],
        "completion_tokens_total": d["requests"]["completion_tokens_total"],
        "ttft_mean_ms": f"{d['requests']['ttft_mean_ms']:.1f}",
        "p95_ms": f"{d['requests']['p95_ms']:.1f}",
    }


# ── poll mode ────────────────────────────────────────────────────────
def poll(args):
    path = Path(args.output)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = CSV_FIELDS

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        print(f"Polling {API} every {POLL_INTERVAL}s → {path}")
        print("Ctrl+C to stop\n")
        n = 0  # samples written to CSV
        seen = 0  # samples fetched overall
        prev_zero = False  # previous fetched sample was a full-zero idle sample
        window = collections.deque(maxlen=int(WINDOW / POLL_INTERVAL))
        ema = {"decode": 0.0, "prefill": 0.0, "vram": 0.0, "ram": 0.0}
        last10 = collections.deque(maxlen=10)
        all_dt = []
        all_pt = []
        try:
            while True:
                d = fetch_stats()
                if d:
                    row = row_from_stats(d)
                    is_zero = _is_full_zero_row(row)
                    if not (is_zero and prev_zero):
                        writer.writerow(row)
                        f.flush()
                        n += 1
                    prev_zero = is_zero
                    seen += 1
                    dt = float(row["decode_tps"])
                    pt = float(row["prefill_tps"])
                    vram = int(row["vram_bytes"]) / 1024**3
                    ram = int(row["ram_available_bytes"]) / 1024**3

                    all_dt.append(dt)
                    all_pt.append(pt)
                    window.append((dt, pt, vram, ram))
                    last10.append((seen, dt, pt, vram, ram))

                    for k, v in [("decode", dt), ("prefill", pt), ("vram", vram), ("ram", ram)]:
                        ema[k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema[k] if ema[k] else v

                    def _stats(vals):
                        if not vals:
                            return 0, 0, 0, 0, 0, 0
                        s = sorted(vals)
                        mn, mx = s[0], s[-1]
                        mean = sum(s) / len(s)
                        median = s[len(s) // 2]
                        var = sum((x - mean) ** 2 for x in s) / len(s)
                        std = math.sqrt(var)
                        return mn, mean, median, std, mx

                    s_dt = _stats(all_dt)
                    s_pt = _stats(all_pt)

                    w_dt = sum(x[0] for x in window) / len(window) if window else 0
                    w_pt = sum(x[1] for x in window) / len(window) if window else 0
                    w_vram = sum(x[2] for x in window) / len(window) if window else 0
                    w_ram = sum(x[3] for x in window) / len(window) if window else 0

                    print("\033[2J\033[H", end="", flush=True)
                    print(f"Polling {API} every {POLL_INTERVAL}s → {path}")
                    print(f"Samples: {seen}  (logged {n})    Ctrl+C to stop\n")

                    print(
                        f"  {'#':>5}  {'Decode':>7}  {'Prefill':>7}  "
                        f"{'VRAM':>6}  {'RAM':>6}"
                    )
                    print(f"  {'─'*45}")
                    for row_n, rdt, rpt, rvram, rram in last10:
                        print(
                            f"  {row_n:>5}  {rdt:6.1f}t  {rpt:6.0f}t  "
                            f"{rvram:5.1f}G  {rram:5.1f}G"
                        )
                    print(f"  {'─'*45}")

                    def _row(label, dt_tup, pt_tup):
                        mn, mean, med, std, mx = dt_tup
                        pmn, pmean, pmed, pstd, pmx = pt_tup
                        print(
                            f"  {label:<5}  {mean:6.1f}t  {pmean:6.0f}t  "
                            f"{'':>6}  {'':>6}"
                        )
                        print(
                            f"  {'':5}  ±{std:5.1f}   ±{pstd:5.0f}   "
                            f"{'':>6}  {'':>6}"
                        )
                        print(
                            f"  {'':5}  [{mn:.1f},{mx:.1f}]"
                            f"  [{pmn:.0f},{pmx:.0f}]"
                            f"  {'':>6}  {'':>6}"
                        )

                    _row("AVG", s_dt, s_pt)

                    print(
                        f"  {'WIN':>5}  {w_dt:6.1f}t  {w_pt:6.0f}t  "
                        f"{w_vram:5.1f}G  {w_ram:5.1f}G"
                    )
                    print(
                        f"  {'EMA':>5}  {ema['decode']:6.1f}t  {ema['prefill']:6.0f}t  "
                        f"{ema['vram']:5.1f}G  {ema['ram']:5.1f}G"
                    )
                else:
                    print("\033[2J\033[H", end="", flush=True)
                    print(f"Polling {API} every {POLL_INTERVAL}s → {path}")
                    print(f"Samples: {seen}  (logged {n})    Ctrl+C to stop\n")
                    print("  (server not responding)", flush=True)
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print(f"\nWrote {n} samples to {path} ({seen} polled, {seen - n} zero-idle skipped)")


# ── view mode ────────────────────────────────────────────────────────
def _safe_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def view(args):
    path = Path(args.csv_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # ── stats ────────────────────────────────────────────────────────
    metrics = {
        "decode_tps": ("Decode tok/s", float),
        "prefill_tps": ("Prefill tok/s", float),
        "vram_bytes": ("VRAM GiB", lambda v: int(v) / 1024**3),
        "ram_available_bytes": ("RAM avail GiB", lambda v: int(v) / 1024**3),
        "ttft_mean_ms": ("TTFT ms", float),
        "p95_ms": ("p95 ms", float),
    }

    # Filter out full-zero (idle) samples so stats reflect actual load
    active_rows = [r for r in rows if not _is_full_zero_row(r)]
    if not active_rows:
        active_rows = rows

    print(f"{'─'*72}")
    print(f"  File: {path}  ({len(rows)} samples, {len(active_rows)} non-idle)")
    print(f"  From: {rows[0]['timestamp']}  to  {rows[-1]['timestamp']}")
    print(f"{'─'*72}")
    print(
        f"  {'Metric':<18} {'Min':>10} {'Mean':>10} {'Median':>10} "
        f"{'Std':>10} {'Max':>10} {'P95':>10}"
    )
    print(f"  {'─'*70}")

    for key, (label, conv) in metrics.items():
        vals = sorted(conv(r[key]) for r in active_rows)
        n = len(vals)
        mn = vals[0]
        mx = vals[-1]
        mean = sum(vals) / n
        median = vals[n // 2]
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        p95_idx = int(n * 0.95)
        p95 = vals[min(p95_idx, n - 1)]
        fmt = ".1f"
        print(
            f"  {label:<18} {mn:>10{fmt}} {mean:>10{fmt}} {median:>10{fmt}} "
            f"{std:>10{fmt}} {mx:>10{fmt}} {p95:>10{fmt}}"
        )

    total_out = sum(int(r["completion_tokens_total"]) for r in rows)
    last = rows[-1]
    dur_s = len(rows) * POLL_INTERVAL
    print(f"  {'─'*70}")
    print(
        f"  Total output:    {total_out} tokens   "
        f"Duration: {dur_s:.0f}s   "
        f"Avg decode: {sum(float(r['decode_tps']) for r in rows)/len(rows):.1f} tok/s"
    )
    print(f"{'─'*60}")

    # ── plot ─────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        print("Install with: pip install matplotlib")
        return

    t0 = datetime.fromisoformat(active_rows[0]["timestamp"])
    xs = [(datetime.fromisoformat(r["timestamp"]) - t0).total_seconds() for r in active_rows]
    decode = [float(r["decode_tps"]) for r in active_rows]
    prefill = [float(r["prefill_tps"]) for r in active_rows]
    vram = [int(r["vram_bytes"]) / 1024**3 for r in active_rows]
    ram = [int(r["ram_available_bytes"]) / 1024**3 for r in active_rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"FreeToken Monitor — {path.stem}", fontsize=13)

    # throughput
    ax = axes[0]
    ax.plot(xs, decode, linewidth=1, label="Decode", color="#2563eb")
    ax.plot(xs, prefill, linewidth=1, label="Prefill", color="#f59e0b")
    ax.set_ylabel("tok/s")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # VRAM
    ax = axes[1]
    ax.plot(xs, vram, linewidth=1, color="#dc2626")
    ax.set_ylabel("VRAM (GiB)")
    ax.grid(True, alpha=0.3)

    # RAM
    ax = axes[2]
    ax.plot(xs, ram, linewidth=1, color="#16a34a")
    ax.set_ylabel("RAM avail (GiB)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save_plot:
        out = Path(args.save_plot)
        fig.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    else:
        plt.savefig(path.with_suffix(".png"), dpi=150)
        print(f"\nPlot saved to {path.with_suffix('.png')}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="FreeToken monitor")
    ap.add_argument("--view", dest="csv_file", metavar="CSV", help="View a saved CSV")
    ap.add_argument("-o", "--output", metavar="CSV", help="Output CSV path")
    ap.add_argument("--save-plot", metavar="PNG", help="Save plot to PNG")
    args = ap.parse_args()

    if args.csv_file:
        view(args)
    else:
        if not args.output:
            args.output = f"samples_{datetime.now():%Y%m%d_%H%M%S}.csv"
        poll(args)


if __name__ == "__main__":
    main()
