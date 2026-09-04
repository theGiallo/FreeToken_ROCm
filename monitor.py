#!/usr/bin/env python3
"""FreeToken monitor — poll /v1/stats into CSV, then show stats and plot.

Usage:
    python monitor.py                  # poll → samples_<date>.csv
    python monitor.py -o my_run.csv    # poll → custom filename
    python monitor.py --no-plot        # poll, text only (no live terminal plot)
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
VRAM_NULL_THRESHOLD = 128 * 1024**2  # 128 MB — below this a zero-tok/s sample is idle
PLOT_HISTORY = 120  # plotted points kept in the live terminal plot
PLOT_MAX_RAW = 720  # raw samples retained for the live plot (~12 min at 1 Hz)

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
        and int(row["vram_bytes"]) < VRAM_NULL_THRESHOLD
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


# ── zero-aware stats & live terminal plot ───────────────────────────
def _stats(vals):
    """(min, mean, trimmed_mean, median, std, max) over non-zero values; zeros if empty."""
    s = sorted(v for v in vals if v != 0)
    if not s:
        return (0.0,) * 6
    mn, mx = s[0], s[-1]
    mean = sum(s) / len(s)
    drop = max(1, int(len(s) * 0.10))
    core = s[drop:-drop] if len(s) - 2 * drop > 0 else s
    trimmed = sum(core) / len(core)
    median = s[len(s) // 2]
    var = sum((x - mean) ** 2 for x in s) / len(s)
    std = math.sqrt(var)
    return mn, mean, trimmed, median, std, mx


def _median(vals) -> float:
    s = sorted(v for v in vals if v != 0)
    return s[len(s) // 2] if s else 0.0


def _window_mean(window, index):
    vals = [x[index] for x in window if x[index] != 0]
    return sum(vals) / len(vals) if vals else 0.0


_BLOCKS = "▁▂▃▄▅▆▇█"


def _term_size():
    try:
        import shutil

        return shutil.get_terminal_size((100, 30))
    except Exception:
        return collections.namedtuple("size", "columns lines")(100, 30)


def _collapse_zeros(pts):
    """Keep one point per consecutive run of zeros so idle gaps don't flatten the plot."""
    out = []
    prev_zero = False
    for t, v in pts:
        z = v == 0
        if z and prev_zero:
            continue
        out.append((t, v))
        prev_zero = z
    return out[-PLOT_HISTORY:]


def _sparkline(vals, columns):
    if not vals:
        return ""
    step = max(1, len(vals) / max(1, columns))
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    out = []
    for c in range(columns):
        a = int(c * step)
        b = max(a + 1, int((c + 1) * step))
        seg = vals[a:b] or [lo]
        v = max(seg)
        out.append(_BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / rng * (len(_BLOCKS) - 1) + 0.5))])
    return "".join(out)


def _plot_lines_plotext(series, width, height):
    import plotext as plt

    fig = plt.figure

    all_x = [p[0] for s in series.values() for p in s] or [0.0]
    xmin, xmax = min(all_x), max(all_x)

    def panel(pairs):
        fig.clear()
        fig.theme("colorless")
        fig.plot_size(width, height)
        fig.ruler("x").lim(xmin, xmax)
        for name, pts in pairs:
            xs = [p[0] for p in pts] or [0.0]
            ys = [p[1] for p in pts] or [0.0]
            sig = fig.signal(xs, ys, marker="hd")
            sig.lines(True)
            if name:
                sig.label(name)
            fig.draw(sig)
        fig.ruler("x").frequency(0)
        return fig.build().string(colorless=True).splitlines()

    out = []
    for caption, pairs in (
        ("Decode (tok/s)", [("", series["decode"])]),
        ("Prefill (tok/s)", [("", series["prefill"])]),
        ("VRAM (GiB)", [("", series["vram"])]),
        ("RAM avail (GiB)", [("", series["ram"])]),
    ):
        out.append(f"  {caption}")
        out += panel(pairs)
        out.append("")
    while out and out[-1] == "":
        out.pop()
    return out


def _plot_lines_fallback(series):
    rows = [("Decode", "decode"), ("Prefill", "prefill"), ("VRAM", "vram"), ("RAM", "ram")]
    return [
        f"  {name:<7}{_sparkline([v for _, v in series[key]], 80)}"
        for name, key in rows
    ]


def _plot_lines(hist, plot_h):
    if not hist:
        return []
    try:
        import plotext
    except ImportError:
        plotext = None

    t, dt, pt, vram, ram = zip(*hist)
    series = {
        "decode": _collapse_zeros(list(zip(t, dt))),
        "prefill": _collapse_zeros(list(zip(t, pt))),
        "vram": _collapse_zeros(list(zip(t, vram))),
        "ram": _collapse_zeros(list(zip(t, ram))),
    }

    if plotext is not None and plot_h >= 15:
        ts = _term_size()
        width = max(40, min(ts.columns - 2, 160))
        height = min(12, max(4, (plot_h - 4) // 4))
        return _plot_lines_plotext(series, width, height)
    return _plot_lines_fallback(series)


# ── poll mode ────────────────────────────────────────────────────────
def poll(args):
    path = Path(args.output)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = CSV_FIELDS
    show_plot = not args.no_plot
    t0 = time.monotonic()

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
        plot_hist = collections.deque(maxlen=PLOT_MAX_RAW)
        ema = {"decode": 0.0, "prefill": 0.0, "vram": 0.0, "ram": 0.0}
        last10 = collections.deque(maxlen=10)
        all_dt = []
        all_pt = []
        last_plot_lines = []
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
                    plot_hist.append((time.monotonic() - t0, dt, pt, vram, ram))
                    if is_zero and last10:
                        last10[-1] = (seen, last10[-1][1], last10[-1][2], last10[-1][3], last10[-1][4])
                    else:
                        last10.append((seen, dt, pt, vram, ram))

                    for k, v in [("decode", dt), ("prefill", pt), ("vram", vram), ("ram", ram)]:
                        if v != 0:
                            ema[k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema[k] if ema[k] else v

                    s_dt = _stats(all_dt)
                    s_pt = _stats(all_pt)

                    w_dt = _window_mean(window, 0)
                    w_pt = _window_mean(window, 1)
                    w_vram = _window_mean(window, 2)
                    w_ram = _window_mean(window, 3)
                    wm_dt = _median([x[0] for x in window])
                    wm_pt = _median([x[1] for x in window])
                    wm_vram = _median([x[2] for x in window])
                    wm_ram = _median([x[3] for x in window])

                    if show_plot:
                        L = _term_size().lines
                        if L >= 41:
                            h = min(12, (L - 29) // 4)
                            plot_lines = _plot_lines(plot_hist, 4 + 4 * h)
                            show_last10 = True
                        elif L >= 28:
                            h = min(12, (L - 16) // 4)
                            plot_lines = _plot_lines(plot_hist, 4 + 4 * h)
                            show_last10 = False
                        else:
                            plot_lines = _plot_lines(plot_hist, 0)
                            show_last10 = 28 <= L
                    else:
                        plot_lines = []
                        show_last10 = True

                    lines = [
                        f"Polling {API} every {POLL_INTERVAL}s → {path}",
                        f"Samples: {seen}  (logged {n})    Ctrl+C to stop",
                        "",
                    ]
                    if show_plot:
                        lines += plot_lines
                        lines.append("")
                    last_plot_lines = plot_lines

                    if show_last10:
                        lines.append(
                            f"  {'#':>5}  {'Decode':>7}  {'Prefill':>7}  "
                            f"{'VRAM':>6}  {'RAM':>6}"
                        )
                        lines.append(f"  {'─'*45}")
                        for row_n, rdt, rpt, rvram, rram in last10:
                            lines.append(
                                f"  {row_n:>5}  {rdt:6.1f}t  {rpt:6.0f}t  "
                                f"{rvram:5.1f}G  {rram:5.1f}G"
                            )
                        lines.append(f"  {'─'*45}")

                    mn_dt, mean_dt, trim_dt, med_dt, std_dt, mx_dt = s_dt
                    mn_pt, mean_pt, trim_pt, med_pt, std_pt, mx_pt = s_pt
                    lines.append(
                        f"  {'AVG':<7}{mean_dt:7.1f}t  {mean_pt:7.0f}t  {'':>6}  {'':>6}"
                    )
                    lines.append(
                        f"  {'TRIM':<7}{trim_dt:7.1f}t  {trim_pt:7.0f}t  {'':>6}  {'':>6}"
                    )
                    lines.append(
                        f"  {'':7}±{std_dt:6.1f}   ±{std_pt:6.0f}   {'':>6}  {'':>6}"
                    )
                    lines.append(
                        f"  {'':7}[{mn_dt:.1f},{mx_dt:.1f}]"
                        f"  [{mn_pt:.0f},{mx_pt:.0f}]"
                        f"  {'':>6}  {'':>6}"
                    )
                    lines.append(
                        f"  {'MED':<7}{med_dt:7.1f}t  {med_pt:7.0f}t  {'':>6}  {'':>6}"
                    )
                    lines.append(
                        f"  {'WIN':<7}{w_dt:7.1f}t  {w_pt:7.0f}t  "
                        f"{w_vram:6.1f}G  {w_ram:6.1f}G"
                    )
                    lines.append(
                        f"  {'WINMED':<7}{wm_dt:7.1f}t  {wm_pt:7.0f}t  "
                        f"{wm_vram:6.1f}G  {wm_ram:6.1f}G"
                    )
                    lines.append(
                        f"  {'EMA':<7}{ema['decode']:7.1f}t  {ema['prefill']:7.0f}t  "
                        f"{ema['vram']:6.1f}G  {ema['ram']:6.1f}G"
                    )

                    print("\033[2J\033[H" + "\n".join(lines), flush=True)
                else:
                    lines = [
                        f"Polling {API} every {POLL_INTERVAL}s → {path}",
                        f"Samples: {seen}  (logged {n})    Ctrl+C to stop",
                        "",
                    ]
                    if show_plot:
                        lines += last_plot_lines
                        lines.append("")
                    lines.append("  (server not responding)")
                    print("\033[2J\033[H" + "\n".join(lines), flush=True)
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
        f"  {'Metric':<18} {'Min':>10} {'Mean':>10} {'Trim10':>10} {'Median':>10} "
        f"{'Std':>10} {'Max':>10} {'P95':>10}"
    )
    print(f"  {'─'*80}")

    for key, (label, conv) in metrics.items():
        vals = sorted(v for v in (conv(r[key]) for r in active_rows) if v != 0)
        n = len(vals)
        if n == 0:
            print(f"  {label:<18} {'0.0':>10} {'0.0':>10} {'0.0':>10} {'0.0':>10} "
                  f"{'0.0':>10} {'0.0':>10} {'0.0':>10}")
            continue
        mn = vals[0]
        mx = vals[-1]
        mean = sum(vals) / n
        drop = max(1, int(n * 0.10))
        core = vals[drop:-drop] if n - 2 * drop > 0 else vals
        trimmed = sum(core) / len(core)
        median = vals[n // 2]
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        p95_idx = int(n * 0.95)
        p95 = vals[min(p95_idx, n - 1)]
        fmt = ".1f"
        print(
            f"  {label:<18} {mn:>10{fmt}} {mean:>10{fmt}} {trimmed:>10{fmt}} {median:>10{fmt}} "
            f"{std:>10{fmt}} {mx:>10{fmt}} {p95:>10{fmt}}"
        )

    total_out = sum(int(r["completion_tokens_total"]) for r in rows)
    last = rows[-1]
    dur_s = len(rows) * POLL_INTERVAL
    decodes = [float(r["decode_tps"]) for r in rows if float(r["decode_tps"]) != 0]
    avg_decode = sum(decodes) / len(decodes) if decodes else 0.0
    print(f"  {'─'*70}")
    print(
        f"  Total output:    {total_out} tokens   "
        f"Duration: {dur_s:.0f}s   "
        f"Avg decode: {avg_decode:.1f} tok/s"
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

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"FreeToken Monitor — {path.stem}", fontsize=13)

    # decode throughput
    ax = axes[0]
    ax.plot(xs, decode, linewidth=1, label="Decode", color="#2563eb")
    ax.set_ylabel("Decode (tok/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # prefill throughput
    ax = axes[1]
    ax.plot(xs, prefill, linewidth=1, label="Prefill", color="#f59e0b")
    ax.set_ylabel("Prefill (tok/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # VRAM
    ax = axes[2]
    ax.plot(xs, vram, linewidth=1, color="#dc2626")
    ax.set_ylabel("VRAM (GiB)")
    ax.grid(True, alpha=0.3)

    # RAM
    ax = axes[3]
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
    ap.add_argument("--no-plot", action="store_true", help="Disable the live terminal plot")
    args = ap.parse_args()

    if args.csv_file:
        view(args)
    else:
        if not args.output:
            args.output = f"samples_{datetime.now():%Y%m%d_%H%M%S}.csv"
        poll(args)


if __name__ == "__main__":
    main()
