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
import itertools
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
    "input_tps",
    "vram_bytes",
    "ram_available_bytes",
    "active",
    "completed",
    "prompt_tokens_total",
    "completion_tokens_total",
    "ttft_mean_ms",
    "p95_ms",
    "last_req_id",
    "last_req_in_tps",
    "last_req_out_tps",
    "last_req_in_tokens",
    "last_req_out_tokens",
    "last_req_cached_tokens",
    "last_req_in_ms",
    "last_req_out_ms",
    "last_req_duration_ms",
]


def fetch_stats() -> dict | None:
    try:
        with urlopen(Request(API, headers={"Accept": "application/json"}), timeout=2) as r:
            return json.loads(r.read())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def read_ram() -> int:
    if sys.platform.startswith("win"):
        return _read_ram_windows()
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


def _read_ram_windows() -> int:
    """Available physical RAM (bytes) via GlobalMemoryStatusEx."""
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        co = ctypes.windll.kernel32.GlobalMemoryStatusEx
        if co(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return 0


def _is_full_zero_row(row: dict) -> bool:
    return (
        float(row["decode_tps"]) == 0.0
        and float(row["prefill_tps"]) == 0.0
        and int(row["vram_bytes"]) < VRAM_NULL_THRESHOLD
    )


def row_from_stats(d: dict) -> dict:
    lr = (d.get("requests") or {}).get("last_request")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decode_tps": f"{d['throughput']['decode_tps']:.2f}",
        "prefill_tps": f"{d['throughput']['prefill_tps']:.2f}",
        "input_tps": f"{d['throughput']['input_tps']:.2f}",
        "vram_bytes": d["vram_bytes"],
        "ram_available_bytes": read_ram(),
        "active": d["requests"]["active"],
        "completed": d["requests"]["completed"],
        "prompt_tokens_total": d["requests"]["prompt_tokens_total"],
        "completion_tokens_total": d["requests"]["completion_tokens_total"],
        "ttft_mean_ms": f"{d['requests']['ttft_mean_ms']:.1f}",
        "p95_ms": f"{d['requests']['p95_ms']:.1f}",
    }
    if lr:
        row["last_req_id"] = lr["id"]
        row["last_req_in_tps"] = f"{lr['input_tps']:.2f}"
        row["last_req_out_tps"] = f"{lr['output_tps']:.2f}"
        row["last_req_in_tokens"] = lr["input_tokens"]
        row["last_req_out_tokens"] = lr["output_tokens"]
        row["last_req_in_ms"] = lr["input_ms"]
        row["last_req_out_ms"] = lr["output_ms"]
        row["last_req_cached_tokens"] = lr["cached_tokens"]
        row["last_req_duration_ms"] = lr["duration_ms"]
    else:
        row.update({k: "" for k in ("last_req_id", "last_req_in_tps", "last_req_out_tps",
                                    "last_req_in_tokens", "last_req_out_tokens",
                                    "last_req_cached_tokens", "last_req_in_ms",
                                    "last_req_out_ms", "last_req_duration_ms")})
    return row


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


def _sparkline(vals, columns, skip_zero=False):
    if not vals:
        return ""
    if skip_zero:
        vals = [v for v in vals if v != 0]
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

    def panel(pairs):
        fig.clear()
        fig.theme("colorless")
        fig.plot_size(width, height)
        for name, pts, skip_zero in pairs:
            if skip_zero:
                pts = [p for p in pts if p[1] != 0]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            sig = fig.signal(xs, ys, marker="hd")
            sig.lines(True)
            if name:
                sig.label(name)
            fig.draw(sig)
        fig.ruler("x").frequency(0)
        return fig.build().string(colorless=True).splitlines()

    out = []
    for caption, pairs in (
        ("Decode (tok/s)", [("", series["decode"], True)]),
        ("Prefill (tok/s)", [("", series["prefill"], True)]),
        ("Input (tok/s)", [("", series["input"], True)]),
        ("VRAM (GiB)", [("", series["vram"], True)]),
        ("RAM avail (GiB)", [("", series["ram"], False)]),
    ):
        out.append(f"  {caption}")
        out += panel(pairs)
        out.append("")
    while out and out[-1] == "":
        out.pop()
    return out


def _plot_lines_fallback(series):
    rows = [("Decode", "decode", True), ("Prefill", "prefill", True),
            ("Input", "input", True), ("VRAM", "vram", True), ("RAM", "ram", False)]
    return [
        f"  {name:<7}{_sparkline([v for _, v in series[key]], 80, skip_zero)}"
        for name, key, skip_zero in rows
    ]


def _plot_lines(hist, plot_h):
    if not hist:
        return []
    try:
        import plotext
    except ImportError:
        plotext = None

    # Live plot is a sliding window over the last PLOT_HISTORY raw samples, x = the
    # sample's position within the window. Index-based x (not wall-clock) makes the
    # plot scroll one column per fetched sample even when the server answers
    # sparsely, so a steady value draws as a flat line forming on the right edge
    # while older samples move out on the left. Every panel shares the same window,
    # so the x axes stay aligned without forcing xlim (which crashes plotext 6 when
    # a series collapses to a single point inside a wide window).
    win = list(itertools.islice(hist, max(0, len(hist) - PLOT_HISTORY), None))
    _, dt, pt, it, vram, ram = zip(*win)
    idx = list(range(len(win)))
    series = {
        "decode": list(zip(idx, dt)),
        "prefill": list(zip(idx, pt)),
        "input": list(zip(idx, it)),
        "vram": list(zip(idx, vram)),
        "ram": list(zip(idx, ram)),
    }

    if plotext is not None and plot_h >= 18:
        ts = _term_size()
        width = max(40, min(ts.columns - 2, 160))
        height = min(12, max(4, (plot_h - 5) // 5))
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
        ema = {"decode": 0.0, "prefill": 0.0, "input": 0.0, "vram": 0.0, "ram": 0.0}
        last10 = collections.deque(maxlen=10)
        all_dt = []
        all_pt = []
        all_it = []
        last_plot_lines = []
        # Per-request recap: key every finished request by (instance_id, uid) so a request
        # whose /v1/stats the poller sees more than once is counted exactly once. The
        # aggregate is a duration-weighted average (llama.cpp server_metrics style): sum of
        # tokens over sum of phase times, so a request's contribution scales with how long
        # its input/output phases actually ran. Input tokens = prompt minus cache hits;
        # output tokens = n_gen - 1 (the first token rides the prompt batch's logits).
        last_req_key = None
        req_n = 0
        req_in_proc = 0          # sum of (input_tokens - cached_tokens)
        req_out_eff = 0          # sum of max(0, output_tokens - 1)
        req_in_ms = 0            # sum of input_ms
        req_out_ms = 0           # sum of output_ms
        cur_req = None
        try:
            while True:
                d = fetch_stats()
                if d:
                    row = row_from_stats(d)
                    # Per-request recaps: key each finished request by (instance_id, uid) so
                    # the same request seen across polls is counted exactly once.
                    lr = (d.get("requests") or {}).get("last_request")
                    inst = d.get("instance_id")
                    if lr:
                        key = (inst, lr["id"])
                        if key != last_req_key:
                            last_req_key = key
                            req_n += 1
                            req_in_proc += max(0, lr["input_tokens"] - lr["cached_tokens"])
                            req_out_eff += max(0, lr["output_tokens"] - 1)
                            req_in_ms += lr["input_ms"]
                            req_out_ms += lr["output_ms"]
                        cur_req = lr
                    is_zero = _is_full_zero_row(row)
                    if not (is_zero and prev_zero):
                        writer.writerow(row)
                        f.flush()
                        n += 1
                    prev_zero = is_zero
                    seen += 1
                    dt = float(row["decode_tps"])
                    pt = float(row["prefill_tps"])
                    it = float(row["input_tps"])
                    vram = int(row["vram_bytes"]) / 1024**3
                    ram = int(row["ram_available_bytes"]) / 1024**3

                    all_dt.append(dt)
                    all_pt.append(pt)
                    all_it.append(it)
                    window.append((dt, pt, it, vram, ram))
                    plot_hist.append((time.monotonic() - t0, dt, pt, it, vram, ram))
                    # recap of the last completed request (id + its averages), filled in
                    # once that request finishes; empty until the first completion.
                    lrid = str(lr["id"]) if lr else ""
                    rin = lr["input_tps"] if lr else 0.0
                    rout = lr["output_tps"] if lr else 0.0
                    rec = (seen, dt, pt, it, vram, ram, lrid, rin, rout)
                    if is_zero and last10:
                        # Slide the index forward but keep the idle sample's real values,
                        # so the row doesn't report the last prefill burst as current.
                        prev = last10[-1]
                        last10[-1] = (seen, prev[1], prev[2], prev[3], prev[4], prev[5],
                                      prev[6], prev[7], prev[8])
                    else:
                        last10.append(rec)

                    for k, v in [("decode", dt), ("prefill", pt), ("input", it),
                                 ("vram", vram), ("ram", ram)]:
                        if v != 0:
                            ema[k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema[k] if ema[k] else v

                    s_dt = _stats(all_dt)
                    s_pt = _stats(all_pt)
                    s_it = _stats(all_it)

                    w_dt = _window_mean(window, 0)
                    w_pt = _window_mean(window, 1)
                    w_it = _window_mean(window, 2)
                    w_vram = _window_mean(window, 3)
                    w_ram = _window_mean(window, 4)
                    wm_dt = _median([x[0] for x in window])
                    wm_pt = _median([x[1] for x in window])
                    wm_it = _median([x[2] for x in window])
                    wm_vram = _median([x[3] for x in window])
                    wm_ram = _median([x[4] for x in window])

                    if show_plot:
                        L = _term_size().lines
                        if L >= 44:
                            h = min(12, (L - 36) // 5)
                            plot_lines = _plot_lines(plot_hist, 5 + 5 * h)
                            show_last10 = True
                        elif L >= 30:
                            h = min(12, (L - 20) // 5)
                            plot_lines = _plot_lines(plot_hist, 5 + 5 * h)
                            show_last10 = False
                        else:
                            plot_lines = _plot_lines(plot_hist, 0)
                            show_last10 = 30 <= L
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
                            f"{'Input':>7}  {'VRAM':>6}  {'RAM avail':>10}  "
                            f"{'REQ':>5}  {'in-avg':>7}  {'out-avg':>7}"
                        )
                        lines.append(f"  {'─'*76}")
                        for row in last10:
                            row_n, rdt, rpt, rit, rvram, rram, lrid, rin, rout = row
                            lines.append(
                                f"  {row_n:>5}  {rdt:6.1f}t  {rpt:6.0f}t  "
                                f"{rit:6.0f}t  {rvram:5.1f}G  {rram:5.1f}G  "
                                f"{lrid:>5}  {rin:6.1f}t  {rout:6.1f}t"
                            )
                        lines.append(f"  {'─'*76}")

                    mn_dt, mean_dt, trim_dt, med_dt, std_dt, mx_dt = s_dt
                    mn_pt, mean_pt, trim_pt, med_pt, std_pt, mx_pt = s_pt
                    mn_it, mean_it, _, med_it, _, mx_it = s_it
                    lines.append(
                        f"  {'AVG':<7}{mean_dt:7.1f}t  {mean_pt:7.0f}t  "
                        f"{mean_it:7.0f}t  {'':>6}  {'':>10}"
                    )
                    lines.append(
                        f"  {'TRIM':<7}{trim_dt:7.1f}t  {trim_pt:7.0f}t  "
                        f"{'':>7}  {'':>6}  {'':>10}"
                    )
                    lines.append(
                        f"  {'':7}±{std_dt:6.1f}   ±{std_pt:6.0f}   {'':>7}  {'':>6}  {'':>10}"
                    )
                    lines.append(
                        f"  {'':7}[{mn_dt:.1f},{mx_dt:.1f}]"
                        f"  [{mn_pt:.0f},{mx_pt:.0f}]"
                        f"  [{mn_it:.0f},{mx_it:.0f}]"
                        f"  {'':>6}  {'':>10}"
                    )
                    lines.append(
                        f"  {'MED':<7}{med_dt:7.1f}t  {med_pt:7.0f}t  "
                        f"{med_it:7.0f}t  {'':>6}  {'':>10}"
                    )
                    lines.append(
                        f"  {'WIN':<7}{w_dt:7.1f}t  {w_pt:7.0f}t  "
                        f"{w_it:7.0f}t  {w_vram:6.1f}G  {w_ram:7.1f}G"
                    )
                    lines.append(
                        f"  {'WINMED':<7}{wm_dt:7.1f}t  {wm_pt:7.0f}t  "
                        f"{wm_it:7.0f}t  {wm_vram:6.1f}G  {wm_ram:7.1f}G"
                    )
                    lines.append(
                        f"  {'EMA':<7}{ema['decode']:7.1f}t  {ema['prefill']:7.0f}t  "
                        f"{ema['input']:7.0f}t  {ema['vram']:6.1f}G  {ema['ram']:7.1f}G"
                    )

                    if cur_req is not None:
                        lines.append(
                            f"  LAST REQ  #{cur_req['id']}"
                            f"  in {cur_req['input_tokens']} tok"
                            f" @{cur_req['input_tps']:.1f}/s"
                            f"  out {cur_req['output_tokens']} tok"
                            f" @{cur_req['output_tps']:.1f}/s"
                            f"  {cur_req['duration_ms'] / 1e3:.1f}s"
                        )
                    if req_n:
                        # Duration-weighted aggregate: tokens/time. Combined weight uses the
                        # sum of both phase times (input+output) as the requested denominator.
                        in_s = req_in_ms / 1e3
                        out_s = req_out_ms / 1e3
                        w_in = req_in_proc / in_s if in_s > 0 else 0.0
                        w_out = req_out_eff / out_s if out_s > 0 else 0.0
                        both_s = in_s + out_s
                        w_both = (req_in_proc + req_out_eff) / both_s if both_s > 0 else 0.0
                        lines.append(
                            f"  REQS      {req_n}  w-avg in {w_in:6.1f}/s"
                            f"  w-avg out {w_out:6.1f}/s"
                            f"  w-avg all {w_both:6.1f}/s"
                        )
                        lines.append(
                            f"  {'':8}in {req_in_proc} proc tok"
                            f"  out {req_out_eff} tok"
                            f"  in {in_s:.1f}s / out {out_s:.1f}s"
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
        "input_tps": ("Input tok/s", float),
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
        raw = [conv(r.get(key, 0)) for r in active_rows]
        # Skip zeros for everything except RAM: 0 tok/s is idle noise, but 0 avail RAM is a
        # genuine reading that belongs in min (the per-row read is never a 0 on a live host).
        vals = sorted(v for v in raw if v != 0) if key != "ram_available_bytes" else sorted(raw)
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

    # ── per-request recap aggregates (filled in once each request completed) ──
    recaps = {}  # last_req_id -> first row carrying that request's recap
    for r in rows:
        rid = r.get("last_req_id", "")
        if rid and rid not in recaps:
            recaps[rid] = r
    if recaps:
        n_req = len(recaps)
        in_proc = sum(max(0, int(x["last_req_in_tokens"]) - int(x["last_req_cached_tokens"]))
                      for x in recaps.values())
        out_eff = sum(max(0, int(x["last_req_out_tokens"]) - 1) for x in recaps.values())
        in_ms = sum(int(x["last_req_in_ms"]) for x in recaps.values())
        out_ms = sum(int(x["last_req_out_ms"]) for x in recaps.values())
        in_s, out_s = in_ms / 1e3, out_ms / 1e3
        w_in = in_proc / in_s if in_s else 0.0
        w_out = out_eff / out_s if out_s else 0.0
        both_s = in_s + out_s
        w_both = (in_proc + out_eff) / both_s if both_s else 0.0
        print(f"  {'─'*70}")
        print(
            f"  Requests:      {n_req}   "
            f"w-avg in {w_in:.1f}/s   w-avg out {w_out:.1f}/s   w-avg all {w_both:.1f}/s"
        )
        print(
            f"  {'':>14}in {in_proc} proc tok   out {out_eff} tok"
            f"   in {in_s:.1f}s / out {out_s:.1f}s"
        )
        print(f"  {'─'*70}")
    else:
        print(f"  (no per-request recap data in this CSV; regenerate with the current monitor)")

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
    input_ = [float(r.get("input_tps", 0)) for r in active_rows]
    vram = [int(r["vram_bytes"]) / 1024**3 for r in active_rows]
    ram = [int(r["ram_available_bytes"]) / 1024**3 for r in active_rows]

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
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

    # input throughput (scheduler-measured prefill rate)
    ax = axes[2]
    ax.plot(xs, input_, linewidth=1, label="Input", color="#7c3aed")
    ax.set_ylabel("Input (tok/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # VRAM
    ax = axes[3]
    ax.plot(xs, vram, linewidth=1, color="#dc2626")
    ax.set_ylabel("VRAM (GiB)")
    ax.grid(True, alpha=0.3)

    # RAM
    ax = axes[4]
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
