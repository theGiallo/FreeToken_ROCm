#!/usr/bin/env python3
"""End-to-end GPU smoke test for the --kv-persist feature.

Workload mirrors the pi "duck hunt" session: a long deterministic chat context (built from
the duck_hunt workspace files, so both launches tokenize identically) is sent to a freshly
started ft server. The test:

  1. launches ft with a budget that leaves --reserve GiB of VRAM free,
  2. sends the completion and cold-prefills the whole context, then SIGTERMs the server
     and checks the snapshot (meta/tree/kv/gdn.bin) was written,
  3. relaunches ft with the same args, checks the snapshot was loaded, sends the SAME
     completion again, and checks the parked prefix was materialized (usage.cached_tokens
     near the full prompt, much lower ttft / prefill_tps sampled from /v1/stats).

Run inside WSL against the ROCm box:

  cd /mnt/f/programming/llm/FreeToken
  $HOME/.freetoken/venv/bin/python scripts/kv_persist_smoke.py [flags]

Exit code 0 = the feature kicked in end to end; 1 = a check failed or an error occurred.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MIN_PY = (3, 10)
if sys.version_info < MIN_PY:
    sys.exit(f"need Python {'.'.join(map(str, MIN_PY))}+")

# ---- defaults matching launch_qwen35.sh / ft_serve_qwen35.sh -------------------------
DEF_MODEL = "/home/thegiallo/models/qwen3.6-35b-a3b.gguf"
DEF_FT = "/home/thegiallo/.freetoken/venv/bin/ft"
DEF_DUCKDIR = "/mnt/f/programming/llm/test/qwen3.6-35b-a3b_freetoken/ctx262144/duck_hunt"
DEF_BUDGET = "21.15GiB"
DEF_RESERVE = "3GiB"
DEF_CTX = 262144
DEF_KV_DIR = "/home/thegiallo/.cache/freetoken/kv_smoke"

# ---- qwen3.6-35b-a3b geometry (same constants as launch_qwen35.sh) -------------------
TOTAL_EXPERTS = 10240
PER_SLOT_BYTES = 1376256        # gate_up Q4_K + down Q6_K per expert slot
KV_BYTES_PER_TOKEN = 20480      # hybrid GDN/mamba + attention state, measured
WEIGHTS_FIXED = 7.10 * (1 << 30)
KVRESERVE_TOKENS = 32768

DEFAULT_MAX_TOKENS = 512
TARGET_FRACTION = 0.75           # prompt fills up to this fraction of --ctx
RENDEZVOUS_PORT = 1920           # TCPStore listener; must be free before relaunching
TEARDOWN_TIMEOUT = 600           # s to wait for snapshot save + scheduler exit on SIGTERM


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def snapshot_metas(kv_dir: str) -> list[str]:
    return glob.glob(os.path.join(kv_dir, "qwen3.6-35b-a3b*", "meta.json"))


def to_bytes(size: str) -> int:
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmgt]i?b?|b)?$", size.strip(), re.I)
    if not m:
        raise SystemExit(f"bad size: {size!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "G").lower()
    mult = {
        "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
        "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40,
        "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12,
    }
    return int(value * mult[unit])


# ------------------------------------------------------------- http + server plumbing
def _in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_path(path: str) -> str:
    m = re.match(r"^([A-Za-z]):(.*)$", path.replace("\\", "/"))
    return f"/mnt/{m.group(1).lower()}{m.group(2)}" if m else path


def _reexec_in_wsl() -> None:
    """ft only reaches the ROCm device from inside WSL. If launched from a Windows
    interpreter, re-exec this script under the WSL login shell so the ft subprocess
    inherits the Linux driver environment (HOME, PATH, ROCm libs)."""
    if _in_wsl():
        return
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl_exe:
        root = os.environ.get("SystemRoot") or r"C:\Windows"
        for sub in ("System32", "Sysnative"):
            cand = os.path.join(root, sub, "wsl.exe")
            if os.path.exists(cand):
                wsl_exe = cand
                break
    if not wsl_exe:
        sys.exit("could not locate wsl.exe; run this script from a Windows console (Git Bash)")
    wsl_py = "/home/thegiallo/.freetoken/venv/bin/python"
    script = shlex.quote(_wsl_path(os.path.abspath(sys.argv[0])))
    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    cmd = [wsl_exe, "-e", "bash", "-lc", f"exec {wsl_py} -u {script} {args}"]
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        sys.exit(130)


def http_json(method: str, path: str, body=None, timeout: float = 30.0):
    url = f"http://127.0.0.1:{PATH_PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(f"HTTP {exc.code} from {path}: {raw[:400].decode(errors='replace')}")
    if not raw:
        return {}
    return json.loads(raw)


def wait_ready(timeout: float, poll: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            doc = http_json("GET", "/health", timeout=5)
            if doc.get("status") == "ok":
                for _ in range(60):
                    try:
                        stats = http_json("GET", "/v1/stats", timeout=5)
                        if stats.get("model"):
                            return stats
                    except Exception:
                        pass
                    time.sleep(poll)
            last = doc
        except Exception as exc:  # server not up yet
            last = exc
        time.sleep(poll)
    raise RuntimeError(f"server not ready after {timeout:.0f}s; last /health: {last!r}")


class Server:
    def __init__(self, args: list[str], log_path: str, timeout: float):
        self.log_path = log_path
        self.timeout = timeout
        self.proc = subprocess.Popen(
            args,
            stdout=open(log_path, "wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def tail(self, since: float) -> str:
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(since)
                return fh.read().decode(errors="replace")
        except OSError:
            return ""

    def stop(self, kv_dir=None):
        """SIGTERM the server, then (when kv_dir is given) wait until the whole process
        tree is gone and the snapshot finished writing. The scheduler forks a child that
        outlives the parent and holds the TCPStore on RENDEZVOUS_PORT while it saves."""
        if self.proc.poll() is not None:
            rc = self.proc.returncode
        else:
            os.kill(self.proc.pid, signal.SIGTERM)
            try:
                rc = self.proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise SystemExit("server did not stop after SIGTERM; killed it")
        if kv_dir is not None and rc is not None:
            self.wait_teardown(kv_dir)
        return rc

    def wait_teardown(self, kv_dir: str) -> list[str]:
        deadline = time.monotonic() + TEARDOWN_TIMEOUT
        while time.monotonic() < deadline:
            metas = snapshot_metas(kv_dir)
            if metas and not _port_in_use(RENDEZVOUS_PORT):
                return metas
            time.sleep(5)
        return snapshot_metas(kv_dir)


# ------------------------------------------------------------- duck-hunt context build
def build_prompt(duckdir: str, target_chars: int):
    names = ["README.md", "duck_hunt.py", "test_duck_hunt.py", "run_game.sh"]
    files = {}
    missing = []
    for name in names:
        path = os.path.join(duckdir, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            files[name] = fh.read()
    if missing:
        print(f"note: missing workspace files (only {sorted(files)} included): {missing}")

    turns = []
    acc = 0
    i = 0
    while acc < target_chars and files:
        i += 1
        blob = "\n\n".join(
            f"--- {name} (duck_hunt project) ---\n{text}" for name, text in files.items()
        )
        user = (
            f"We are building the terminal Duck Hunt game in {os.path.basename(duckdir)}. "
            f"Here is the current project state (iteration {i}):\n\n{blob}\n\n"
            "Improve the game: fix anything wrong, then add polish."
        )
        turns.append({"role": "user", "content": user})
        # Deterministic assistant filler so the context grows with realistic interleaving.
        filler = (
            "I reviewed the code and will improve it: keep the curses mouse handling, "
            "balance the round timers, and make the golden duck stand out more."
        )
        turns.append({"role": "assistant", "content": filler + f" (iteration {i} note)"})
        acc += len(user) + len(filler)
    turns.append(
        {
            "role": "user",
            "content": (
                "Final request: implement the bonus round and a laugh counter. Keep every "
                "existing feature working."
            ),
        }
    )
    return turns


# ------------------------------------------------------------- stats sampling in vivo
class Sampler:
    def __init__(self):
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(http_json("GET", "/v1/stats", timeout=3))
            except Exception:
                pass
            self._stop.wait(1.5)

    def __enter__(self):
        self.samples.clear()
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join(timeout=5)

    def max_tps(self, key: str) -> float:
        return max((s.get("throughput", {}).get(key, 0) or 0) for s in self.samples)

    def kv_used_pages(self) -> int:
        return max((s.get("kv") or {}).get("used_pages", 0) for s in self.samples)

    def vram_bytes(self) -> int:
        return max(s.get("vram_bytes", 0) or 0 for s in self.samples)


# ------------------------------------------------------------- one run against the API
def run_completion(payload: dict) -> tuple[float, dict, Sampler]:
    sampler = Sampler()
    t0 = time.monotonic()
    with sampler:
        resp = http_json("POST", "/v1/chat/completions", payload, timeout=1800)
    elapsed = time.monotonic() - t0
    usage = resp.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    hist = {
        "elapsed_s": round(elapsed, 2),
        "object": resp.get("object"),
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "usage": usage,
        "cached_tokens": cached,
    }
    stats = http_json("GET", "/v1/stats", timeout=5)
    req = stats.get("requests") or {}
    hist["ttft_mean_ms"] = req.get("ttft_mean_ms")
    hist["p95_ms"] = req.get("p95_ms")
    hist["prompt_tokens_total"] = req.get("prompt_tokens_total")
    hist["completion_tokens_total"] = req.get("completion_tokens_total")
    hist["max_prefill_tps"] = round(sampler.max_tps("prefill_tps"), 1)
    hist["max_decode_tps"] = round(sampler.max_tps("decode_tps"), 1)
    kv = stats.get("kv") or {}
    hist["kv_used_pages"] = kv.get("used_pages")
    hist["kv_total_pages"] = kv.get("total_pages")
    hist["vram_bytes"] = stats.get("vram_bytes")
    return elapsed, hist, stats


# ------------------------------------------------------------- main
def main() -> int:
    global PATH_PORT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--ft-bin", default=DEF_FT)
    ap.add_argument("--duckdir", default=DEF_DUCKDIR)
    ap.add_argument("--budget", default=DEF_BUDGET, help="expert+KV VRAM budget (default 21.15GiB)")
    ap.add_argument("--reserve", default=DEF_RESERVE, help="VRAM kept free after sizing (default 3GiB)")
    ap.add_argument("--ctx", type=int, default=DEF_CTX, help="--num-tokens (default 262144)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--kv-dir", default=DEF_KV_DIR, help="dedicated snapshot dir (persistent in $HOME/.cache)")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--keep", action="store_true", help="leave the final server running")
    ap.add_argument("--keep-snapshot", action="store_true", help="do not wipe the snapshot dir at start")
    args = ap.parse_args()
    PATH_PORT = args.port

    # ---- geometry: mirror launch_qwen35.sh, reserve free VRAM after sizing ----------
    budget = to_bytes(args.budget)
    reserve = to_bytes(args.reserve)
    avail = budget - WEIGHTS_FIXED - reserve
    kv_needed = args.ctx * KV_BYTES_PER_TOKEN
    exp_bytes = avail - kv_needed
    slots = max(0, min(TOTAL_EXPERTS, exp_bytes // PER_SLOT_BYTES))
    rate = slots / TOTAL_EXPERTS
    if slots <= 0:
        raise SystemExit(
            f"geometry leaves no expert cache: budget={args.budget} reserve={args.reserve} "
            f"ctx={args.ctx}; raise --budget or lower --ctx/--reserve"
        )
    print("---- geometry ----")
    print(f"  budget {args.budget} - fixed {WEIGHTS_FIXED / 2**30:.2f}GiB - reserve {args.reserve}")
    print(f"  -> {avail / 2**30:.2f}GiB for expert cache + KV; ctx {args.ctx} tok needs "
          f"{kv_needed / 2**30:.2f}GiB")
    print(f"  moe-cache-rate {rate:.3f} ({slots}/10,240 expert slots), {reserve / 2**30:.2f}GiB kept free")

    # ---- deterministic duck-hunt context -------------------------------------------
    target_chars = int(args.ctx * TARGET_FRACTION * 4)  # ~0.25 tok/char heuristic
    turns = build_prompt(args.duckdir, target_chars)
    prompt_chars = sum(len(t["content"]) for t in turns)
    est_tokens = int(prompt_chars * 0.25)
    print(f"  context prompt: {len(turns)} turns, {prompt_chars:,} chars (~{est_tokens:,} tokens)")

    payload = {
        "model": None,  # filled from /v1/stats once ready
        "messages": turns,
        "max_tokens": args.max_tokens,
        "temperature": 0.3,
        "reasoning_effort": "medium",
    }

    # ---- refuse if a server is already listening (the pi server runs on this port) ---
    try:
        http_json("GET", "/health", timeout=2)
        raise SystemExit(
            f"a server is already answering on :{args.port}; stop it (or use --port) so the "
            "test owns a fresh instance with its own snapshot dir"
        )
    except (RuntimeError, urllib.error.URLError, OSError):
        pass
    if _port_in_use(RENDEZVOUS_PORT):
        raise SystemExit(
            f"TCPStore port {RENDEZVOUS_PORT} is still in use (stray ft from a previous run); "
            "stop it before rerunning the smoke test"
        )

    # ---- snapshot dir: dedicated, wiped so the test is reproducible -----------------
    if args.kv_dir == DEF_KV_DIR and not args.keep_snapshot:
        shutil.rmtree(args.kv_dir, ignore_errors=True)
    os.makedirs(args.kv_dir, exist_ok=True)

    serve = [
        args.ft_bin, "serve",
        "--model", args.model,
        "--moe-backend", "offload",
        "--kv-reserve-tokens", str(KVRESERVE_TOKENS),
        "--moe-cache-rate", f"{rate:.4f}",
        "--num-tokens", str(args.ctx),
        "--tool-call-parser", "qwen35",
        "--enable-cache-report",
        "--kv-persist",
        "--kv-persist-dir", args.kv_dir,
        "--kv-persist-max-gb", str(max(8, (kv_needed * 3) // (1 << 30))),
        "--port", str(args.port),
    ]
    print("  serve:", " ".join(serve))
    print()

    log = "/home/thegiallo/kv_persist_smoke_ft.log"
    results = {}

    for run in (1, 2):
        pos = os.path.getsize(log) if os.path.exists(log) else 0
        server = Server(serve, log, timeout=600)
        print(f"=== run {run}: launching ft (model load can take a few minutes) ===")
        try:
            stats = wait_ready(1200)
            payload["model"] = (stats.get("model") or {}).get("id")
        except Exception:
            print(server.tail(pos))
            print(f"FAIL: server did not become ready on run {run}")
            return 1
        period = server.tail(pos)
        print(f"    ready; served model id = {payload['model']}")
        for h in [l for l in period.splitlines() if "kv-persist:" in l]:
            print(f"    log: {h.strip()}")

        print(f"    sending {est_tokens:,}-token completion "
              f"(run {run}: {'cold prefill' if run == 1 else 'restored prefix'}...)")
        elapsed, hist, _stats = run_completion(payload)
        results[run] = hist
        print(f"    response: {hist['object']} finish={hist['finish_reason']} "
              f"in {hist['elapsed_s']}s")
        print(f"      usage.prompt_tokens={hist['usage'].get('prompt_tokens')} "
              f"completion={hist['usage'].get('completion_tokens')} "
              f"cached_tokens={hist['cached_tokens']}")
        print(f"      /v1/stats: ttft={hist['ttft_mean_ms']}ms p95={hist['p95_ms']}ms "
              f"max_prefill={hist['max_prefill_tps']}t/s max_decode={hist['max_decode_tps']}t/s "
              f"kv={hist['kv_used_pages']}/{hist['kv_total_pages']}pages "
              f"vram={hist['vram_bytes'] / 2**30:.2f}GiB")

        if run == 2 and args.keep:
            print("    --keep: leaving the server running (no shutdown)")
            break
        print("    stopping server (SIGTERM -> graceful shutdown, snapshot save)...")
        rc = server.stop(kv_dir=args.kv_dir)
        print(f"    exited rc={rc} (waited for scheduler teardown + snapshot)")
        tail = server.tail(pos)
        saved = [l for l in tail.splitlines() if "kv-persist: saved" in l]
        for h in saved[-3:]:
            print(f"    log: {h.strip()}")
        if run == 1:
            hits = snapshot_metas(args.kv_dir)
            if not hits:
                print("    ERROR: no snapshot meta.json after teardown wait")
                results["run1_saved"] = False
            else:
                size = sum(
                    os.path.getsize(os.path.join(d, p))
                    for d in {os.path.dirname(h) for h in hits}
                    for p in os.listdir(d)
                    if os.path.isfile(os.path.join(d, p))
                )
                print(f"    snapshot dir: {os.path.dirname(hits[0])} ({size / 2**30:.2f}GiB)")
                results["run1_saved"] = True

    # ---- verdict --------------------------------------------------------------------
    r1, r2 = results.get(1), results.get(2)
    if r1 is None or r2 is None:
        print("FAIL: one of the runs did not complete; see the log above")
        return 1
    ok = True
    checks = []
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        checks.append(bool(cond))

    p1 = r1["usage"].get("prompt_tokens") or 0
    p2 = r2["usage"].get("prompt_tokens") or 0
    cached2 = r2["cached_tokens"]
    check("snapshot saved at shutdown", results.get("run1_saved", False))
    check("served request with equal prompt tokens", p1 == p2, f"{p1} == {p2}")
    frac = cached2 / p2 if p2 else 0
    check(
        "prefix materialized (cached_tokens/prompt_tokens)",
        p2 > 0 and frac >= 0.9,
        f"{cached2}/{p2} = {frac:.3f} (>= 0.90)",
    )
    t1 = r1["ttft_mean_ms"] or 0
    t2 = r2["ttft_mean_ms"] or 0
    if t1 and t2:
        check("ttft after restart is faster", t2 < t1 / 2, f"{t2}ms vs {t1}ms")
    else:
        check("ttft after restart is faster (elapsed)", r2["elapsed_s"] < r1["elapsed_s"] / 2,
              f"{r2['elapsed_s']}s vs {r1['elapsed_s']}s")
    r1p, r2p = r1["max_prefill_tps"], r2["max_prefill_tps"]
    r1d, r2d = r1["max_decode_tps"], r2["max_decode_tps"]
    if r2d != 0:
        d_ok = r2d >= r1d * 0.8
        check("decode rate held across restore", d_ok, f"{r2d}t/s vs {r1d}t/s")
    print(f"    (info) max prefill tps is per-window noise, not comparable "
          f"(r1={r1p}, r2={r2p}); elapsed x{r1['elapsed_s']/max(r2['elapsed_s'],1):.1f} split "
          f"(r1={r1['elapsed_s']}s, r2={r2['elapsed_s']}s)")

    print()
    print("summary:")
    for run, h in results.items():
        if isinstance(run, int):
            print(f"  run {run}: elapsed={h['elapsed_s']}s ttft={h['ttft_mean_ms']}ms "
                  f"prefill={h['max_prefill_tps']}t/s cached={h['cached_tokens']} "
                  f"prompt={h['usage'].get('prompt_tokens')}")
    print("PASS: kv-persist restored the session prefix after restart"
          if ok else "FAIL: see the failing checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    _reexec_in_wsl()
    raise SystemExit(main())