#!/usr/bin/env python3
"""Context-length throughput sweep for qwen3.6-35b-a3b on the ROCm box.

Rebuilds our old /tmp ctx sweeps (up8k_8kto98k, ctx64to262k, ...) around a REAL pi
session: the harness reads the pi session JSONL, maps user/assistant/toolResult
messages onto OpenAI roles, truncates the OLDEST turns so the surviving context fills a
target fraction of each requested ctx, and ends with the actual last user request. Each
step launches a fresh `ft serve` sized with the launch_qwen35.sh geometry (budget /
reserve -> --moe-cache-rate), sends the completion, samples /v1/stats while it runs, and
reports decode + input throughput plus VRAM/KV/cache-rate per ctx.

Throughput numbers mirror llama.cpp's own "prompt eval speed" / "eval speed":
  input t/s   = usage.prompt_tokens / (t_first_sse - t_request_start)
  decode t/s  = (usage.completion_tokens - 1) / (t_last_sse - t_first_sse)
plus the sampled mean/median/trimmed decode_tps and max prefill_tps from /v1/stats
(what our old sweeps reported), so we can compare apples-to-apples with the llama.cpp
result table we keep in report_test_qwen3.8-27B.md.

Run inside WSL against the ROCm box:
  cd /mnt/f/programming/llm/FreeToken
  $HOME/.freetoken/venv/bin/python scripts/bench_session_sweep.py --ctx 98304 [--port 1922]

Output: RESULT<k> lines per ctx on stdout (stable for the shell-wrap that used to
produce /tmp/ctx_*.log) and a CSV row appended to results/bench_sweep.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.name == "nt":
    sys.exit("run this with the WSL interpreter (the ft subprocess needs the ROCm env)")

MIN_PY = (3, 10)
if sys.version_info < MIN_PY:
    sys.exit(f"need Python {'.'.join(map(str, MIN_PY))}+")

# ---- defaults matching launch_qwen35.sh / kv_persist_smoke.py --------------------------
DEF_MODEL = "/home/thegiallo/models/qwen3.6-35b-a3b.gguf"
DEF_FT = "/home/thegiallo/.freetoken/venv/bin/ft"
DEF_SESSION = "/home/thegiallo/.pi/agent/sessions/--mnt-f-programming-llm-test-qwen3.6-35b-a3b_freetoken-ctx262144--/2026-09-03T22-38-11-987Z_01a0696c-2c12-73ac-9864-936192f7bfe1.jsonl"
DEF_BUDGET = "21.15GiB"
DEF_RESERVE = "1.5GiB"
DEF_CTX = 131072
DEF_MAX_TOKENS = 512
TARGET_FRACTION = 0.75
PORT = 1922
RENDEZVOUS = 1923  # TCPStore must not collide
TEARDOWN = 600

# ---- qwen3.6-35b-a3b geometry (same constants as launch_qwen35.sh) ---------------------
TOTAL_EXPERTS = 10240
PER_SLOT_BYTES = 1376256        # gate_up Q4_K + down Q6_K per expert slot
KV_BYTES_PER_TOKEN = 20480      # measured 20 KiB/token
WEIGHTS_FIXED = 7.10 * (1 << 30)
KVRESERVE_TOKENS = 32768

# The session's actual final user request; the harness always appends it, whatever ctx cap.
LAST_USER_REQUEST = None


def to_bytes(size: str) -> int:
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmgt]i?b?|b)?$", size.strip(), re.I)
    if not m:
        raise SystemExit(f"bad size: {size!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "G").lower()
    mult = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
            "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40,
            "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
    return int(value * mult[unit])


def geometry(ctx: int, budget_bytes: int, reserve_bytes: int) -> dict:
    """Mirror launch_qwen35.sh: budget - weights - reserve goes to expert cache + KV."""
    avail = budget_bytes - WEIGHTS_FIXED - reserve_bytes
    kv_needed = ctx * KV_BYTES_PER_TOKEN
    exp_bytes = avail - kv_needed
    slots = max(0, min(TOTAL_EXPERTS, exp_bytes // PER_SLOT_BYTES))
    rate = slots / TOTAL_EXPERTS
    return {
        "ctx": ctx,
        "budget_gi": budget_bytes / 2 ** 30,
        "reserve_gi": reserve_bytes / 2 ** 30,
        "kv_gi": kv_needed / 2 ** 30,
        "avail_gi": avail / 2 ** 30,
        "exp_gi": exp_bytes / 2 ** 30,
        "slots": slots,
        "rate": round(rate, 4),
    }


# ------------------------------------------------------------- session replay ----------
def _flatten(part: dict) -> str:
    if part.get("type") == "text":
        return part.get("text") or ""
    if part.get("type") in ("thinking", "reasoning"):
        return ""
    return ""


def load_session(path: str) -> tuple[list[dict], str]:
    """Parse the pi JSONL into OpenAI-style messages + the final user request.

    user/assistant/toolResult -> user/assistant/tool. Assistant messages carry only their
    visible text (thinking parts are dropped: they were server-side reasoning, not part of
    the context the model re-reads). toolResult keeps tool_call_id + name so the qwen35
    chat template can render it as a tool slot."""
    if path.startswith("~"):
        path = os.path.expanduser(path)
    msgs = []
    last_user = ""
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        parts = m.get("content") if isinstance(m.get("content"), list) else None
        if role == "user":
            text = "".join(_flatten(p) for p in parts) if parts else (m.get("content") or "")
            msgs.append({"role": "user", "content": text})
            last_user = text
        elif role == "assistant":
            text = "".join(_flatten(p) for p in parts) if parts else (m.get("content") or "")
            if not text.strip():
                continue  # pi sometimes records an empty assistant turn after a tool error
            msgs.append({"role": "assistant", "content": text})
        elif role == "toolResult":
            text = "".join(_flatten(p) for p in parts) if parts else (m.get("content") or "")
            msgs.append({
                "role": "tool",
                "content": text,
                "tool_call_id": m.get("toolCallId") or "call_x",
                "name": m.get("toolName") or "bash",
            })
    return msgs, last_user


class PromptCounter:
    """Exact prompt tokenizer, mirroring the engine's frontend tokenizer path."""
    _loaded = None

    @classmethod
    def _manager(cls, model_path: str):
        if cls._loaded is None or cls._loaded[0] != model_path:
            from freetoken.tokenizer.tokenize import TokenizeManager
            from freetoken.utils.hf import load_tokenizer

            cls._loaded = (model_path, TokenizeManager(load_tokenizer(model_path)))
        return cls._loaded[1]

    @staticmethod
    def _msg(messages: list[dict]):
        from freetoken.core import SamplingParams
        from freetoken.message import TokenizeMsg

        return TokenizeMsg(
            uid=0,
            text=messages,
            sampling_params=SamplingParams(),
            chat_template_kwargs=None,
            tools=None,
        )

    def count(self, model_path: str, messages: list[dict]) -> int:
        return int(self._manager(model_path).tokenize([self._msg(messages)])[0].numel())


def fit_context(
    model_path: str,
    msgs: list[dict],
    keep_last_user: str,
    budget_tokens: int,
) -> list[dict]:
    """Produce the final message list: the whole pi session if it fits budget_tokens,
    padded with deterministic filler turns BEFORE the session if it is short,
    or the oldest turns cut (exact tokenizer counts, binary search) if it is long.
    The final user request always stays the last message."""
    counter = PromptCounter()
    n_full = counter.count(model_path, msgs)
    if n_full <= budget_tokens:
        pad_budget = budget_tokens - n_full
        filler, _ = _filler_turns(counter, model_path, max(0, pad_budget - 512))
        return filler + msgs
    lo, hi = 0, len(msgs)
    while lo < hi:
        mid = (lo + hi) // 2
        if counter.count(model_path, msgs[mid:]) <= budget_tokens:
            hi = mid
        else:
            lo = mid + 1
    return msgs[lo:]


def _filler_turns(counter, model_path: str, target_tokens: int) -> tuple[list[dict], int]:
    """Deterministic filler turns sized by the exact tokenizer (binary search
    over turn count). Returns (turns, exact_token_count_of_turns)."""
    pair = [
        {"role": "user", "content": "We are building the terminal game; help me balance it."},
        {"role": "assistant",
         "content": "I reviewed the code and will improve it: keep the input handling, "
                    "balance the round timers, and make the score display stand out more."},
    ]
    if target_tokens <= 0:
        return [], 0
    lo, hi = 0, max(4, (target_tokens + 24) // 25)  # single pair ~ tens of tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        turns = []
        for i in range(1, mid + 1):
            turns.append({
                "role": "user",
                "content": pair[0]["content"] + f" (session iteration {i})",
            })
            turns.append({
                "role": "assistant",
                "content": pair[1]["content"] + f" (iteration {i} note)",
            })
        if counter.count(model_path, turns) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    turns = []
    for i in range(1, lo + 1):
        turns.append({
            "role": "user",
            "content": pair[0]["content"] + f" (session iteration {i})",
        })
        turns.append({
            "role": "assistant",
            "content": pair[1]["content"] + f" (iteration {i} note)",
        })
    return turns, counter.count(model_path, turns)


def exact_prompt_tokens(model_path: str, messages: list[dict]) -> int:
    return PromptCounter().count(model_path, messages)


# ------------------------------------------------------------- http + server -----------
def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_json(method: str, path: str, body=None, timeout: float = 30.0):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {path}: {exc.read()[:400]!r}")
    if not raw:
        return {}
    return json.loads(raw)


def wait_ready(timeout: float) -> dict:
    deadline = time.monotonic() + timeout
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
                    time.sleep(2)
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready after {timeout:.0f}s")


class Server:
    def __init__(self, args: list[str], log: str, timeout: float):
        self.log = log
        self.timeout = timeout
        self.proc = subprocess.Popen(args, stdout=open(log, "wb"), stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def tail(self, since: float) -> str:
        try:
            with open(self.log, "rb") as fh:
                fh.seek(since)
                return fh.read().decode(errors="replace")
        except OSError:
            return ""

    def stop(self):
        if self.proc.poll() is not None:
            return self.proc.returncode
        os.kill(self.proc.pid, signal.SIGTERM)
        try:
            return self.proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise SystemExit("server did not stop after SIGTERM; killed it")


class Sampler:
    """Samples /v1/stats every 1.5s; recorder threads may overlap a run for its window."""

    def __init__(self):
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.samples.clear()
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(http_json("GET", "/v1/stats", timeout=3))
            except Exception:
                pass
            self._stop.wait(1.5)

    def max_dt(self) -> float:
        return max((s.get("throughput", {}).get("decode_tps", 0) or 0) for s in self.samples)

    def max_pt(self) -> float:
        return max((s.get("throughput", {}).get("prefill_tps", 0) or 0) for s in self.samples)

    def mean_dt(self) -> float:
        v = [s.get("throughput", {}).get("decode_tps", 0) or 0 for s in self.samples if s.get("throughput", {}).get("decode_tps")]
        return sum(v) / len(v) if v else 0.0

    def med_dt(self) -> float:
        v = sorted(s.get("throughput", {}).get("decode_tps", 0) or 0
                   for s in self.samples if s.get("throughput", {}).get("decode_tps"))
        if not v:
            return 0.0
        return v[len(v) // 2]

    def trimmed_mean_dt(self) -> float:
        v = sorted(s.get("throughput", {}).get("decode_tps", 0) or 0
                   for s in self.samples if s.get("throughput", {}).get("decode_tps"))
        if len(v) < 6:
            return sum(v) / len(v) if v else 0.0
        trim = max(1, len(v) // 10)
        inner = v[trim:-trim] or v
        return sum(inner) / len(inner)

    def kv_pages(self) -> int:
        return max((s.get("kv") or {}).get("used_pages", 0) for s in self.samples)

    def vram(self) -> int:
        return max(s.get("vram_bytes", 0) or 0 for s in self.samples)


def sse_times(payload: dict) -> tuple[float | None, float | None, dict, str | None, float]:
    """POST the completion; return (t_first_tok, t_last_tok, usage, finish_reason, t_send).

    Times anchor on SSE events that actually carry generated content (text/tool-call/reason
    deltas), never the leading role announcement. t_send is the moment the request socket
    opened, so input_tps = prompt_tokens / (t_first_tok - t_send) measures the full prefill
    the same way llama.cpp's "prompt eval" timings do."""
    t_send = time.monotonic()
    first = last = None
    usage = {}
    finish_reason = None
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            now = time.monotonic()
            delta = (ev.get("choices") or [{}])[0].get("delta") or {}
            has_tok = bool(delta.get("content") or delta.get("tool_calls")
                           or delta.get("reasoning_content"))
            if has_tok:
                if first is None:
                    first = now
                last = now
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
    return first, last, usage, finish_reason, t_send


# ------------------------------------------------------------- main --------------------
CSV_FIELDS = [
    "ctx", "rate", "slots", "kv_gi", "exp_gi", "label", "verdict", "wall_s", "req_s",
    "prompt_tok", "comp_tok", "input_tps", "decode_tps", "max_decode", "max_prefill",
    "mean_decode", "med_decode", "trimmed_mean", "ttft_ms", "n_deg", "kv_pages", "vram_gi",
]


def main() -> int:
    global PORT, RENDEZVOUS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--ft-bin", default=DEF_FT)
    ap.add_argument("--session", default=DEF_SESSION)
    ap.add_argument("--budget", default=DEF_BUDGET)
    ap.add_argument("--reserve", default=DEF_RESERVE)
    ap.add_argument("--ctx", type=int, default=DEF_CTX)
    ap.add_argument("--max-tokens", type=int, default=DEF_MAX_TOKENS)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--label", default="base", help="tag for the CSV/log; e.g. base, nooverlap")
    args = ap.parse_args()
    PORT = args.port
    RENDEZVOUS = args.port + 1

    assert not _port_in_use(PORT), f"port {PORT} busy; stop it or pass --port"
    assert not _port_in_use(RENDEZVOUS), f"TCPStore port {RENDEZVOUS} busy"

    msgs, last_user = load_session(args.session)
    global LAST_USER_REQUEST
    LAST_USER_REQUEST = last_user
    print(f"[session] {len(msgs)} messages, {sum(len(m.get('content') or '') for m in msgs):,} chars")
    print(f"[last-user] {last_user!r}")

    g = geometry(args.ctx, to_bytes(args.budget), to_bytes(args.reserve))
    if g["slots"] <= 0:
        raise SystemExit(f"geometry leaves no expert cache at ctx={args.ctx} budget={args.budget}")
    print("---- geometry ----")
    print(f"  budget {args.budget} - weights {WEIGHTS_FIXED/2**30:.2f}GiB - reserve {args.reserve}")
    print(f"  -> avail {g['avail_gi']:.2f}GiB; ctx {args.ctx} KV = {g['kv_gi']:.2f}GiB")
    print(f"  moe-cache-rate {g['rate']:.3f} ({g['slots']}/10240 slots)")

    budget_tokens = int(args.ctx * TARGET_FRACTION) - args.max_tokens  # headroom for output
    messages = fit_context(args.model, msgs, last_user, budget_tokens)
    exact = exact_prompt_tokens(args.model, messages)
    print(f"  context: {len(messages)} messages, exact {exact:,} tokens "
          f"(budget {budget_tokens:,})")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.abspath(os.path.join(log_dir, f"bench_ft_{args.ctx}.log"))
    serve = [
        args.ft_bin, "serve",
        "--model", args.model,
        "--moe-backend", "offload",
        "--kv-reserve-tokens", str(KVRESERVE_TOKENS),
        "--moe-cache-rate", f"{g['rate']:.4f}",
        "--num-tokens", str(args.ctx),
        "--tool-call-parser", "qwen35",
        "--enable-cache-report",
        "--port", str(PORT),
        "--decode-log-interval", "8",
    ]
    print("  serve:", " ".join(serve), "\n")

    pos = os.path.getsize(log) if os.path.exists(log) else 0
    server = Server(serve, log, timeout=600)
    try:
        stats = wait_ready(1200)
    except Exception:
        print(server.tail(pos))
        raise
    model_id = (stats.get("model") or {}).get("id")
    start_period = server.tail(pos)
    for h in [l for l in start_period.splitlines() if "ccache-rate" in l or "moe-cache" in l or "Allocating" in l]:
        print(f"    log: {h.strip()[:200]}")

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.monotonic()
    with Sampler() as sampler:
        try:
            first, last, usage, finish_reason, t_send = sse_times(payload)
        except Exception:
            print(server.tail(pos))
            raise
    wall = time.monotonic() - t0

    prompt_tok = usage.get("prompt_tokens", 0)
    comp_tok = usage.get("completion_tokens", 0)
    ttft_ms = (first - t_send) * 1e3 if first is not None else 0.0
    input_tps = prompt_tok / (first - t_send) if (first is not None and first > t_send) else 0.0
    decode_tps = (comp_tok - 1) / (last - first) if (first is not None and last is not None and last > first) else 0.0
    n_deg = comp_tok

    res = {
        "ctx": args.ctx, "rate": g["rate"], "slots": g["slots"],
        "kv_gi": round(g["kv_gi"], 2), "exp_gi": round(g["exp_gi"], 2),
        "verdict": "PASS" if finish_reason or comp_tok else "HANG",
        "wall_s": round(wall, 1), "req_s": round(wall, 1),
        "prompt_tok": prompt_tok, "comp_tok": comp_tok,
        "input_tps": round(input_tps, 1), "decode_tps": round(decode_tps, 2),
        "max_decode": round(sampler.max_dt(), 2), "max_prefill": round(sampler.max_pt(), 1),
        "mean_decode": round(sampler.mean_dt(), 2), "med_decode": round(sampler.med_dt(), 2),
        "trimmed_mean": round(sampler.trimmed_mean_dt(), 2),
        "ttft_ms": round(ttft_ms, 1), "n_deg": n_deg,
        "kv_pages": sampler.kv_pages(),
        "vram_gi": round(sampler.vram() / 2 ** 30, 2),
    }
    print(f"[done] done={res['verdict']} wall={wall:.1f}s prompt_tok={prompt_tok} comp_tok={comp_tok}")
    print(f"[stats] max_decode={res['max_decode']} mean_decode={res['mean_decode']} "
          f"med_decode={res['med_decode']} trimmed_mean={res['trimmed_mean']} n_deg={n_deg} "
          f"max_prefill={res['max_prefill']}")
    print(f"[rates] input_tps={res['input_tps']} decode_tps={res['decode_tps']} ttft_ms={res['ttft_ms']}")
    print(f"RESULT {res['ctx']},{res['verdict']},{res['wall_s']},{res['max_decode']},"
          f"{res['trimmed_mean']},{n_deg},{res['max_prefill']}")

    out = Path(__file__).resolve().parent.parent / "results" / "bench_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        res["label"] = args.label
        w.writerow({k: res.get(k, "") for k in CSV_FIELDS})

    if not args.keep:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())