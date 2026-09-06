#!/usr/bin/env bash
# Launch the FreeToken server with the Qwen3.6-35B-A3B ROCm tuning.
# Usage: bash scripts/ft_serve_qwen35.sh
set -euo pipefail

MODEL="${FT_MODEL:-$HOME/models/qwen3.6-35b-a3b.gguf}"
FT_BIN="/home/thegiallo/.freetoken/venv/bin/ft"
LOG="$HOME/ft_server.log"

# KV-persist: 1 = save/restore the prefix cache across restarts, 0 = off.
KV_PERSIST="${FT_KV_PERSIST:-0}"
# Snapshot directory ($XDG_CACHE_HOME/ft/KV_cache if unset).
KV_PERSIST_DIR="${FT_KV_PERSIST_DIR:-}"
# Skip writing a snapshot larger than this many GiB (unset = unlimited).
KV_PERSIST_MAX_GB="${FT_KV_PERSIST_MAX_GB:-}"
# Load only snapshots younger than this many hours (unset = keep forever).
KV_PERSIST_MAX_AGE_H="${FT_KV_PERSIST_MAX_AGE_H:-}"

if curl -s --max-time 2 http://127.0.0.1:1919/v1/stats > /dev/null 2>&1; then
    echo "Server already running on :1919"
    curl -s http://127.0.0.1:1919/v1/stats | head -c 300
    echo
    exit 0
fi

echo "Starting FreeToken server (model=$MODEL)"
echo "Log: $LOG"
_ARGS=(
    --model "$MODEL"
    --moe-backend offload
    --moe-cache-auto
    --kv-reserve-tokens 32768
    --num-tokens 131072
    --tool-call-parser qwen35
)
if [ "$KV_PERSIST" = "1" ]; then
    _ARGS+=(--kv-persist)
    [ -n "$KV_PERSIST_DIR" ] && _ARGS+=(--kv-persist-dir "$KV_PERSIST_DIR")
    [ -n "$KV_PERSIST_MAX_GB" ] && _ARGS+=(--kv-persist-max-gb "$KV_PERSIST_MAX_GB")
    [ -n "$KV_PERSIST_MAX_AGE_H" ] && _ARGS+=(--kv-persist-max-age-h "$KV_PERSIST_MAX_AGE_H")
fi
setsid nohup "$FT_BIN" serve "${_ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
disown
echo "Server PID: $!"
echo "Waiting for /v1/stats... (model load takes ~2-3 min)"
for i in $(seq 1 60); do
    if curl -s --max-time 2 http://127.0.0.1:1919/v1/stats > /dev/null 2>&1; then
        echo "Ready after ~${i}x5s."
        curl -s http://127.0.0.1:1919/v1/stats | head -c 300
        echo
        exit 0
    fi
    sleep 5
done
echo "Server did not become ready. Check $LOG"
tail -20 "$LOG"
exit 1