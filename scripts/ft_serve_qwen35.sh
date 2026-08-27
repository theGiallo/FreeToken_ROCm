#!/usr/bin/env bash
# Launch the FreeToken server with the Qwen3.6-35B-A3B ROCm tuning.
# Usage: bash scripts/ft_serve_qwen35.sh
set -euo pipefail

MODEL="${FT_MODEL:-$HOME/models/qwen3.6-35b-a3b.gguf}"
FT_BIN="/home/thegiallo/.freetoken/venv/bin/ft"
LOG="$HOME/ft_server.log"

if curl -s --max-time 2 http://127.0.0.1:1919/v1/stats > /dev/null 2>&1; then
    echo "Server already running on :1919"
    curl -s http://127.0.0.1:1919/v1/stats | head -c 300
    echo
    exit 0
fi

echo "Starting FreeToken server (model=$MODEL)"
echo "Log: $LOG"
setsid nohup "$FT_BIN" serve \
    --model "$MODEL" \
    --moe-backend offload \
    --moe-cache-auto \
    --kv-reserve-tokens 32768 \
    --num-tokens 131072 \
    --tool-call-parser qwen35 \
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