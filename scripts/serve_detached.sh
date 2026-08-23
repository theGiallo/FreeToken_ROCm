#!/usr/bin/env bash
# Launch the freetoken server detached from the calling console.
# Usage: serve_detached.sh <model_path> [extra ft serve args...]
set -u
MODEL="$1"
shift || true
LOG=/mnt/f/programming/llm/FreeToken/ft_serve.log
: > "$LOG"
exec ~/venv-ft/bin/ft serve --model-path "$MODEL" "$@" >>"$LOG" 2>&1
