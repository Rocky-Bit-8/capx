#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
MODEL="$ROOT/capx/molmo/allenai--Molmo2-8B/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
VENV="$ROOT/molmo-venv"
LOG="$ROOT/molmo-vllm.log"
PIDFILE="$ROOT/molmo-vllm.pid"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Molmo model is missing: $MODEL" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Molmo venv is missing: $VENV" >&2
  exit 2
fi
if [[ ! -e /dev/nvidia0 ]]; then
  echo "No NVIDIA device is visible. Start this script in a GPU instance/container." >&2
  exit 3
fi
if [[ -f "$PIDFILE" ]] && kill -0 "$(<"$PIDFILE")" 2>/dev/null; then
  echo "Molmo vLLM already running (PID $(<"$PIDFILE"))"
  exit 0
fi

nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name allenai/Molmo2-8B \
  --host 0.0.0.0 --port 8122 \
  --dtype bfloat16 --gpu-memory-utilization 0.45 \
  --max-model-len 4096 --trust-remote-code \
  >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "Started Molmo vLLM (PID $!), log: $LOG"
