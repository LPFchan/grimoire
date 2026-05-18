#!/usr/bin/env bash
# Start an isolated bench llama-server for DFlash testing.
# Uses the grimoire:local image but runs llama-server directly
# (bypasses the Python gateway).
#
# Usage:
#   bash scripts/bench.sh start              # default bench on port 8082
#   bash scripts/bench.sh start --port 8085  # custom port
#   bash scripts/bench.sh stop               # stop bench container
#   bash scripts/bench.sh restart            # restart with same args
#
# The bench container is named "grimoire-bench" and uses a separate GPU slot.

set -euo pipefail

NAME="grimoire-bench"
PORT=8082
IMAGE="grimoire:local"
MODEL="/models/gguf/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="/models/gguf/dflash-draft-3.6-q8_0.gguf"

# Parse --port from args if present
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

stop() {
    echo "Stopping $NAME..."
    rtk docker rm -f "$NAME" 2>/dev/null || true
}

start() {
    echo "Starting $NAME on port $PORT..."
    rtk docker rm -f "$NAME" 2>/dev/null || true
    rtk docker run -d --name "$NAME" \
        --gpus '"device=0"' \
        --network host \
        -v /home/yeowool/models:/models:ro \
        -e GGML_DFLASH_MAX_VERIFY_TOKENS=52 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v /dev/shm:/dev/shm \
        --entrypoint /opt/grimoire-llama-cpp/bin/llama-server \
        "$IMAGE" \
        --model "$MODEL" \
        --spec-draft-model "$DRAFT" \
        --spec-type dflash \
        --spec-dflash-cross-ctx 1024 \
        --spec-dflash-max-slots 4 \
        --spec-draft-n-max 16 \
        --spec-branch-budget 0 \
        --spec-draft-temp 0 --temp 0 \
        --ctx-size 32000 --parallel 1 \
        --no-spec-dm-adaptive \
        --flash-attn on -b 2048 -ub 256 --jinja \
        --host 127.0.0.1 --port "$PORT" \
        --n-gpu-layers 99 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        "$@"

    echo "Waiting for model load..."
    for i in $(seq 1 60); do
        if rtk curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "$NAME ready on port $PORT"
            exit 0
        fi
        sleep 2
    done
    echo "Timed out waiting for $NAME"
    exit 1
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    *)
        echo "Usage: $0 [start|stop|restart] [--port N]"
        exit 1
        ;;
esac
