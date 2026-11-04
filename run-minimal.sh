#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export MODEL_PATH="${MODEL_PATH:-$HOME/.cache/huggingface/hub/models--jdopensource--JoyAI-VL-Interaction-INT4/snapshots/1bfadbd2b41c11864cd7c359c7ea961aa6fd835c}"
export SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
export PYTHON_BIN="$PWD/services/.venv/bin/python"
export VENV_ACTIVATE="$PWD/services/.venv/bin/activate"
export MAIN_MODEL="jdopensource/JoyAI-VL-Interaction-INT4"
export MAIN_API_BASE="http://127.0.0.1:7060/v1"
export SUMMARIZER_MODEL="$SUMMARY_MODEL_PATH"
export SUMMARIZER_API_BASE="http://127.0.0.1:8065/v1"
export CHUNK=8 ASYNC_SUMMARY_LEAD_FRAMES=2 COMPRESS_EVERY_N_CHUNKS=3
export MAIN_MAX_TOKENS=128 SUMMARIZER_MAX_PIXELS=65536 SUMMARIZER_KEY_FRAMES=4
export MID_TERM_MAX_TOKENS=512 MID_TERM_TARGET_TOKEN_COUNT=384
export LONG_TERM_MAX_TOKENS=512 LONG_TERM_TARGET_TOKEN_COUNT=384
export LONG_TERM_MEMORY_WINDOW=2

LOG_DIR="$PWD/services/webinfer/logs/minimal"
READY_TIMEOUT="${SERVICE_READY_TIMEOUT:-900}"
PIDS=()
LAST_PID=""

for command in setsid curl; do
    command -v "$command" >/dev/null || {
        echo "Required command not found: $command" >&2
        exit 1
    }
done
for path in "$PYTHON_BIN" "$MODEL_PATH/config.json" "$SUMMARY_MODEL_PATH/config.json"; do
    [[ -f "$path" ]] || {
        echo "Required file not found: $path" >&2
        exit 1
    }
done

"$PYTHON_BIN" - <<'PY'
import socket
import vllm
import transformers

for port in (7060, 8065, 8070, 8099):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
PY

mkdir -p "$LOG_DIR"

cleanup() {
    local status=$?
    local pid
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        # Each service owns a process group, including its workers.
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_service() {
    local name="$1"
    shift
    setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
    LAST_PID=$!
    PIDS+=("$LAST_PID")
    echo "Starting $name; log: $LOG_DIR/$name.log"
}

wait_ready() {
    local name="$1" url="$2" log="$3"
    local deadline=$((SECONDS + READY_TIMEOUT))
    local pid
    while (( SECONDS < deadline )); do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "A service exited while waiting for $name. Logs: $LOG_DIR" >&2
                tail -n 40 "$log" >&2
                return 1
            fi
        done
        if curl -kfsS --max-time 2 "$url" >/dev/null 2>&1; then
            echo "$name ready."
            return 0
        fi
        sleep 2
    done
    echo "Timed out waiting for $name. Log: $log" >&2
    return 1
}

# Bound vision/prefill workloads and reserve 1.5 GiB of KV cache per server.
VLLM_ARGS=(
    --host 127.0.0.1
    --max-model-len 8192
    --max-num-seqs 1
    --max-num-batched-tokens 1024
    --enforce-eager
    --kv-cache-memory-bytes 1610612736
    --limit-mm-per-prompt '{"image":16,"video":0}'
    --mm-processor-kwargs '{"max_pixels":65536}'
    --enable-prefix-caching
    --enable-chunked-prefill
)

start_service main env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MAIN_MODEL" --port 7060 \
    --gpu-memory-utilization 0.45 "${VLLM_ARGS[@]}"
wait_ready main "$MAIN_API_BASE/models" "$LOG_DIR/main.log"

start_service summary env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    -m vllm.entrypoints.openai.api_server \
    --model "$SUMMARY_MODEL_PATH" --served-model-name "$SUMMARIZER_MODEL" --port 8065 \
    --gpu-memory-utilization 0.50 "${VLLM_ARGS[@]}"
wait_ready summary "$SUMMARIZER_API_BASE/models" "$LOG_DIR/summary.log"

start_service adapter bash services/webinfer/scripts/start_adapter.sh --max-pixels 65536
wait_ready adapter http://127.0.0.1:8070/v1/models "$LOG_DIR/adapter.log"

start_service webui bash services/webui/scripts/start_server.sh --no-ssl
wait_ready webui http://127.0.0.1:8099/ "$LOG_DIR/webui.log"
echo "Open http://127.0.0.1:8099 (Ctrl+C stops all services)."

wait -n "${PIDS[@]}"
echo "A service exited; stopping the remaining services. Logs: $LOG_DIR" >&2
exit 1
