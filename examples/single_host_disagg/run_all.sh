#!/bin/bash

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# One-shot verification of tpu_raiden_connector in single-host disaggregated
# serving. Both engines (prefill = KV producer, decode = KV consumer) run locally
# on distinct TPU chips (chip 0 and chip 1).
#
# ASSUMES THE LOCAL TPUs ARE FREE -- there is NO pre-run cleanup / TPU process
# killing. Starts router + prefill + decode, waits for both engines to be healthy,
# runs the benchmark, prints results, then stops only the servers this script
# started.
#
# Prerequisites: the venv is active and `bash setup.sh` has been run. Then:
#   bash run_all.sh
set -u
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPTS_DIR}/tmp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"

wait_for_server() {
  local host=$1 port=$2 pid=$3
  echo "Waiting for server at ${host}:${port}..."
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/health >/dev/null 2>&1; do
      if [ -n '${pid}' ] && ! kill -0 ${pid} 2>/dev/null; then
        echo \"Error: server on ${host}:${port} (PID ${pid}) exited before becoming healthy\" >&2
        exit 1
      fi
      sleep 1
    done" && echo "Server at ${host}:${port} is ready." && return 0
  echo "Error: timed out waiting for ${host}:${port}" >&2
  return 1
}

# Stop ONLY the servers this run launched (the local router/prefill/decode
# children). This is run teardown, not a TPU-wide cleanup.
shutdown() {
  echo "Stopping servers started by this run..."
  [ -n "${ROUTER_PID:-}" ]  && kill "${ROUTER_PID}"  2>/dev/null
  [ -n "${PREFILL_PID:-}" ] && kill "${PREFILL_PID}" 2>/dev/null
  [ -n "${DECODE_PID:-}" ]  && kill "${DECODE_PID}"  2>/dev/null
  # Single-host: both engines are local; reap the vllm worker trees they spawned.
  pkill -9 -u "$(whoami)" -f "bin/vllm serve" 2>/dev/null
  pkill -9 -u "$(whoami)" -f "VLLM::Eng" 2>/dev/null
  pkill -9 -f "toy_proxy_server" 2>/dev/null
}
trap shutdown EXIT INT TERM

# 1. Router
echo "Starting router... (log: ${LOG_DIR}/router.log)"
bash "${SCRIPTS_DIR}/router.sh" >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

# 2. Prefill (local) + decode (remote) in parallel
echo "Starting prefill server (local :8400)... (log: ${LOG_DIR}/prefill.log)"
bash "${SCRIPTS_DIR}/prefill.sh" >"${LOG_DIR}/prefill.log" 2>&1 &
PREFILL_PID=$!

echo "Starting decode server (local :9400)... (log: ${LOG_DIR}/decode.log)"
bash "${SCRIPTS_DIR}/decode.sh" >"${LOG_DIR}/decode.log" 2>&1 &
DECODE_PID=$!

# 3. Wait for health
wait_for_server localhost 8400 "${PREFILL_PID}" || exit 1
wait_for_server localhost 9400 "${DECODE_PID}" || exit 1

# 4. Benchmark
echo "Both servers ready. Running benchmark... (log: ${LOG_DIR}/bm.log)"
bash "${SCRIPTS_DIR}/bm.sh" >"${LOG_DIR}/bm.log" 2>&1
cat "${LOG_DIR}/bm.log"
echo ""
echo "Done. Logs in ${LOG_DIR}"