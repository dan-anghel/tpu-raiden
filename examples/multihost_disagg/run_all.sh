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
# One-shot verification of tpu_raiden_connector in MULTI-HOST disaggregated
# serving. The prefill (KV producer) + router run on THIS VM; the decode (KV
# consumer) runs on a SECOND VM, started here over SSH. The decode pulls KV from
# this VM's prefill over the network.
#
# ASSUMES THE TPUs ON BOTH VMs ARE FREE -- there is no pre-run cleanup. Starts
# router + prefill (local) + decode (remote via ssh), waits for both engines to
# be healthy, runs the benchmark, prints results, then stops only what it started
# (local children + a remote vllm pkill).
#
# Prerequisites (see README.md):
#   - Both VMs: tpu-raiden installed + `bash setup.sh` run, in a venv.
#   - Passwordless SSH from this VM to the decode VM (Step 3 in the README).
# Usage:
#   bash run_all.sh <decode-vm-ip>
#   # or: DECODE_HOST=<ip> bash run_all.sh
set -u

DECODE_HOST="${1:-${DECODE_HOST:-}}"
if [ -z "${DECODE_HOST}" ]; then
  echo "Usage: bash run_all.sh <decode-vm-ip>   (or set DECODE_HOST=<ip>)" >&2
  exit 2
fi
export DECODE_HOST

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPTS_DIR}/tmp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"

# --- Remote (decode VM) settings -- override if the two VMs differ. -----------
# The decode VM is assumed to mirror this one: same user, same repo path, same
# venv path. The active venv is reused on the remote so decode.sh can resolve
# tpu_raiden / vllm there.
RAIDEN_VENV="${RAIDEN_VENV:-${VIRTUAL_ENV:-}}"
REMOTE_DIR="${REMOTE_DIR:-${SCRIPTS_DIR}}"
SSH_TARGET="${SSH_USER:+${SSH_USER}@}${DECODE_HOST}"
# BatchMode: fail fast (don't hang on a password) if SSH keys aren't set up.
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
if [ -z "${RAIDEN_VENV}" ]; then
  echo "ERROR: no active venv (VIRTUAL_ENV is empty) and RAIDEN_VENV unset." >&2
  echo "Activate the venv (so the remote decode can be launched in the same one)," >&2
  echo "or set RAIDEN_VENV=<path-to-venv-on-decode-vm>." >&2
  exit 1
fi

wait_for_server() {
  local host=$1 port=$2 pid=$3
  echo "Waiting for server at ${host}:${port}..."
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/health >/dev/null 2>&1; do
      if [ -n '${pid}' ] && ! kill -0 ${pid} 2>/dev/null; then
        echo \"Error: server for ${host}:${port} (PID ${pid}) exited before becoming healthy\" >&2
        exit 1
      fi
      sleep 1
    done" && echo "Server at ${host}:${port} is ready." && return 0
  echo "Error: timed out waiting for ${host}:${port}" >&2
  return 1
}

# Stop ONLY what this run launched: the local router/prefill children, the SSH
# session to the decode VM, and a best-effort remote kill of the decode port.
# This is run teardown, not a TPU-wide cleanup.
shutdown() {
  echo "Stopping servers started by this run..."
  [ -n "${ROUTER_PID:-}" ]     && kill "${ROUTER_PID}"     2>/dev/null
  [ -n "${PREFILL_PID:-}" ]    && kill "${PREFILL_PID}"    2>/dev/null
  [ -n "${DECODE_SSH_PID:-}" ] && kill "${DECODE_SSH_PID}" 2>/dev/null
  # Local: reap the prefill vllm worker tree + the proxy.
  pkill -9 -u "$(whoami)" -f "bin/vllm serve" 2>/dev/null
  pkill -9 -u "$(whoami)" -f "VLLM::Eng" 2>/dev/null
  pkill -9 -f "toy_proxy_server" 2>/dev/null
  # Remote: kill whatever holds the decode port :9400 on the decode VM (surgical
  # -- avoids reaping unrelated processes there).
  timeout 20 ssh ${SSH_OPTS} "${SSH_TARGET}" \
    "lsof -ti:9400 2>/dev/null | xargs -r kill -9" 2>/dev/null
}
trap shutdown EXIT INT TERM

# 1. Router (local; routes to local prefill :8400 and remote decode :9400)
echo "Starting router... (log: ${LOG_DIR}/router.log)"
bash "${SCRIPTS_DIR}/router.sh" >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

# 2. Prefill (local) + decode (remote over SSH) in parallel
echo "Starting prefill server (local :8400)... (log: ${LOG_DIR}/prefill.log)"
bash "${SCRIPTS_DIR}/prefill.sh" >"${LOG_DIR}/prefill.log" 2>&1 &
PREFILL_PID=$!

echo "Starting decode server (remote ${DECODE_HOST}:9400 via ssh)... (log: ${LOG_DIR}/decode.log)"
echo "  remote venv: ${RAIDEN_VENV} | remote dir: ${REMOTE_DIR}"
ssh ${SSH_OPTS} "${SSH_TARGET}" \
  "source '${RAIDEN_VENV}/bin/activate' && cd '${REMOTE_DIR}' && exec bash decode.sh" \
  >"${LOG_DIR}/decode.log" 2>&1 &
DECODE_SSH_PID=$!

# 3. Wait for health (local prefill by PID; remote decode by the ssh PID)
wait_for_server localhost 8400 "${PREFILL_PID}" || exit 1
wait_for_server "${DECODE_HOST}" 9400 "${DECODE_SSH_PID}" || exit 1

# 4. Benchmark (against the local router :8000)
echo "Both servers ready. Running benchmark... (log: ${LOG_DIR}/bm.log)"
bash "${SCRIPTS_DIR}/bm.sh" >"${LOG_DIR}/bm.log" 2>&1
cat "${LOG_DIR}/bm.log"
echo ""
echo "Done. Logs in ${LOG_DIR} (decode.log is streamed from the decode VM)."