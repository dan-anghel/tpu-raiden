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
# Decode server (KV consumer) for tpu_raiden_connector MULTI-HOST disaggregated
# serving. Runs on the decode VM (launched over SSH by run_all.sh on the prefill
# VM); pulls KV from the prefill VM over the network.
set -u
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tpu-raiden repo root: examples/multihost_disagg/ -> ../.. (holds the source
# tree + the importable tpu_raiden package, for the build-from-source path).
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SCRIPTS_DIR}/.jax_cache}"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-info}

# Make tpu_raiden importable (no-op if wheel-installed; adds the source tree to
# PYTHONPATH if built from source). vllm + tpu_inference are pip-installed, so they
# need no PYTHONPATH.
source "${SCRIPTS_DIR}/raiden_env.sh"

# Raiden engine tunables (known-good defaults).
export RAIDEN_NUM_SLOTS=${RAIDEN_NUM_SLOTS:-30}
export RAIDEN_TRANSPORT_PARALLELISM=${RAIDEN_TRANSPORT_PARALLELISM:-32}

# MULTI-HOST: decode is on its own VM, so no localhost collision with prefill --
# use the default 9100. It pulls from the prefill's advertised endpoint (host:port
# threaded into each request by the proxy), not from this local port.
export TPU_KV_TRANSFER_PORT=${TPU_KV_TRANSFER_PORT:-9100}

KV_TRANSFER_CONFIG="{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_raiden_connector\",\"kv_role\":\"kv_consumer\"}"
echo "decode.sh --> tpu_raiden_connector (kv_consumer)"

MODEL_IMPL_TYPE=vllm vllm serve Qwen/Qwen3-32B \
  --load-format runai_streamer \
  --port 9400 \
  --max-model-len 9216 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype=fp8 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 8 \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  --async-scheduling --block-size 128