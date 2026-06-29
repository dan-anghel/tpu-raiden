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
# Benchmark the disagg endpoint (router on :8000). 10 reqs, 1024-in/128-out @ 2 RPS.
# MODEL must match prefill.sh / decode.sh.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
vllm bench serve \
  --backend vllm \
  --host localhost \
  --port 8000 \
  --model "${MODEL}" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --num-prompts 10 \
  --request-rate 2 \
  --trust-remote-code \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "95,97,99"