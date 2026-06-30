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
# Disagg proxy/router (runs on the prefill VM): fans requests to the local prefill
# (8400) and the REMOTE decode (DECODE_HOST:9400) servers.
set -u
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tpu-inference is cloned into the hidden in-tree .src/ by setup.sh.
SRC_DIR="${SRC_DIR:-${SCRIPTS_DIR}/.src}"

# The decode VM's IP/host -- exported by run_all.sh from its first argument.
DECODE_HOST="${DECODE_HOST:?DECODE_HOST is not set (run via run_all.sh <decode-ip>)}"

python3 "${SRC_DIR}/tpu-inference/examples/disagg/toy_proxy_server.py" \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts localhost \
    --prefiller-ports 8400 \
    --decoder-hosts "${DECODE_HOST}" \
    --decoder-ports 9400