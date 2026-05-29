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

#!/bin/bash
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
BAZEL_CACHE_BASE="${BAZEL_CACHE_DIR:-${HOME}/.bazel_cache}"
BAZEL_DISK_CACHE="${BAZEL_CACHE_BASE}/disk_cache"
BAZEL_REPO_CACHE="${BAZEL_CACHE_BASE}/repo_cache"
BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-/tmp/tpu_raiden_bazel_output_${USER}}"

echo "=== Navigating to workspace directory ==="
cd "${WORKSPACE_DIR}"

# 0. Set up standalone Bazel environment based on .bazelversion in /tmp
BAZEL_VERSION="7.7.0"
VERSION_FILE="${WORKSPACE_DIR}/.bazelversion"
if [[ -f "${VERSION_FILE}" ]]; then
  BAZEL_VERSION="$(cat "${VERSION_FILE}" | tr -d '\r\n ')"
  echo "Parsed target Bazel version from .bazelversion: ${BAZEL_VERSION}"
fi

BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -f "${BAZEL_BIN}" ]]; then
  echo "Bootstrapping standalone Bazel ${BAZEL_VERSION} to temporary folder ${BAZEL_BIN}..."
  curl -Lo "${BAZEL_BIN}" "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_BIN}"
fi

"${BAZEL_BIN}" --version

# Default behavior based on auto-detection
BUILD_JAX=true
BUILD_TORCH=true

if [ ! -d "../torch_tpu" ]; then
  echo "Sibling torch_tpu checkout not found. Defaulting to JAX-only build."
  BUILD_TORCH=false
fi

# Parse command line arguments
if [ "$#" -gt 0 ]; then
  case "$1" in
    jax)
      BUILD_JAX=true
      BUILD_TORCH=false
      ;;
    torch)
      BUILD_JAX=false
      BUILD_TORCH=true
      ;;
    both)
      BUILD_JAX=true
      BUILD_TORCH=true
      ;;
    *)
      echo "Usage: $0 [jax|torch|both]"
      exit 1
      ;;
  esac
fi

BAZEL_TARGETS=()
DEFINE_FLAGS=""

if [ "$BUILD_JAX" = true ]; then
  echo "Configuring build for JAX..."
  BAZEL_TARGETS+=(
    "//raiden_lib/raw_transfer/jax:raw_transfer"
    "//api/jax:_kv_cache_manager"
    "//api/jax:_kv_cache_manager_ffi"
    "//api/jax:_weight_synchronizer"
  )
else
  DEFINE_FLAGS+=" --define with_jax=false"
fi

if [ "$BUILD_TORCH" = true ]; then
  echo "Configuring build for Torch..."
  BAZEL_TARGETS+=(
    "//raiden_lib/raw_transfer/torch:_torch_raw_transfer"
    "//api/torch:_kv_cache_manager"
    "//api/torch:_weight_synchronizer"
  )
else
  DEFINE_FLAGS+=" --define with_torch=false"
fi

if [ ${#BAZEL_TARGETS[@]} -eq 0 ]; then
  echo "No targets selected to build!"
  exit 1
fi

echo "=== Building targets with Bazel ==="
"${BAZEL_BIN}" --install_base="${BAZEL_OUTPUT_BASE}/install_base" --output_base="${BAZEL_OUTPUT_BASE}" --host_jvm_args="-Xmx32g" --host_jvm_args="-Xms2g" build -c opt --check_visibility=false --verbose_failures --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12} \
  --repo_env=PIP_INDEX_URL="https://pypi.org/simple" \
  --repo_env=PIP_EXTRA_INDEX_URL="" \
  --repo_env=PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring" \
  --repo_env=PIP_CONFIG_FILE="/dev/null" \
  "${BAZEL_TARGETS[@]}" \
  ${DEFINE_FLAGS} \
  --disk_cache=${BAZEL_DISK_CACHE} \
  --repository_cache=${BAZEL_REPO_CACHE} \
  "$@"


echo "=== Copying compiled shared libraries to source directory ==="
if [ "$BUILD_JAX" = true ]; then
  echo "Copying JAX artifacts..."
  cp -f "${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/jax/raw_transfer.so" "${WORKSPACE_DIR}/raiden_lib/raw_transfer/jax/"
  cp -f "${WORKSPACE_DIR}/bazel-bin/api/jax/_kv_cache_manager.so" "${WORKSPACE_DIR}/api/jax/"
  cp -f "${WORKSPACE_DIR}/bazel-bin/api/jax/_kv_cache_manager_ffi.so" "${WORKSPACE_DIR}/api/jax/"
  cp -f "${WORKSPACE_DIR}/bazel-bin/api/jax/_weight_synchronizer.so" "${WORKSPACE_DIR}/api/jax/"
fi

if [ "$BUILD_TORCH" = true ]; then
  echo "Copying Torch artifacts..."
  cp -f "${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/torch/_torch_raw_transfer.so" "${WORKSPACE_DIR}/raiden_lib/raw_transfer/torch/"
  cp -f "${WORKSPACE_DIR}/bazel-bin/api/torch/_kv_cache_manager.so" "${WORKSPACE_DIR}/api/torch/"
  cp -f "${WORKSPACE_DIR}/bazel-bin/api/torch/_weight_synchronizer.so" "${WORKSPACE_DIR}/api/torch/"
fi


echo "=== Build Complete! ==="
if [ "$BUILD_JAX" = true ]; then
  echo "JAX Artifacts are located in: ${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/jax/"
fi
if [ "$BUILD_TORCH" = true ]; then
  echo "Torch Artifacts are located in: ${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/torch/"
fi

echo "=== Install Python Dependencies! ==="
pip install -r requirements.txt

echo "=== Installation Complete! ==="
