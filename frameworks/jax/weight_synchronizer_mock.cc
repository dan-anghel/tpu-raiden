// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optional>
#include <stdexcept>

#include "frameworks/jax/weight_synchronizer.h"
#include "weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace jax {

WeightSynchronizer::WeightSynchronizer(const nb::list& jax_arrays,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock,
                                       std::optional<int> control_port)
    : WeightSynchronizerBase({}, local_port, std::nullopt,
                             unsafe_skip_buffer_lock, parallelism,
                             control_port) {
  throw std::runtime_error(
      "Python WeightSynchronizer constructor is unsupported in pure C++ FFI "
      "unit tests.");
}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace jax
}  // namespace tpu_raiden
