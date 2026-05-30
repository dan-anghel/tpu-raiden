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

# Copyright 2026 Google LLC
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

"""JAX raw H2D/D2H transfer."""

# pylint: disable=g-import-not-at-top,g-bad-import-order
from frameworks.jax import _raw_transfer as _impl

# pylint: enable=g-import-not-at-top,g-bad-import-order

PjRtCopyFuture = _impl.PjRtCopyFuture
await_all = _impl.await_all
is_ready = _impl.is_ready
transfer_d2h_async = _impl.transfer_d2h_async
transfer_h2d_async = _impl.transfer_h2d_async
transfer_d2h = _impl.transfer_d2h
transfer_h2d = _impl.transfer_h2d
transfer_d2h_batch_async_naive = _impl.transfer_d2h_batch_async_naive
transfer_h2d_batch_async_naive = _impl.transfer_h2d_batch_async_naive
transfer_d2h_batch_async = _impl.transfer_d2h_batch_async
transfer_h2d_batch_async = _impl.transfer_h2d_batch_async
transfer_d2h_batch = _impl.transfer_d2h_batch
transfer_h2d_batch = _impl.transfer_h2d_batch

__all__ = [
    "PjRtCopyFuture",
    "await_all",
    "is_ready",
    "transfer_d2h_async",
    "transfer_h2d_async",
    "transfer_d2h",
    "transfer_h2d",
    "transfer_d2h_batch_async_naive",
    "transfer_h2d_batch_async_naive",
    "transfer_d2h_batch_async",
    "transfer_h2d_batch_async",
    "transfer_d2h_batch",
    "transfer_h2d_batch",
]
