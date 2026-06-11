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

import torch
from typing import List, Any, Union

class RawHostBuffer:
  def __init__(self, size_bytes: int) -> None: ...
  @property
  def size_bytes(self) -> int: ...
  @property
  def data_ptr(self) -> int: ...
  @property
  def is_pjrt_backed(self) -> bool: ...

class PjRtCopyFuture:
  def Await(self) -> None: ...
  def wait(self) -> None: ...
  def IsReady(self) -> bool: ...
  def is_ready(self) -> bool: ...

class PreparedTorchRawTransfer:
  def __init__(
      self,
      tpu_tensor: torch.Tensor,
      host_buffer: RawHostBuffer,
      unsafe_skip_buffer_lock: bool = True,
  ) -> None: ...
  @property
  def physical_size_bytes(self) -> int: ...
  @property
  def host_buffer(self) -> RawHostBuffer: ...
  def d2h_async(self) -> PjRtCopyFuture: ...
  def h2d_async(self) -> PjRtCopyFuture: ...
  def d2h(self) -> None: ...
  def h2d(self) -> None: ...

def await_all(futures: Union[PjRtCopyFuture, List[PjRtCopyFuture]]) -> None: ...
def is_ready(futures: Union[PjRtCopyFuture, List[PjRtCopyFuture]]) -> bool: ...

def transfer_d2h_async(
    src_arr: torch.Tensor,
    dst_arr: torch.Tensor,
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_h2d_async(
    src_arr: torch.Tensor,
    dst_arr: torch.Tensor,
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_d2h(
    src_arr: torch.Tensor,
    dst_arr: torch.Tensor,
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> None: ...

def transfer_h2d(
    src_arr: torch.Tensor,
    dst_arr: torch.Tensor,
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> None: ...

def transfer_d2h_batch_async(
    src_arrs: List[torch.Tensor],
    dst_arrs: List[torch.Tensor],
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_h2d_batch_async(
    src_arrs: List[torch.Tensor],
    dst_arrs: List[torch.Tensor],
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_d2h_batch(
    src_arrs: List[torch.Tensor],
    dst_arrs: List[torch.Tensor],
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> None: ...

def transfer_h2d_batch(
    src_arrs: List[torch.Tensor],
    dst_arrs: List[torch.Tensor],
    *,
    src_offsets_major_dim: List[int] = ...,
    dst_offsets_major_dim: List[int] = ...,
    copy_sizes_major_dim: List[int] = ...,
) -> None: ...
