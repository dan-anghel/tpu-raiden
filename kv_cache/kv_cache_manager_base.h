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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/stream_executor/stream.h"
#include "kv_cache/logical_block_manager.h"
#include "raiden_lib/raw_transfer/raw_transfer_impl.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheManagerBase {
 public:
  // Core C++ Constructor wrapping raw PJRT buffers directly (used by JAX and
  // PyTorch E2E)
  KVCacheManagerBase(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      int block_size = 1, std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // Standard CPU-only Constructor for remote workers E2E
  KVCacheManagerBase(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int block_size = 1,
                     std::optional<int> local_port = std::nullopt,
                     std::optional<int> host_blocks_to_allocate = std::nullopt,
                     int parallelism = 1);

  virtual ~KVCacheManagerBase();

  // Async on-chip H2D offloads returning PJRT copy future E2E
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Async on-chip D2H offloads E2E
  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Auto-allocating offloads E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {},
                  int64_t entity_id = 0);

  // Symmetrical H2H writes E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Symmetrical H2H reads E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Pure StreamExecutor H2D copy using raw C++ device pointers
  absl::Status H2dDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  // Pure StreamExecutor D2H copy using raw C++ device pointers
  absl::Status D2hDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  absl::StatusOr<raiden::PjRtCopyFuture> H2dDirect(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {}, int64_t device_id = -1);

  absl::StatusOr<raiden::PjRtCopyFuture> D2hDirect(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {}, int64_t device_id = -1);

  void SetExternalHostBuffer(
      const std::vector<raiden::BufferHoldAndAlias>& buffer_holds);

  std::optional<int> local_port() const;

  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;

  void SetExternalHostPointers(const std::vector<const uint8_t*>& host_ptrs,
                               const std::vector<size_t>& host_sizes);

  size_t num_layers() const { return num_layers_; }
  size_t num_shards() const { return num_shards_; }
  size_t slice_byte_size() const { return slice_byte_size_; }

 protected:
  struct ShardBufferInfoBase : public raiden::BufferHoldAndAlias {
    const uint8_t* host_ptr = nullptr;
    size_t host_size = 0;
    size_t device_size = 0;
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_host_buffer = {
        nullptr, [](void*) {}};
  };

  struct LayerInfoBase {
    std::vector<ShardBufferInfoBase> shards;
  };

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
  int block_size_ = 1;
  int parallelism_ = 1;
  size_t shard_factor_ = 1;
  int64_t major_dim_size_ = 0;

  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;

  std::unique_ptr<LogicalBlockManager> block_manager_;

  struct BlockTransportServer;
  std::unique_ptr<BlockTransportServer> server_;

  std::vector<LayerInfoBase> layers_;

  absl::StatusOr<raiden::PjRtCopyFuture> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes, int64_t device_id = -1);

  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t blocks_per_stream,
                      const std::vector<int>& src_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t blocks_per_stream, size_t remote_blocks_per_stream,
                     int base_remote_id, const std::vector<int>& allocated_ids,
                     std::vector<absl::Status>& statuses);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_
