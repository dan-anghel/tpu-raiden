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

#include "tpu_raiden/kv_cache/kv_cache_store.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/create_channel.h"
#include "grpcpp/grpcpp.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/lru_cache.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStore::KVCacheStore(size_t capacity, std::string global_registry_address,
                           RaidenId raiden_id)
    : lru_cache_(capacity), raiden_id_(std::move(raiden_id)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_unique<global_registry::GlobalRegistryClient>(channel);
  }
}

KVCacheStore::~KVCacheStore() = default;

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes, bool enable_global) {
  BlockSliceList results;

  size_t local_hits = 0;
  size_t limit = 0;
  {
    absl::MutexLock lock(mutex_);
    limit = std::min(block_hashes.size(), lru_cache_.available_space());
    results.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      const std::string& hash = block_hashes[i];
      std::vector<RaidenBlockID>* existing = lru_cache_.Get(hash);
      if (!existing || existing->empty()) {
        break;
      }
      results.push_back(std::make_pair(hash, *existing));
      local_hits++;
    }
  }

  if (enable_global && local_hits < limit && registry_client_) {
    std::vector<std::string> remaining_hashes(block_hashes.begin() + local_hits,
                                              block_hashes.begin() + limit);
    auto global_results_or = registry_client_->Lookup(remaining_hashes);
    if (global_results_or.ok()) {
      const auto& global_results = global_results_or.value();
      for (size_t i = 0; i < global_results.size(); ++i) {
        const auto& metadata = global_results[i];
        RaidenId remote_id;
        remote_id.job_name = metadata.host_address();
        remote_id.job_replica_id = "0";
        remote_id.data_name = "kv_cache";
        remote_id.data_replica_idx = metadata.block_id();
        results.push_back(std::make_pair(
            remaining_hashes[i], std::vector<RaidenBlockID>{RaidenBlockID(
                                     remote_id, -1, BlockStatus::REMOTE)}));
      }
    } else {
      LOG(WARNING) << "Global registry lookup failed: "
                   << global_results_or.status().message();
    }
  }

  return results;
}

std::pair<bool, BlockSliceList> KVCacheStore::Insert(
    const std::vector<std::string>& block_hashes,
    const std::vector<std::vector<RaidenBlockID>>& slices, bool /*on_host*/) {
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;
  bool all_inserted = true;

  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    if (lru_cache_.Contains(hash)) {
      // NOTE(jcgu): This is technically true, as the key already exists in the
      // cache. However, for the purpose of this insert, we treat it as if it
      // was inserted.
      all_inserted = false;
      continue;
    }
    std::optional<std::pair<std::string, std::vector<RaidenBlockID>>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, {});
    }
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

// Categorizes block_hashes into existing (local) and new (remote) hashes.
// 1. Pins all existing block hashes.
// 2. Inserts new block hashes into the LRU cache if space permits.
// 3. Pins the newly inserted block hashes, with full rollback on failure.
std::pair<bool, BlockSliceList> KVCacheStore::InsertAndPin(
    const std::vector<std::string>& block_hashes,
    const std::vector<std::vector<RaidenBlockID>>& slices, bool /*on_host*/) {
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  // 1. Pin all existing block_hashes
  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return std::make_pair(false, std::move(evicted_entries));
    }
  }

  // 2. Check if free space in lru_cache can hold all new block_hashes
  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return std::make_pair(false, std::move(evicted_entries));
  }

  // Insert all new block hashes into the lru cache list
  for (size_t i : new_indices) {
    const std::string& hash = block_hashes[i];
    std::optional<std::pair<std::string, std::vector<RaidenBlockID>>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, {});
    }
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  // 3. Pin the newly inserted block_hashes in the lru cache list
  for (size_t idx = 0; idx < new_indices.size(); ++idx) {
    size_t i = new_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[new_indices[j]]);
      }
      for (size_t j : existing_indices) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      for (size_t j : new_indices) {
        lru_cache_.Erase(block_hashes[j]);
      }
      for (auto it = evicted_entries.rbegin(); it != evicted_entries.rend();
           ++it) {
        lru_cache_.PutBack(it->first, std::move(it->second));
      }
      return std::make_pair(false, BlockSliceList{});
    }
  }

  return std::make_pair(true, std::move(evicted_entries));
}

// Reverts an InsertAndPin operation.
// 1. Unpins all block hashes in the LRU cache.
// 2. Erases remote block hashes whose pin count reaches 0.
// 3. Restores evicted entries to the back of the LRU cache in reverse order for
//    each deleted remote block. Returns the number of deleted remote blocks and
//    the remaining unrestored evicted entries.
std::pair<size_t, BlockSliceList> KVCacheStore::ReleaseAndDelete(
    const std::vector<std::string>& block_hashes,
    BlockSliceList pending_evict_entries) {
  absl::MutexLock lock(mutex_);
  size_t deleted_remote_blocks = 0;
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && !val->empty() &&
        (*val)[0].status == BlockStatus::REMOTE &&
        lru_cache_.GetPinCount(hash) == 0) {
      lru_cache_.Erase(hash);
      deleted_remote_blocks++;
    }
  }

  if (deleted_remote_blocks > pending_evict_entries.size()) {
    LOG(WARNING) << "Number of deleted remote blocks (" << deleted_remote_blocks
                 << ") exceeds number of pending evict entries ("
                 << pending_evict_entries.size() << ").";
  }

  size_t to_restore =
      std::min(deleted_remote_blocks, pending_evict_entries.size());
  for (size_t i = 0; i < to_restore; ++i) {
    auto& entry = pending_evict_entries.back();
    lru_cache_.PutBack(entry.first, std::move(entry.second));
    pending_evict_entries.pop_back();
  }

  return std::make_pair(deleted_remote_blocks,
                        std::move(pending_evict_entries));
}

void KVCacheStore::Delete(
    const std::vector<std::string>& block_hashes,
    const std::vector<std::vector<RaidenBlockID>>& slices) {
  absl::MutexLock lock(mutex_);
  for (const std::string& hash : block_hashes) {
    lru_cache_.Erase(hash);
  }
}

bool KVCacheStore::Pin(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void KVCacheStore::Release(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
  }
}

int KVCacheStore::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t KVCacheStore::capacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

}  // namespace kv_cache
}  // namespace tpu_raiden
