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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_WORKER_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_WORKER_SERVICE_H_

#include <memory>

#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {

// Wrapper class that unites KVCacheManagerWithTransfer (managing device/host KV
// cache memory transfers) and WorkerServiceImpl (handling gRPC memory
// allocation requests from the Raiden controller). Uses a process-wide
// singleton gRPC server to avoid repeatedly creating the server across wrapper
// instances.
class KVCacheManagerWithWorkerService {
 public:
  // Constructs the wrapper with an existing transfer manager and an optional
  // host memory allocator. Automatically starts the singleton WorkerService
  // gRPC server on the given port (defaulting to 0 for an ephemeral port) if
  // not already running.
  explicit KVCacheManagerWithWorkerService(
      std::unique_ptr<KVCacheManagerWithTransfer> transfer_manager,
      std::shared_ptr<HostMemoryAllocator> host_allocator = nullptr,
      int grpc_port = 0);

  ~KVCacheManagerWithWorkerService();

  // Accessors for wrapped components.
  KVCacheManagerWithTransfer* transfer_manager() const {
    return transfer_manager_.get();
  }

  // Returns the listening port of the singleton gRPC server, or 0 if not
  // running.
  int GetGrpcPort() const;

 private:
  std::unique_ptr<KVCacheManagerWithTransfer> transfer_manager_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_WORKER_SERVICE_H_
