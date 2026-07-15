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

#include "tpu_raiden/core/controller/raiden_controller.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/core/controller/controller_server.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

absl::StatusOr<proto::CreateBuffersResponse> CreateBuffersForWorker(
    WorkerServiceClient& client, const proto::CreateBuffersRequest& request,
    absl::string_view worker_id, int num_blocks) {
  ASSIGN_OR_RETURN(auto response, client.CreateBuffers(request));
  if (!response.success()) {
    return absl::InternalError(
        absl::StrCat("WorkerService CreateBuffers failed on worker ", worker_id,
                     " in RaidenController constructor: ", response.message()));
  }
  if (response.buffers_size() != num_blocks) {
    return absl::InternalError(absl::StrCat(
        "WorkerService on worker ", worker_id,
        " returned unexpected number of buffers: ", response.buffers_size(),
        " vs expected ", num_blocks));
  }
  return response;
}

absl::StatusOr<std::vector<proto::BufferProto>> CreateBuffersForAllWorkers(
    core::controller::WorkerRegistry& worker_registry,
    const rpc::RaidenIdProto& unit, int num_blocks, int num_shards,
    int64_t shard_size_bytes) {
  proto::CreateBuffersRequest request;
  *request.mutable_unit() = unit;
  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    auto* spec = request.add_buffers();
    spec->set_num_shards(num_shards);
    spec->set_size_bytes(shard_size_bytes);
  }

  auto registered_workers = worker_registry.GetRegisteredWorkers();
  if (registered_workers.empty()) {
    return absl::FailedPreconditionError(
        "No WorkerServiceClient available in RaidenController constructor");
  }

  std::vector<proto::BufferProto> created_buffers;
  created_buffers.reserve(num_blocks);
  for (const auto& reg : registered_workers) {
    if (!reg.worker_service_client) continue;

    ASSIGN_OR_RETURN(auto resp,
                     CreateBuffersForWorker(*reg.worker_service_client, request,
                                            reg.worker_id, num_blocks));

    if (created_buffers.empty()) {
      for (int block_id = 0; block_id < num_blocks; ++block_id) {
        created_buffers.push_back(resp.buffers(block_id));
      }
    }
  }

  if (created_buffers.empty()) {
    return absl::FailedPreconditionError(
        "No active WorkerServiceClient available to create buffers");
  }

  return created_buffers;
}

}  // namespace

RaidenController::RaidenController(
    const rpc::RaidenIdProto& unit,
    absl::Span<const std::string> worker_addresses, int num_blocks,
    int num_shards, int64_t shard_size_bytes)
    : unit_(unit),
      num_shards_(num_shards),
      shard_size_bytes_(shard_size_bytes),
      worker_registry_(std::make_shared<core::controller::WorkerRegistry>()),
      block_manager_(
          std::make_unique<kv_cache::LogicalBlockManager>(num_blocks)) {
  for (size_t i = 0; i < worker_addresses.size(); ++i) {
    std::string worker_id = absl::StrCat("worker_", i);
    absl::Status status = worker_registry_->RegisterWorker(
        worker_id, worker_addresses[i], /*raiden_transfer_endpoint=*/"");
    if (!status.ok()) {
      LOG(WARNING) << "Failed to register worker address "
                   << worker_addresses[i]
                   << " in RaidenController: " << status.message();
    }
  }

  InitControlPlaneAndBuffers(num_blocks);
}





void RaidenController::InitControlPlaneAndBuffers(int num_blocks) {
  absl::Status server_status =
      core::controller::ControllerServer::GetInstance().StartServer(
          worker_registry_, /*port=*/0);
  if (!server_status.ok()) {
    LOG(WARNING) << "Failed to start ControllerServer in RaidenController: "
                 << server_status.message();
  }

  auto buffers_or = CreateBuffersForAllWorkers(
      *worker_registry_, unit_, num_blocks, num_shards_, shard_size_bytes_);
  if (!buffers_or.ok()) {
    throw std::runtime_error(std::string(buffers_or.status().message()));
  }

  all_sharded_buffers_ = std::move(*buffers_or);
  for (int block_id = 0; block_id < all_sharded_buffers_.size(); ++block_id) {
    const auto& sharded_buf = all_sharded_buffers_[block_id];
    for (const auto& handle_proto : sharded_buf.buffer_handles()) {
      handle_to_block_id_[handle_proto.handle()] = block_id;
    }
  }
}

RaidenController::~RaidenController() {
  if (all_sharded_buffers_.empty() || !worker_registry_) return;

  proto::DeleteBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (const auto& sharded_buf : all_sharded_buffers_) {
    *request.add_sharded_buffers() = sharded_buf;
  }

  for (const auto& reg : worker_registry_->GetRegisteredWorkers()) {
    if (reg.worker_service_client) {
      auto resp_or = reg.worker_service_client->DeleteBuffers(request);
      if (!resp_or.ok()) {
        LOG(ERROR)
            << "Failed to delete buffers on worker in ~RaidenController: "
            << resp_or.status().message();
      } else if (!resp_or->success()) {
        LOG(ERROR)
            << "WorkerService DeleteBuffers failed in ~RaidenController: "
            << resp_or->message();
      }
    }
  }
}

absl::StatusOr<std::vector<proto::BufferProto>> RaidenController::Allocate(
    int num_blocks) {
  if (num_blocks <= 0) {
    return absl::InvalidArgumentError("num_blocks must be positive");
  }

  std::vector<int> block_ids;
  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(block_ids,
                     block_manager_->Allocate(num_blocks, /*lock=*/true));
  }

  std::vector<proto::BufferProto> sharded_buffers;
  sharded_buffers.reserve(block_ids.size());
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= all_sharded_buffers_.size()) {
      return absl::InternalError(
          absl::StrCat("Allocated block_id out of range: ", block_id,
                       ", total buffers: ", all_sharded_buffers_.size()));
    }
    sharded_buffers.push_back(all_sharded_buffers_[block_id]);
  }

  return sharded_buffers;
}

absl::Status RaidenController::Deallocate(
    absl::Span<const proto::BufferProto> sharded_buffers) {
  if (sharded_buffers.empty()) {
    return absl::OkStatus();
  }

  std::vector<int> block_ids_to_unlock;
  block_ids_to_unlock.reserve(sharded_buffers.size());

  for (const auto& sharded_buf : sharded_buffers) {
    if (sharded_buf.buffer_handles().empty()) {
      return absl::InvalidArgumentError("ShardedBuffer has no buffer_handles");
    }
    uint64_t handle = sharded_buf.buffer_handles(0).handle();
    auto it = handle_to_block_id_.find(handle);
    if (it == handle_to_block_id_.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ShardedBuffer handle not recognized by this RaidenController: ",
          handle));
    }
    block_ids_to_unlock.push_back(it->second);
  }

  absl::MutexLock lock(mutex_);
  RETURN_IF_ERROR(block_manager_->Unlock(block_ids_to_unlock));
  return absl::OkStatus();
}

absl::StatusOr<proto::TransferBuffersResponse>
RaidenController::TransferBuffers(rpc::MemoryType src_mem_type,
                                  rpc::MemoryType dst_mem_type,
                                  absl::Span<const int64_t> src_offsets,
                                  absl::Span<const int64_t> dst_offsets,
                                  absl::Span<const int64_t> copy_sizes,
                                  absl::string_view peer) {
  if (src_offsets.empty() || src_offsets.size() != dst_offsets.size()) {
    return absl::InvalidArgumentError(
        "Source and destination offsets must have the same non-zero length");
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_offsets.size()) {
    return absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_offsets");
  }

  proto::TransferBuffersRequest request;
  auto* transfer = request.mutable_transfer();
  transfer->set_src_mem_type(src_mem_type);
  transfer->set_dst_mem_type(dst_mem_type);
  transfer->mutable_src_offsets()->Add(src_offsets.begin(), src_offsets.end());
  transfer->mutable_dst_offsets()->Add(dst_offsets.begin(), dst_offsets.end());
  transfer->mutable_copy_sizes()->Add(copy_sizes.begin(), copy_sizes.end());
  if (!peer.empty()) {
    transfer->set_peer(std::string(peer));
  }


  auto workers = worker_registry_->GetRegisteredWorkers();
  if (workers.empty()) {
    return absl::FailedPreconditionError(
        "No registered workers available for TransferBuffers");
  }

  proto::TransferBuffersResponse last_response;
  bool issued_any = false;
  for (const auto& reg : workers) {
    if (reg.worker_service_client) {
      ASSIGN_OR_RETURN(last_response,
                       reg.worker_service_client->TransferBuffers(request));
      issued_any = true;
    }
  }

  if (!issued_any) {
    return absl::FailedPreconditionError(
        "No active WorkerServiceClient available for TransferBuffers");
  }

  return last_response;
}

}  // namespace controller
}  // namespace tpu_raiden
