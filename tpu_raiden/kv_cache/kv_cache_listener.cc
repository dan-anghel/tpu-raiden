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

#include "tpu_raiden/kv_cache/kv_cache_listener.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;

KVCacheListener::KVCacheListener(KVCacheManagerBase* engine,
                                 int listener_port)
    : engine_(engine), listener_port_(listener_port) {
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create C++ KVCacheListener socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  int v6only = 0;
  if (setsockopt(server_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only))) {
    LOG(WARNING) << "setsockopt IPV6_V6ONLY=0 failed: " << std::strerror(errno);
  }

  sockaddr_in6 address{
      .sin6_family = AF_INET6,
      .sin6_port = htons(listener_port_),
      .sin6_addr = in6addr_any,
  };

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    LOG(FATAL) << "C++ KVCacheListener bind failed on port "
               << listener_port_ << ": " << std::strerror(errno);
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "C++ KVCacheListener listen failed: "
               << std::strerror(errno);
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                  &addr_len) == 0) {
    listener_port_ = ntohs(address.sin6_port);
  }

  LOG(INFO) << "Native C++ KVCacheListener actively listening on port: "
            << listener_port_;
  LOG(ERROR) << "JETS_DEBUG: KVCacheListener constructor finished, port=" << listener_port_;

  listener_thread_ = std::thread(&KVCacheListener::ListenerLoop, this);
}

KVCacheListener::~KVCacheListener() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    int sock = socket(AF_INET6, SOCK_STREAM, 0);
    if (sock >= 0) {
      sockaddr_in6 serv_addr{};
      serv_addr.sin6_family = AF_INET6;
      serv_addr.sin6_port = htons(listener_port_);
      inet_pton(AF_INET6, "::1", &serv_addr.sin6_addr);
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv_addr),
                  sizeof(serv_addr)) == 0) {
        ControlRequest req;
        req.set_command(ControlRequest::COMMAND_SHUTDOWN);
        std::string payload;
        if (req.SerializeToString(&payload)) {
          uint32_t net_len = htonl(payload.size());
          write(sock, &net_len, sizeof(net_len));
          write(sock, payload.data(), payload.size());
        }
      }
      close(sock);
    }
    close(server_fd_);
  }

  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }

  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void KVCacheListener::ListenerLoop() {
  while (!stopping_) {
    sockaddr_in6 client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (stopping_) break;
      LOG(ERROR) << "JETS_DEBUG: accept failed on port " << listener_port_ << ": " << std::strerror(errno);
      continue;
    }
    LOG(ERROR) << "JETS_DEBUG: accepted connection on port " << listener_port_ << " from fd " << client_fd;

    worker_threads_.push_back(
        std::thread(&KVCacheListener::ConnectionWorker, this, client_fd));
  }
}

void KVCacheListener::ConnectionWorker(int client_fd) {
  LOG(ERROR) << "JETS_DEBUG: ConnectionWorker started for fd " << client_fd;
  uint32_t net_len = 0;
  if (read(client_fd, &net_len, sizeof(net_len)) != sizeof(net_len)) {
    LOG(ERROR) << "JETS_DEBUG: failed to read net_len for fd " << client_fd << ": " << std::strerror(errno);
    close(client_fd);
    return;
  }
  uint32_t payload_len = ntohl(net_len);

  LOG(ERROR) << "JETS_DEBUG: read net_len=" << payload_len << " for fd " << client_fd;
  std::vector<char> buffer(payload_len);
  size_t total_read = 0;
  while (total_read < payload_len) {
    ssize_t n =
        read(client_fd, buffer.data() + total_read, payload_len - total_read);
    if (n <= 0) {
      LOG(ERROR) << "JETS_DEBUG: failed to read payload for fd " << client_fd << ", read " << total_read << "/" << payload_len << ": " << std::strerror(errno);
      close(client_fd);
      return;
    }
    total_read += n;
  }

  ControlRequest req;
  if (!req.ParseFromString(absl::string_view(buffer.data(), buffer.size()))) {
    LOG(ERROR) << "JETS_DEBUG: Failed to parse ControlRequest Protobuf for fd " << client_fd;
    close(client_fd);
    return;
  }
  LOG(ERROR) << "JETS_DEBUG: parsed request command=" << req.command() << " for fd " << client_fd;

  ControlResponse resp;
  resp.set_success(true);
  resp.set_message("SUCCESS");

  if (req.command() == ControlRequest::COMMAND_START_TRANSFER) {
    if (req.has_start_transfer_request()) {
      const auto& start_req = req.start_transfer_request();
      if (start_req.is_sender()) {
        LOG(ERROR) << "JETS_DEBUG: C++ KVCacheListener received START_TRANSFER (Sender) for uuid " << start_req.uuid() << " on fd " << client_fd;
        {
          LOG(ERROR) << "JETS_DEBUG: StartTransferRequest Details:";
          LOG(ERROR) << "  uuid: " << start_req.uuid();
          LOG(ERROR) << "  is_sender: " << start_req.is_sender();
          LOG(ERROR) << "  req_id: " << start_req.req_id();
          LOG(ERROR) << "  schedules count: " << start_req.shard_push_schedules_size();
          for (const auto& [shard_idx, schedule] : start_req.shard_push_schedules()) {
            std::set<std::string> peers;
            int64_t total_bytes = 0;
            int min_src = std::numeric_limits<int>::max();
            int max_src = std::numeric_limits<int>::min();
            int min_dst = std::numeric_limits<int>::max();
            int max_dst = std::numeric_limits<int>::min();
            for (const auto& entry : schedule.entries()) {
              peers.insert(entry.dst_peer());
              total_bytes += entry.size_bytes() * entry.count();
              min_src = std::min(min_src, static_cast<int>(entry.src_block_id()));
              max_src = std::max(max_src, static_cast<int>(entry.src_block_id()));
              min_dst = std::min(min_dst, static_cast<int>(entry.dst_block_id()));
              max_dst = std::max(max_dst, static_cast<int>(entry.dst_block_id()));
            }
            std::string peers_str;
            for (const auto& p : peers) peers_str += p + ", ";
            LOG(ERROR) << "  Schedule for Shard " << shard_idx << ":";
            LOG(ERROR) << "    entries: " << schedule.entries_size();
            LOG(ERROR) << "    peers: " << peers_str;
            LOG(ERROR) << "    total_bytes: " << total_bytes;
            LOG(ERROR) << "    src_blocks: [" << min_src << ", " << max_src << "]";
            LOG(ERROR) << "    dst_blocks: [" << min_dst << ", " << max_dst << "]";
          }
        }
        absl::Status status = engine_->PushKVCacheResharded(start_req);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "JETS_DEBUG: PushKVCacheResharded native execution failed: " << status;
        } else {
          LOG(ERROR) << "JETS_DEBUG: PushKVCacheResharded native execution succeeded for uuid " << start_req.uuid();
        }
      } else {
        LOG(ERROR) << "JETS_DEBUG: C++ KVCacheListener received START_TRANSFER (Receiver) for uuid " << start_req.uuid() << " on fd " << client_fd;
        absl::Status status = engine_->RegisterActivePlan(
            start_req.uuid(), start_req, /*is_sender=*/false);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "JETS_DEBUG: RegisterRecv native execution failed: " << status;
        } else {
          LOG(ERROR) << "JETS_DEBUG: RegisterRecv native execution succeeded for uuid " << start_req.uuid();
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing start_transfer_request");
      LOG(ERROR) << "JETS_DEBUG: Missing start_transfer_request in START_TRANSFER command for fd " << client_fd;
    }
  } else if (req.command() == ControlRequest::COMMAND_SHUTDOWN) {
    LOG(ERROR) << "JETS_DEBUG: C++ KVCacheListener received SHUTDOWN command on fd " << client_fd;
    absl::Status status = engine_->WaitForPendingWork();
    if (!status.ok()) {
      LOG(ERROR) << "JETS_DEBUG: WaitForPendingWork failed during shutdown: " << status;
    }
    stopping_ = true;
  } else {
    resp.set_success(false);
    resp.set_message("COMMAND_UNSPECIFIED");
    LOG(ERROR) << "JETS_DEBUG: C++ KVCacheListener received unknown or unspecified Protobuf command on fd " << client_fd;
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    if (write(client_fd, &resp_net_len, sizeof(resp_net_len)) != sizeof(resp_net_len)) {
      LOG(ERROR) << "JETS_DEBUG: failed to write response length for fd " << client_fd;
    }
    if (write(client_fd, resp_str.data(), resp_str.size()) != resp_str.size()) {
      LOG(ERROR) << "JETS_DEBUG: failed to write response payload for fd " << client_fd;
    } else {
      LOG(ERROR) << "JETS_DEBUG: sent response success=" << resp.success() << " message=" << resp.message() << " for fd " << client_fd;
    }
  } else {
    LOG(ERROR) << "JETS_DEBUG: failed to serialize response for fd " << client_fd;
  }
  close(client_fd);
}

}  // namespace kv_cache
}  // namespace tpu_raiden
