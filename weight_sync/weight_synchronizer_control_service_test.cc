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

// Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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

#include "weight_sync/weight_synchronizer_control_service.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "weight_sync/weight_synchronizer_base.h"
#include "weight_sync/weight_synchronizer_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

int ConnectToControlPort(int port) {
  int sock = socket(AF_INET6, SOCK_STREAM, 0);
  if (sock < 0) {
    return -1;
  }

  struct sockaddr_in6 addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_port = htons(port);
  addr.sin6_addr = in6addr_loopback;

  if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) <
      0) {
    close(sock);
    return -1;
  }
  return sock;
}

TEST(WeightSynchronizerControlServiceTest, PushWeightsCommandSuccess) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*control_port=*/std::nullopt);

  WeightSynchronizerControlService control_service(&engine, /*control_port=*/0);
  ASSERT_GT(control_service.control_port(), 0);
  EXPECT_TRUE(control_service.is_active());

  int sock = ConnectToControlPort(control_service.control_port());
  ASSERT_GE(sock, 0);

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*control_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers("127.0.0.1:" + std::to_string(*dst_engine.local_port()));

  std::string payload;
  ASSERT_TRUE(req.SerializeToString(&payload));
  uint32_t req_len = htonl(payload.size());

  EXPECT_EQ(write(sock, &req_len, sizeof(req_len)), sizeof(req_len));
  EXPECT_EQ(write(sock, payload.data(), payload.size()), payload.size());

  // Read response
  uint32_t resp_len_net = 0;
  ASSERT_EQ(read(sock, &resp_len_net, sizeof(resp_len_net)),
            sizeof(resp_len_net));
  uint32_t resp_len = ntohl(resp_len_net);

  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(read(sock, resp_bytes.data(), resp_len), resp_len);

  ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(resp.success());

  close(sock);
}

TEST(WeightSynchronizerControlServiceTest, ShutdownCommandStopsService) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*control_port=*/std::nullopt);

  WeightSynchronizerControlService control_service(&engine, /*control_port=*/0);
  EXPECT_TRUE(control_service.is_active());

  int sock = ConnectToControlPort(control_service.control_port());
  ASSERT_GE(sock, 0);

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_SHUTDOWN);

  std::string payload;
  ASSERT_TRUE(req.SerializeToString(&payload));
  uint32_t req_len = htonl(payload.size());

  EXPECT_EQ(write(sock, &req_len, sizeof(req_len)), sizeof(req_len));
  EXPECT_EQ(write(sock, payload.data(), payload.size()), payload.size());

  // Read response
  uint32_t resp_len_net = 0;
  ASSERT_EQ(read(sock, &resp_len_net, sizeof(resp_len_net)),
            sizeof(resp_len_net));
  uint32_t resp_len = ntohl(resp_len_net);

  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(read(sock, resp_bytes.data(), resp_len), resp_len);

  ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(resp.success());

  close(sock);
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
