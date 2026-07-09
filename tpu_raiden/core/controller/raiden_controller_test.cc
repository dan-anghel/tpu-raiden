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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<TestServer> test_server_;
};

TEST_F(RaidenControllerTest, AllocateAndDeallocateSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/10,
                                /*num_shards=*/2, /*shard_size_bytes=*/1024);

    // 10 logical blocks * 2 shards = 20 buffer allocations created on worker in
    // constructor.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.Allocate(/*num_blocks=*/3);
    ASSERT_TRUE(alloc_or.ok());
    const auto& sharded_buffers = *alloc_or;
    ASSERT_EQ(sharded_buffers.size(), 3);

    // Buffers remain on worker; locked blocks increase locally.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 3);

    // Verify structure of first sharded buffer.
    EXPECT_EQ(sharded_buffers[0].buffer_handles_size(), 2);
    EXPECT_NE(sharded_buffers[0].buffer_handles(0).handle(),
              sharded_buffers[0].buffer_handles(1).handle());

    // Deallocate the buffers locally.
    ASSERT_TRUE(controller.Deallocate(sharded_buffers).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  // When controller goes out of scope, its destructor deletes all buffers on
  // worker.
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, ConstructWithServerAddressWorks) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(unit_, test_server_->server_address,
                                /*num_blocks=*/5, /*num_shards=*/2,
                                /*shard_size_bytes=*/512);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.Allocate(/*num_blocks=*/2);
    ASSERT_TRUE(alloc_or.ok());
    EXPECT_EQ(alloc_or->size(), 2);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 2);

    ASSERT_TRUE(controller.Deallocate(*alloc_or).ok());
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, AllocateExceedingCapacityFails) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  auto alloc_or = controller.Allocate(/*num_blocks=*/10);
  EXPECT_FALSE(alloc_or.ok());
  EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
}

TEST_F(RaidenControllerTest, DeallocateNonExistentBufferFails) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  proto::BufferProto fake_buffer;
  fake_buffer.add_buffer_handles()->set_handle(9999);

  std::vector<proto::BufferProto> to_delete = {fake_buffer};
  EXPECT_FALSE(controller.Deallocate(to_delete).ok());
}

TEST_F(RaidenControllerTest, ConstructorThrowsOnBufferCreationFailure) {
  // Creating a controller with an invalid address fails to connect and create
  // buffers.
  EXPECT_THROW(
      {
        RaidenController controller(unit_, "localhost:1", /*num_blocks=*/5,
                                    /*num_shards=*/1, /*shard_size_bytes=*/512);
      },
      std::runtime_error);
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
