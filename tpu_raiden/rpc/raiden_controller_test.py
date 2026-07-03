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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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
"""Tests for Raiden Controller high-level transfer API under rpc/."""

import asyncio
from absl.testing import absltest
from tpu_raiden.rpc import raiden_controller


class DummyWorkerRpcClient(raiden_controller.WorkerRpcClient):

  async def start_transfer(self, target_id, transfer_plan) -> None:
    pass


class RaidenControllerTest(absltest.TestCase):

  def test_dynamic_balancing_and_overlap_planner(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=dummy_client
    )

    src_unit_0 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="0",
        data_name="kv_cache",
    )
    src_unit_1 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="1",
        data_name="kv_cache",
    )
    target_unit = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="215",
        data_name="kv_cache",
    )

    controller.register_work_unit(
        src_unit_0, ["10.0.0.1:8000", "10.0.0.2:8000"]
    )
    controller.register_work_unit(
        src_unit_1, ["10.0.0.3:8000", "10.0.0.4:8000"]
    )
    controller.register_work_unit(
        target_unit, ["10.0.0.5:8000", "10.0.0.6:8000"]
    )

    # First transfer routes to src_unit_0
    future_1 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    asyncio.run(future_1.wait())

    self.assertTrue(future_1.done())
    self.assertEqual(future_1.session_id, 0)

    plan_1 = controller.get_plan("req_0")
    self.assertEqual(plan_1.src_units[0].job_replica_id, "0")

    # Verify generalized NDSlice fully qualified overlap push schedule dict
    self.assertIn(src_unit_0, plan_1.plan)
    unit_0_plan = plan_1.plan[src_unit_0]
    self.assertLen(unit_0_plan, 2)
    self.assertEqual(unit_0_plan[0], [(target_unit, 0, [[(0, 2)]])])
    self.assertEqual(unit_0_plan[1], [(target_unit, 1, [[(0, 2)]])])

    # Second transfer routes dynamically to least-loaded src_unit_1
    future_2 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    self.assertEqual(future_2.session_id, 1)

    plan_2 = controller.get_plan("req_1")
    self.assertEqual(plan_2.src_units[0].job_replica_id, "1")

  def test_fan_out_multiple_targets(self):
    controller = raiden_controller.RaidenController(port=10001)

    src = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="layer0.weights",
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="10",
        data_name="layer0.weights",
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="11",
        data_name="layer0.weights",
    )

    controller.register_work_unit(src, ["10.0.0.1:8000"])
    controller.register_work_unit(target_0, ["10.0.0.2:8000"])
    controller.register_work_unit(target_1, ["10.0.0.3:8000"])

    future_multi = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1],
    )
    self.assertEqual(future_multi.session_id, 0)

    plan_multi = controller.get_plan("req_0")
    self.assertLen(plan_multi.dst_units, 2)

  def test_rpc_client_push_coordination(self):
    recorded_actions = []

    class MockWorkerClient(raiden_controller.WorkerRpcClient):

      async def start_transfer(self, orchestrator_id, plan) -> None:
        recorded_actions.append(("start", [orchestrator_id]))

    mock_client = MockWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10002, worker_rpc_client=mock_client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    dst = raiden_controller.RaidenId(
        job_name="sampler", job_replica_id="0", data_name="weights"
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
    )

    asyncio.run(future.wait())

    self.assertEqual(
        recorded_actions,
        [
            ("start", [dst]),
            ("start", [src]),
        ],
    )

  def test_enforce_metadata_completeness(self):
    controller = raiden_controller.RaidenController(port=10003)
    unit = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )

    # 1. Failure if some shape/layout fields are missing
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "all of them must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          # layout and global_shape are missing
      )

    # 2. Failure if itemsize is missing when metadata is provided
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "itemsize must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          layout=[4, 3, 2, 1, 0],
          global_shape=[128, 16, 8, 2, 128],
          # itemsize is missing
      )

    # 3. Success if all are provided
    controller.register_work_unit(
        unit,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1, 4, 1, 1],
        layout=[4, 3, 2, 1, 0],
        global_shape=[128, 16, 8, 2, 128],
        itemsize=4,
    )

  def test_multi_shard_worker_resharding(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=dummy_client
    )
    src = raiden_controller.RaidenId("trainer", "0", "weights")
    dst = raiden_controller.RaidenId("sampler", "1", "weights")

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8001", "10.0.0.1:8002", "10.0.0.1:8003"],
        mesh_shape=(2, 2),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )
    controller.register_work_unit(
        dst,
        ["10.0.0.2:8000", "10.0.0.2:8001", "10.0.0.2:8002", "10.0.0.2:8003"],
        mesh_shape=(1, 4),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("req_0")
    self.assertIn(src, plan.shard_push_schedules)
    schedules = plan.shard_push_schedules[src]
    self.assertSetEqual(set(schedules.keys()), {0, 1, 2, 3})


if __name__ == "__main__":
  absltest.main()
