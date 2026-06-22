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
"""Controller RPC client for KV Cache resharding."""

from typing import Optional

from tpu_raiden.kv_cache import kv_cache_service_pb2
from tpu_raiden.rpc import raiden_controller
from tpu_raiden.rpc.raiden_controller import RaidenId
from tpu_raiden.rpc.raiden_controller import TransferPlan


def _raiden_id_to_proto(
    unit: RaidenId,
) -> kv_cache_service_pb2.RaidenIdProto:
  return kv_cache_service_pb2.RaidenIdProto(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
  )


# TODO(raiden-dev): We should be able to use the same RPC client for both KV
# cache transfer and weights sync. So this can go to Raiden controller.
class KVCacheWorkerRpcClient(raiden_controller.WorkerRpcClient):
  """Worker RPC client using KV Cache service proto."""

  def _encode_start_transfer(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> Optional[bytes]:
    if (
        target_id not in transfer_plan.src_units
        and target_id not in transfer_plan.dst_units
    ):
      return None

    peers = []
    for dst in transfer_plan.dst_units:
      dst_coords = transfer_plan.worker_data_addresses.get(
          dst, ["127.0.0.1:8000"]
      )
      peers.extend(dst_coords)

    req = kv_cache_service_pb2.ControlRequest(
        command=kv_cache_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=peers,
    )

    is_sender = target_id in transfer_plan.src_units
    start_req = kv_cache_service_pb2.StartTransferRequest(
        src_units=[_raiden_id_to_proto(u) for u in transfer_plan.src_units],
        dst_units=[_raiden_id_to_proto(u) for u in transfer_plan.dst_units],
        uuid=transfer_plan.uuid,
        is_sender=is_sender,
        dst_mem_type=int(transfer_plan.dst_mem_type),
        use_block_chunks=transfer_plan.use_block_chunks,
    )

    if transfer_plan.shard_push_schedules:
      if target_id in transfer_plan.dst_units:
        # Receiver path: send FILTERED plan, only containing entries for this receiver
        target_endpoints = transfer_plan.worker_data_addresses.get(
            target_id, []
        )
        for (
            src_unit,
            push_schedules,
        ) in transfer_plan.shard_push_schedules.items():
          src_replica_idx = int(src_unit.job_replica_id)
          # We assume each source worker has only 1 shard (shard_idx = 0)
          schedule = push_schedules.get(0)
          if schedule:
            schedule_proto = kv_cache_service_pb2.ShardPushScheduleProto()
            for (
                dst_peer,
                dst_shard_idx,
                dst_offset,
                src_offset,
                size,
                src_block_id,
                dst_block_id,
            ) in schedule:
              if dst_peer in target_endpoints:
                entry_proto = schedule_proto.entries.add()
                entry_proto.dst_peer = dst_peer
                entry_proto.dst_shard_idx = dst_shard_idx
                entry_proto.dst_offset_bytes = dst_offset
                entry_proto.src_offset_bytes = src_offset
                entry_proto.size_bytes = size
                entry_proto.src_block_id = src_block_id
                entry_proto.dst_block_id = dst_block_id
            if len(schedule_proto.entries) > 0:
              start_req.shard_push_schedules[src_replica_idx].CopyFrom(
                  schedule_proto
              )
      else:
        # Sender path: only send local schedule
        push_schedules = transfer_plan.shard_push_schedules.get(target_id)
        if push_schedules:
          for shard_idx, entries in push_schedules.items():
            schedule_proto = kv_cache_service_pb2.ShardPushScheduleProto()
            for (
                dst_peer,
                dst_shard_idx,
                dst_offset,
                src_offset,
                size,
                src_block_id,
                dst_block_id,
            ) in entries:
              entry_proto = schedule_proto.entries.add()
              entry_proto.dst_peer = dst_peer
              entry_proto.dst_shard_idx = dst_shard_idx
              entry_proto.dst_offset_bytes = dst_offset
              entry_proto.src_offset_bytes = src_offset
              entry_proto.size_bytes = size
              entry_proto.src_block_id = src_block_id
              entry_proto.dst_block_id = dst_block_id
            start_req.shard_push_schedules[shard_idx].CopyFrom(schedule_proto)

    req.start_transfer_request.CopyFrom(start_req)
    return req.SerializeToString()

  def _verify_response(self, resp_bytes: bytes) -> None:
    resp = kv_cache_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(
          f"KV Cache remote native execution failed: {resp.message}"
      )

  def _encode_shutdown(self) -> bytes:
    req = kv_cache_service_pb2.ControlRequest(
        command=kv_cache_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    return req.SerializeToString()
