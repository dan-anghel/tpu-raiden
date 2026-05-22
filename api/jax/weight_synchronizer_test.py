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

"""Unit tests and usage examples for JAX WeightSynchronizer Python API."""

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from api.jax.weight_synchronizer import WeightSynchronizer


class WeightSynchronizerPythonApiTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Set up a local JAX CPU mesh for array sharding representation E2E
    devices = jax.devices("cpu")
    self.mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("data")
    )

  @mock.patch(
      "api.jax._weight_synchronizer.WeightSynchronizer",
      autospec=True,
  )
  def test_weight_synchronizer_api_usage_e2e(self, mock_impl):
    # Mock properties behavior cleanly for the FFI layer!
    mock_instance = mock_impl.return_value
    mock_instance.local_port = 43121
    mock_instance.num_layers = 1
    mock_instance.num_shards = 1
    mock_instance.slice_byte_size = 16384

    # 1. Prepare JAX weight arrays sharded over local devices E2E!
    shape = (64, 64)
    dtype = jnp.float32
    sharded_weights = jax.device_put(
        jnp.ones(shape, dtype=dtype), self.sharding
    )

    # 2. Instantiate the high-level WeightSynchronizer wrapper (documentation usage!)
    ws = WeightSynchronizer(
        jax_arrays=[sharded_weights], local_port=None, parallelism=1
    )

    # Verify wrapper correctly forwards parameters to Nanobind constructor
    mock_impl.assert_called_once_with([sharded_weights], None, 1)

    # Assert properties exposed from the FFI layer
    self.assertEqual(ws.local_port, 43121)
    self.assertEqual(ws.num_layers, 1)
    self.assertEqual(ws.num_shards, 1)
    self.assertEqual(ws.slice_byte_size, 16384)

    # ==========================================================================
    # Example A: Trainer pushing weights to peer inference servers E2E!
    # ==========================================================================
    peers = ["localhost:38219", "localhost:34359"]
    ws.push_weights(peers)

    # Verify underlying Nanobind PushWeights was triggered correctly
    mock_instance.PushWeights.assert_called_once_with(peers)

    # ==========================================================================
    # Example B: Inference server pulling weights from a source peer E2E!
    # ==========================================================================
    source_coordinate = "localhost:43121"
    ws.pull_weights(source_coordinate)

    # Verify underlying Nanobind PullWeights was triggered correctly
    mock_instance.PullWeights.assert_called_once_with(source_coordinate)


if __name__ == "__main__":
  absltest.main()
