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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "torch_raw_transfer",
    srcs = ["torch_raw_transfer.py"],
    deps = [
        "//frameworks/torch:_torch_raw_transfer",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gf",
    timeout = "long",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_perf_test",
    tests = [
        ":torch_raw_transfer_perf_test_gf",
        ":torch_raw_transfer_perf_test_gl",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gf",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_test",
    tests = [
        ":torch_raw_transfer_test_gf",
        ":torch_raw_transfer_test_gl",
    ],
)
