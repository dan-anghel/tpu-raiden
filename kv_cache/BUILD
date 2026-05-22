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

load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "logical_block_manager",
    srcs = ["logical_block_manager.cc"],
    hdrs = ["logical_block_manager.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "logical_block_manager_test",
    srcs = ["logical_block_manager_test.cc"],
    deps = [
        ":logical_block_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_manager_cc_lib",
    srcs = ["kv_cache_manager.cc"],
    hdrs = ["kv_cache_manager.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_block_manager",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "kv_cache_manager_lib",
    srcs = [
        "kv_cache_manager.cc",
    ],
    hdrs = [
        "kv_cache_manager.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":logical_block_manager",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

nanobind_extension(
    name = "kv_cache_manager",
    srcs = [
        "kv_cache_manager_pybind.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_lib",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/tsl/platform:errors",
    ],
)

nanobind_extension(
    name = "kv_cache_store",
    srcs = [
        "kv_cache_store.cc",
        "kv_cache_store.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_lib",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
    ],
)

py_test(
    name = "kv_cache_manager_test_gl",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager",
        "//learning/brain/research/jax:tpu_support",
        "//raiden_lib/raw_transfer/jax:raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_test(
    name = "kv_cache_manager_test_gf",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
)

cc_library(
    name = "kv_cache_manager_ffi",
    srcs = [
        "kv_cache_manager_ffi.cc",
    ],
    hdrs = [
        "kv_cache_manager_ffi.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_cc_lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/ffi",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:device_address",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/stream_executor/tpu:tpu_executor",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_kv_cache_manager_ffi",
    srcs = [
        "kv_cache_manager_ffi_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_ffi",
        "//kv_cache:kv_cache_manager_cc_lib",
        "@nanobind",
    ],
)

py_library(
    name = "kv_cache_manager_ffi_py",
    srcs = ["kv_cache_manager_ffi.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager_ffi",
        "@jax//jax",
        "@jax//jax/experimental:compute_on",
        "@numpy//numpy",
    ],
)
