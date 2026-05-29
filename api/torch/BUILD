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
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_tpu_stablehlo_headers",
    deps = [
        "//third_party/stablehlo:chlo_ops",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/stablehlo/stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "//third_party/stablehlo/stablehlo/integrations/cpp/builder:chlo_builder",
        "//third_party/stablehlo/stablehlo/integrations/cpp/builder:func_builder",
        "//third_party/stablehlo/stablehlo/integrations/cpp/builder:mlir_builder",
        "//third_party/stablehlo/stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)

cc_library(
    name = "kv_cache_manager_torch",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "kv_cache_manager.cc",
        ],
    }),
    hdrs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "kv_cache_manager.h",
        ],
    }),
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":torch_tpu_stablehlo_headers",
            "//core:xla_raw_transfer_headers",
            "//kv_cache:kv_cache_manager_base",
            "//third_party/py/torch:aten_headers",
            "//third_party/py/torch:torch_headers",
            "//third_party/py/torch/c10/core:base_headers",
            "//third_party/py/torch/torch/headeronly:torch_headeronly",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@torch_tpu//eager:device_buffer",
            "@torch_tpu//eager:materialize",
            "@torch_tpu//eager:tensor_to_buffer",
            "@xla//xla/pjrt:pjrt_client",
        ],
    }),
)

nanobind_extension(
    name = "_kv_cache_manager",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "kv_cache_manager_module.cc",
        ],
    }),
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":kv_cache_manager_torch",
            "//core:raw_transfer_core",
            "//kv_cache:kv_cache_manager_base",
            "//third_party/py/torch:torch_headers",
            "//third_party/pybind11",
            "@com_google_absl//absl/status:statusor",
        ],
    }),
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": ["kv_cache_manager.py"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":_kv_cache_manager",
            "//third_party/py/torch:pytorch",
            "@torch_tpu",
        ],
    }),
)

py_test(
    name = "kv_cache_manager_test",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": ["kv_cache_manager_test.py"],
    }),
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":kv_cache_manager_torch_py",
            "//third_party/py/torch:pytorch",
            "@com_google_absl_py//absl/testing:absltest",
            "@com_google_absl_py//absl/testing:parameterized",
            "@pypi//numpy",
            "@torch_tpu",
        ],
    }),
)

cc_library(
    name = "weight_synchronizer_torch",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "weight_synchronizer.cc",
        ],
    }),
    hdrs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "weight_synchronizer.h",
        ],
    }),
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "//third_party/py/torch:aten_headers",
            "//third_party/py/torch:torch_headers",
            "//third_party/py/torch/c10/core:base_headers",
            "//third_party/py/torch/torch/headeronly:torch_headeronly",
            "//third_party/stablehlo:chlo_ops",
            "//third_party/stablehlo:stablehlo_ops",
            "//third_party/stablehlo/stablehlo/integrations/cpp/builder:attr_type_builder_util",
            "//third_party/stablehlo/stablehlo/integrations/cpp/builder:chlo_builder",
            "//third_party/stablehlo/stablehlo/integrations/cpp/builder:func_builder",
            "//third_party/stablehlo/stablehlo/integrations/cpp/builder:mlir_builder",
            "//third_party/stablehlo/stablehlo/integrations/cpp/builder:stablehlo_builder",
            "//weight_sync:weight_synchronizer_base",
            "@com_google_absl//absl/status",
            "@torch_tpu//eager:device_buffer",
            "@torch_tpu//eager:materialize",
            "@torch_tpu//eager:tensor_to_buffer",
            "@xla//xla/pjrt:pjrt_client",
        ],
    }),
)

nanobind_extension(
    name = "_weight_synchronizer",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            "weight_synchronizer_module.cc",
        ],
    }),
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":weight_synchronizer_torch",
            "//core:raw_transfer_core",
            "//third_party/py/torch:torch_headers",
            "//third_party/pybind11",
            "//weight_sync:weight_synchronizer_base",
            "@com_google_absl//absl/status",
        ],
    }),
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": ["weight_synchronizer.py"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":_weight_synchronizer",
            "//third_party/py/torch:pytorch",
            "@torch_tpu",
        ],
    }),
)

py_test(
    name = "weight_synchronizer_test",
    srcs = select({
        "//:torch_disabled": [],
        "//conditions:default": ["weight_synchronizer_test.py"],
    }),
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = select({
        "//:torch_disabled": [],
        "//conditions:default": [
            ":weight_synchronizer_torch_py",
            "//third_party/py/torch:pytorch",
            "@com_google_absl_py//absl/testing:absltest",
            "@com_google_absl_py//absl/testing:parameterized",
            "@pypi//numpy",
            "@torch_tpu",
        ],
    }),
)

nanobind_extension(
    name = "_raiden_transfer_engine",
    srcs = ["raiden_transfer_engine.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        "//core:raw_transfer_core",
        "//kv_cache:kv_cache_manager_base",
        "//third_party/py/torch:aten_headers",
        "//third_party/py/torch:torch_headers",
        "//third_party/pybind11",
        "//transport:socket_transport",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_library(
    name = "raiden_transfer_engine",
    srcs = ["raiden_transfer_engine.py"],
    data = [
        "raiden_transfer_engine.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_raiden_transfer_engine",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu:_loader",
        "@torch_tpu",
    ],
)

py_test(
    name = "raiden_transfer_engine_test",
    srcs = ["raiden_transfer_engine_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":raiden_transfer_engine",
        "//third_party/py/torch:pytorch",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu",
    ],
)
