# Raiden Microbenchmarks

## Overview
This folder contains microbenchmarks designed to test the raw DMA bandwidth (Host-to-Device and Device-to-Host) of the Raiden engine. These benchmarks measure the raw hardware throughput, isolating the performance from Python or other framework-level overhead.

## How to Run
**Prerequisite:** Ensure you have already installed the package in your environment before running these scripts.

To execute the microbenchmark, navigate to this directory (`examples/microbenchmarks/`) and run:

```bash
PYTHONPATH=../.. python jax_dma_kv_cache_benchmark.py --telemetry_log_path=/tmp/${USER}_benchmark.jsonl
```

*Note: Setting PYTHONPATH=../.. points the Python interpreter back to the repository root so it can discover the tpu_raiden package, and the `--telemetry_log_path` flag prevents permission errors when writing output on shared VMs.*

## How to Read the Output
The microbenchmark will run through a suite of test cases (varying data types like BF16, FP32, INT32, and different tensor shapes).

For each test case, the standard output will display a performance comparison between three implementations:

1. **KVCacheManager**: The TPU Raiden raw DMA engine.
2. **JAX Pinned Host Baseline**: Native JAX transfers using pinned host memory.
3. **JAX Standard Baseline**: Native JAX transfers using standard unpinned NumPy arrays.

The script prints the median latency (in seconds) and the calculated throughput (in GB/s) for both **D2H (Device-to-Host)** and **H2D (Host-to-Device)** transfers.

When evaluating the performance, look specifically at the **`KVCacheManager D2H bandwidth`** and **`KVCacheManager H2D bandwidth`** lines and compare them against the JAX baselines to observe the throughput gains achieved by bypassing the framework overhead.
