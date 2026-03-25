# Benchmark Results

## Hardware

| Component | Model |
|---|---|
| CPU | Intel Core i5-6500 @ 3.20GHz |
| GPU | NVIDIA GeForce GTX 1050 Ti (4GB) |
| CUDA | 12.2 |
| Driver | 580.95.05 |

## Model

6-layer MLP: 784 → 4096 → 4096 → 2048 → 1024 → 256 → 10 (~30M parameters)

## Results (same config: batch=1024, lr=0.001, 10 epochs)

### CPU (NdArray backend)

| Metric | Value |
|---|---|
| Total time | 1458.4s (24.3 min) |
| Per epoch | ~145.8s |
| Final accuracy | 98.05% |
| Final loss | 0.0210 |
| Peak memory | 987 MB |

### GPU (WGPU backend)

| Metric | Value |
|---|---|
| Total time | 205.2s (3.4 min) |
| Per epoch (steady-state) | ~13.5s |
| First epoch (shader compile) | 83.7s |
| Final accuracy | 97.78% |
| Final loss | 0.0166 |
| Peak memory | 987 MB |

### Comparison

| Metric | Value |
|---|---|
| **Speedup (total)** | **7.1x** |
| **Speedup (steady-state per epoch)** | **10.8x** |
| Loss curves | Overlapping (identical training) |
| Accuracy difference | < 0.3% (stochastic variance) |
