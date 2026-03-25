# Benchmark Results

## Hardware

| Component | Model |
|---|---|
| CPU | Intel Core i5-6500 @ 3.20GHz |
| GPU | NVIDIA GeForce GTX 1050 Ti (4GB) |
| CUDA | 12.2 |
| Driver | 580.95.05 |

## Results

Training 3-layer MLP on MNIST (60k train / 10k test), 5 epochs, release mode.

### Before: CPU-only (NdArray backend, batch_size=32)

| Metric | Value |
|---|---|
| Total wall time | 2m 26s |
| Training time | ~146s |
| Per epoch (steady-state) | ~29s |
| Peak memory | 480 MB |
| Final accuracy | 97.25% |

### After: GPU-accelerated (WGPU backend, batch_size=256)

| Metric | Value |
|---|---|
| Total wall time | 5.1s |
| Training time | 3.98s |
| Per epoch (steady-state) | ~500ms |
| Peak memory | 583 MB |
| Final accuracy | 96.47% |

### Speedup

| Metric | Improvement |
|---|---|
| Total wall time | **29x faster** |
| Training time | **37x faster** |
| Per epoch (steady-state) | **58x faster** |
