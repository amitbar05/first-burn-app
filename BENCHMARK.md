# Benchmark Results

## Hardware

| Component | Model |
|---|---|
| CPU | Intel Core i5-6500 @ 3.20GHz |
| GPU | NVIDIA GeForce GTX 1050 Ti (4GB) |
| CUDA | 12.2 |
| Driver | 580.95.05 |

## Results

Training 3-layer MLP on MNIST (60k train / 10k test), 20 epochs, release mode.

### CPU-only (NdArray backend, batch_size=32, 5 epochs)

| Metric | Value |
|---|---|
| Total wall time | 2m 26s |
| Training time | ~146s |
| Per epoch (steady-state) | ~29s |
| Peak memory | 480 MB |
| Final accuracy | 97.25% |

### GPU-accelerated (WGPU backend, batch_size=256, 20 epochs)

| Metric | Value |
|---|---|
| Total wall time | 12.49s |
| Training time | 12.49s |
| Per epoch (steady-state) | ~530ms |
| Peak memory | 583 MB |
| Final accuracy | 97.04% |

### Speedup (per epoch)

| Metric | Improvement |
|---|---|
| Per epoch (steady-state) | **58x faster** |
| 20 GPU epochs vs 5 CPU epochs | **4x more training in 1/12th the time** |
