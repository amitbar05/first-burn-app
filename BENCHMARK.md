# Benchmark Results

## Hardware

| Component | Model |
|---|---|
| CPU | Intel Core i5-6500 @ 3.20GHz |
| GPU | NVIDIA GeForce GTX 1050 Ti (4GB) |
| CUDA | 12.2 |
| Driver | 580.95.05 |

## Results

Training 3-layer MLP on MNIST (60k train / 10k test), release mode.

### CPU-only (NdArray backend, batch_size=32, lr=0.001, 5 epochs)

| Metric | Value |
|---|---|
| Total wall time | 2m 26s |
| Training time | ~146s |
| Per epoch (steady-state) | ~29s |
| Peak memory | 480 MB |
| Final accuracy | 97.25% |

### GPU-accelerated (WGPU backend, batch_size=256, lr=0.008, 20 epochs)

| Metric | Value |
|---|---|
| Total wall time | 15.1s |
| Training time | 14.03s |
| Per epoch (steady-state) | ~650ms |
| Peak memory | 583 MB |
| Final accuracy | 97.60% |
| Peak accuracy | 97.60% (epochs 13, 20) |

### Speedup

| Metric | Improvement |
|---|---|
| Per epoch (steady-state) | **45x faster** |
| To reach 97.25% accuracy | CPU: 146s (5 ep) vs GPU: ~8s (4 ep) — **18x faster** |
| Best accuracy | GPU 97.60% > CPU 97.25% |
