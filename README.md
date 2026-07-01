# GPU-Accelerated Parallel Training of Deep Neural Networks for Medical Image Classification

Benchmarking study comparing five progressively-optimised training configurations for a ResNet-18 brain tumor MRI classifier: CPU baseline, standard GPU, GPU + Automatic Mixed Precision (AMP), GPU + AMP + parallel data loading, and simulated 2-process distributed training with PyTorch DDP.

**Author:** Saim Ali Abbasi

## Overview

Training deep CNNs on medical images is compute-heavy, and while GPU acceleration is well known, the *individual* contribution of mixed precision, parallel data loading, and distributed training is rarely isolated and measured on consumer-grade hardware. This project benchmarks each optimisation independently and in combination, on a single NVIDIA GTX 1660 Super, using ResNet-18 on the Kaggle Brain Tumor MRI dataset (~7,000 images, 4 classes: glioma, meningioma, pituitary, no tumor).

Each configuration was run for 20 epochs, repeated 3 times, under identical conditions.

## Results

| Config | Device | Total Time (s) | Throughput (img/s) | Peak GPU Mem (MB) | Test Acc (%) | Speedup |
|---|---|---|---|---|---|---|
| 1 — CPU Baseline | CPU | 8,244 | 13.6 | 0.0 | 95.31 | 1.0x |
| 2 — Standard GPU | CUDA | 1,088 | 114.8 | 896.9 | 95.54 | 7.6x |
| 3 — GPU + AMP | CUDA | 2,161 | 51.8 | 577.4 | 95.17 | 3.8x |
| 4 — GPU + AMP + Parallel DL | CUDA | 1,995 | 56.1 | 577.4 | 95.31 | 4.1x |
| 5 — DDP (2 processes) | CUDA | 1,004 | 111.5 | 940.4 | 95.48 | 8.2x |

*Mean of 3 runs, batch size 32, 20 epochs. Std dev in training time stayed under 1% across all configs; test accuracy varied by at most ±0.13%.*

**Key finding — the AMP anomaly:** AMP *slowed down* training by ~2x on this hardware instead of speeding it up. The GTX 1660 Super (Turing TU116 die) lacks full Tensor Core support, so float16 autocast overhead outweighs any compute benefit — despite this, AMP still cut peak GPU memory by 35.6% (896.9 MB → 577.4 MB), useful when VRAM is the bottleneck rather than time.

**DDP scaling:** best overall speedup (8.2x vs. CPU) but only 1.08x over single-GPU training (48.5% scaling efficiency), since both simulated processes share one physical GPU.

## Repository Structure

````
.
├── train.py                # Training loop for Configs 1-4 (CPU, GPU, AMP, AMP+parallel DL)
├── ddp_train.py             # Config 5 — PyTorch DDP training (torchrun, 2 simulated processes)
├── analyze_results.py         # Aggregates and analyzes benchmark results
├── average_runs.py             # Averages metrics across repeated runs
├── results.json                # Raw results — Configs 1-4
├── results_ddp.json             # Raw results — Config 5 (DDP)
├── results_summary.csv           # Summary table across all configs
├── runs_breakdown.json            # Per-run breakdown (3 runs per config)
├── figures/                        # Benchmark plots (speedup, loss curves, batch-size sweep, etc.)
├── requirements.txt
└── README.md
````

## Setup

````bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

## Usage

````bash
# Configs 1-4 (CPU baseline, GPU, GPU+AMP, GPU+AMP+parallel DL)
python train.py --config cpu
python train.py --config gpu
python train.py --config amp
python train.py --config amp_parallel --num-workers 2 --batch-size 32

# Config 5 — distributed training (2 simulated processes on one GPU)
torchrun --nproc_per_node=2 ddp_train.py

# Aggregate + analyze
python average_runs.py
python analyze_results.py
````

## Methodology

- **Model:** ResNet-18, ImageNet-pretrained, final FC layer replaced with 4-class head, fully fine-tuned
- **Dataset:** Kaggle Brain Tumor MRI Dataset (~7,000 images, 4 classes), resized to 224×224, ImageNet normalization, random horizontal flip + up to 10° rotation (train only)
- **Optimizer:** Adam, lr=0.001, weight decay=1e-4, cross-entropy loss, cosine annealing LR schedule
- **Training:** 20 epochs, fixed seed (42), 3 repeated runs per configuration, batch size 32 (except the Config 4 batch-size sweep at 16/32/64/128)
- **DDP:** 2 simulated processes on a single physical GPU, Gloo backend, `DistributedSampler`
- **Hardware:** Intel Core i7 (8th gen), 16GB DDR4 RAM, NVIDIA GTX 1660 Super (6GB VRAM)
- **Software:** PyTorch 2.1, torchvision 0.16, Python 3.11, CUDA 12.4, Windows 11

## Metrics Tracked

- Total training time (s) and average time/epoch
- Throughput (images/sec)
- Peak GPU memory (`torch.cuda.max_memory_allocated()`)
- Final test accuracy
- Speedup S = T_CPU / T_config, and DDP scaling efficiency E = (speedup / N) × 100%

## Limitations

- DDP is simulated with 2 processes on one physical GPU, not true multi-GPU parallelism — results reflect process-sharing overhead rather than genuine hardware scaling.
- AMP results are specific to Turing-architecture cards without full Tensor Core support (e.g. GTX 1660 Super); results will differ on RTX-class or newer GPUs.
- Single dataset, single architecture (ResNet-18) — findings may not generalize to larger models or datasets.
