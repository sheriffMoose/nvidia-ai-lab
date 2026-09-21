# GPU Environment — pinned

## Rental strategy

Provider(s): **RunPod**
Default tier: **L4 24GB @ ~$0.49/hr**
Burst tier (FP8): **A100 80GB @ ~$1.59/hr — Wk 4 D4, Wk 6 D4 only**
Free tiers in use: **build.nvidia.com**
Teardown rule: **stop if <24h, destroy if longer**

## Hardware (of the rented box)

GPU: **NVIDIA L4 24GB**
Compute capability: **8.9**
VRAM: **24 GB** GPU count: **1**

## Rebuild recipe

1. Rent L4 on RunPod (or Vast.ai)
2. SSH into the pod
3. Verify GPU is visible: `nvidia-smi`

## Spend log

| Date  | Tier    | Hours | $     | What it bought |
| ----- | ------- | ----- | ----- | -------------- |
| 09-21 | Default | 21    | $0.30 | Initial Setup  |

## Change log

| Date  | What changed   | Why | Benchmarks invalidated? |
| ----- | -------------- | --- | ----------------------- |
| 09-21 | initial update |     |                         |
