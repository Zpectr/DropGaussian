# DropGaussian — Improvement (SLAM Final Project)

Fork of the official **DropGaussian** (CVPR 2025) with two methodological improvements
targeting its sparse-view overfitting behaviour. All changes live on the `improvement` branch.

> See the accompanying report (`report/`) for the full motivation, analysis and experiments.
> Section references below point to that report.

## 1. What was changed (vs. the report)

| ID | Name | Idea | Report § | Code |
|----|------|------|----------|------|
| **H1** | Opacity-aware adaptive dropout | Replace the uniform `nn.Dropout` on opacity with a per-Gaussian, variance-aware drop probability (low-opacity/redundant Gaussians dropped more often, structural Gaussians protected) + unbiased compensation. | §6.1 | [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py) |
| **H2** | Drop-aware densification | De-bias the screen-space gradient used for clone/split by the dropout opacity scaling, so densification no longer reacts to dropout-perturbed gradients. | §6.2 | [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py), [`scene/gaussian_model.py`](scene/gaussian_model.py), [`train.py`](train.py) |

Both are exposed as options (default = original DropGaussian behaviour, so the baseline is preserved):

```
--drop_mode {original,none,opacity_aware}   # none = 3DGS sparse baseline; opacity_aware = H1
--drop_aware_densify                         # enable H2
```

## 2. Reproduce (one-liners)

Training is LLFF 3-view, 10k iters, `-r 8`. Run inside the container (see §4).

```bash
# Original DropGaussian (baseline)
python train.py -s dataset/nerf_llff_data/room -m output/room_orig --eval -r 8 --n_views 3
python render.py -m output/room_orig -r 8

# Ours (H1 + H2)
python train.py -s dataset/nerf_llff_data/room -m output/room_ours --eval -r 8 --n_views 3 \
    --drop_mode opacity_aware --drop_aware_densify
python render.py -m output/room_ours -r 8

# 3DGS sparse reference (no dropout)
python train.py -s dataset/nerf_llff_data/room -m output/room_3dgs --eval -r 8 --n_views 3 --drop_mode none
```

Full ablation on fern+room: `bash scripts/run_ablation.sh "fern room"`.
8-scene baseline: `bash scripts/reproduce_baseline.sh` (+ `scripts/repro_remaining6.sh` for multi-GPU).

## 3. Results (LLFF 3-view)

_(filled from `results/` — see `results/ablation_metrics.csv`)_

8-scene mean (LLFF 3-view, same env/batch); $\Delta$ and wins are paired vs. original:

| Config | PSNR | SSIM | LPIPS | #Gauss | Δ / wins |
|--------|------|------|-------|--------|----------|
| 3DGS (none) | 19.85 | 0.682 | 0.212 | — | −0.53 |
| DropGaussian (orig) | 20.38 | 0.707 | 0.203 | 76.6k | — |
| + H1 | 20.33 | 0.708 | 0.199 | — | −0.05 / 3·8 |
| + H1 + H2 | 20.42 | 0.710 | 0.198 | — | +0.05 / 5·8 |
| **+ H2 (best)** | **20.60** | **0.709** | **0.202** | **68.4k (−10.7%)** | **+0.22 / 6·8** |

**Findings:** H2 (drop-aware densification) is the win — **+0.22 dB, wins 6/8 scenes,
and a deterministic ~10% reduction in Gaussian count on all 8 scenes** at equal-or-better
SSIM/LPIPS and negligible cost. H1 (opacity-aware dropout) **fails** (−0.05 dB): protecting
high-opacity Gaussians weakens DropGaussian's break-the-dominance mechanism. Noise floor
(4 repeats): PSNR std 0.07 (fern) / 0.25 (room).

Baseline reproduction matches the paper within 1.6% PSNR (8-scene mean 20.43 vs 20.76).

## 4. Environment

Reproducible Docker image (matches the paper's `environment.yaml`: PyTorch 2.5.1 / CUDA 12.1):

```bash
# Base image (Python 3.11, torch 2.5.1+cu121, nvcc 12.1)
docker run -d --name dropgaussian --gpus all -v $PWD:/workspace -w /workspace \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel sleep infinity

# CUDA extensions (RTX 3090 = sm_86)
docker exec dropgaussian bash -lc \
  'TORCH_CUDA_ARCH_LIST=8.6 pip install ./submodules/simple-knn ./submodules/diff-gaussian-rasterization'

# Python deps: plyfile matplotlib torchmetrics==1.2.0 opencv-python-headless scipy
# LPIPS(vgg) weights must be cached at /root/.cache/torch/hub/checkpoints/
#   vgg16-397923af.pth  (torchvision VGG16 IMAGENET1K_V1)
#   vgg.pth             (richzhang LPIPS v0.1)
```

- **Hardware used**: NVIDIA RTX 3090 (24 GB), driver 535 / CUDA 12.2.
- **Python** 3.11, **CUDA** 12.1, **PyTorch** 2.5.1.

## 5. Logs & data

- Training/eval logs: `output/**/*.log`, per-run metrics `output/**/metrics_10000.txt`.
- Aggregated metrics: `results/*.csv`. Qualitative figures: `results/figs/` (via `tools/make_comparison.py`).

## 6. Attribution

Built on the official [DropGaussian](https://arxiv.org/abs/2504.00773) release and
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) / [FSGS](https://github.com/VITA-Group/FSGS).
Only the files listed in §1 were modified for this work.
