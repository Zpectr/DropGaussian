#!/bin/bash
# 复现 DropGaussian 官方基线 —— LLFF 3-view, 2 个场景 (fern 室外 + room 室内)
# 论文报告 (8场景均值): PSNR 20.76 / SSIM 0.713 / LPIPS 0.200
# 在容器内运行:  bash scripts/reproduce_baseline.sh
set -e

exp_name='baseline_llff3'
scenes=("fern" "room")          # 2 个差异较大的真实场景
dataset_path='dataset/nerf_llff_data'
n_views=3
gpu=${1:-0}                       # 默认用 GPU 0, 可传参覆盖

export CUDA_VISIBLE_DEVICES=$gpu

for scene in "${scenes[@]}"; do
  echo "==================== [训练] $scene ===================="
  python train.py -s $dataset_path/$scene/ \
    -m output/$exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views

  echo "==================== [渲染+评估] $scene ===================="
  python render.py -m output/$exp_name/$scene -r 8
done

echo "==================== [汇总指标] ===================="
python metric.py --path output/$exp_name
cat output/$exp_name/metrics_mean.txt
