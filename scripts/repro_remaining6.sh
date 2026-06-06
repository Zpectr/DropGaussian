#!/bin/bash
# 并行复现指定 LLFF 场景 (每场景独立 GPU + 独立 port 避免 network_gui 端口冲突)
# 用法: bash repro_remaining6.sh  (默认复现剩余失败的5个场景)
cd /workspace
scenes=(flower horns leaves orchids trex)
gpus=(1 2 3 4 5)
for i in ${!scenes[@]}; do
  s=${scenes[$i]}; g=${gpus[$i]}; port=$((6030+i))
  (
    CUDA_VISIBLE_DEVICES=$g python train.py -s dataset/nerf_llff_data/$s -m output/baseline_llff3/$s --eval -r 8 --n_views 3 --port $port \
    && CUDA_VISIBLE_DEVICES=$g python render.py -m output/baseline_llff3/$s -r 8
  ) > output/repro_$s.log 2>&1 &
done
wait
echo "ALL_SCENES_DONE"
python metric.py --path output/baseline_llff3
echo "METRIC_DONE"
