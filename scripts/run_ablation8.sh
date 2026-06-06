#!/bin/bash
# Full 8-scene ablation: none/original/H2/H1/H1+H2, 5 configs x 8 LLFF scenes.
# 7-way GPU parallelism in batches. One run per GPU at a time (fair timing).
cd /workspace
configs=(none original h2 h1 h1h2)
scenes=(fern flower fortress horns leaves orchids room trex)
gpus=(1 2 3 4 5 6 7); ng=${#gpus[@]}
mkdir -p output/ablation8

flags_for(){ case $1 in
  none)     echo "--drop_mode none";;
  original) echo "--drop_mode original";;
  h2)       echo "--drop_mode original --drop_aware_densify";;
  h1)       echo "--drop_mode opacity_aware";;
  h1h2)     echo "--drop_mode opacity_aware --drop_aware_densify";;
esac; }

idx=0
for cfg in ${configs[@]}; do
  for scene in ${scenes[@]}; do
    g=${gpus[$((idx % ng))]}; port=$((6400 + idx))
    m=output/ablation8/$cfg/$scene
    flags=$(flags_for $cfg)
    ( CUDA_VISIBLE_DEVICES=$g python train.py -s dataset/nerf_llff_data/$scene -m $m --eval -r 8 --n_views 3 --port $port $flags \
      && CUDA_VISIBLE_DEVICES=$g python render.py -m $m -r 8 ) > output/ablation8/${cfg}_${scene}.log 2>&1 &
    idx=$((idx+1))
    [ $((idx % ng)) -eq 0 ] && wait
  done
done
wait
echo "ABLATION8_DONE"
for cfg in ${configs[@]}; do
  echo "=== $cfg ==="
  python metric.py --path output/ablation8/$cfg 2>/dev/null
  cat output/ablation8/$cfg/metrics_mean.txt 2>/dev/null
done
echo "METRIC_DONE"
