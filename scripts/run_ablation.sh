#!/bin/bash
# Ablation: none / original / H2 / H1 / H1+H2  on given scenes (LLFF 3-view).
# Each config runs on its own GPU + port. Per scene: 5 configs in parallel.
# Usage (inside container): bash scripts/run_ablation.sh "fern room"
cd /workspace
scenes=(${1:-fern room})
configs=(none original h2 h1 h1h2)
gpus=(1 3 5 6 7)
mkdir -p output/ablation

flags_for() {
  case $1 in
    none)     echo "--drop_mode none";;
    original) echo "--drop_mode original";;
    h2)       echo "--drop_mode original --drop_aware_densify";;
    h1)       echo "--drop_mode opacity_aware";;
    h1h2)     echo "--drop_mode opacity_aware --drop_aware_densify";;
  esac
}

for scene in "${scenes[@]}"; do
  for i in ${!configs[@]}; do
    cfg=${configs[$i]}; g=${gpus[$i]}; port=$((6200+i))
    m=output/ablation/${cfg}/${scene}
    flags=$(flags_for $cfg)
    (
      CUDA_VISIBLE_DEVICES=$g python train.py -s dataset/nerf_llff_data/$scene -m $m --eval -r 8 --n_views 3 --port $port $flags \
      && CUDA_VISIBLE_DEVICES=$g python render.py -m $m -r 8
    ) > output/ablation/${cfg}_${scene}.log 2>&1 &
  done
  wait
  echo "SCENE_${scene}_DONE"
done

echo "ABLATION_DONE"
for cfg in "${configs[@]}"; do
  echo "=== config: $cfg ==="
  python metric.py --path output/ablation/$cfg 2>/dev/null
  cat output/ablation/$cfg/metrics_mean.txt 2>/dev/null
done
echo "METRIC_DONE"
