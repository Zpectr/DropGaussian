#!/bin/bash
# Rerun only H1 and H1+H2 configs (after H1 v2 design change) on fern+room.
cd /workspace
rm -rf output/ablation/h1 output/ablation/h1h2
gpus=(1 3 5 6); i=0
for spec in "h1|--drop_mode opacity_aware" "h1h2|--drop_mode opacity_aware --drop_aware_densify"; do
  cfg=${spec%%|*}; flags=${spec#*|}
  for scene in fern room; do
    g=${gpus[$i]}; port=$((6300+i))
    m=output/ablation/$cfg/$scene
    ( CUDA_VISIBLE_DEVICES=$g python train.py -s dataset/nerf_llff_data/$scene -m $m --eval -r 8 --n_views 3 --port $port $flags \
      && CUDA_VISIBLE_DEVICES=$g python render.py -m $m -r 8 ) > output/ablation/${cfg}_${scene}_v2.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo "RERUN_DONE"
for cfg in h1 h1h2; do echo "=== $cfg ==="; python metric.py --path output/ablation/$cfg; cat output/ablation/$cfg/metrics_mean.txt; done
echo "METRIC_DONE"
