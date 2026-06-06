#!/bin/bash
# Estimate run-to-run PSNR std (CUDA non-determinism) for the ORIGINAL config.
# 4 repeats x {fern, room}, different seeds via different ports/GPUs.
cd /workspace
mkdir -p output/noise
scenes=(fern room)
gpus=(1 2 3 4 5 6 7 1); i=0
for s in "${scenes[@]}"; do
  for r in 0 1 2 3; do
    g=${gpus[$i]}; port=$((6700+i))
    m=output/noise/${s}_r${r}
    ( CUDA_VISIBLE_DEVICES=$g python train.py -s dataset/nerf_llff_data/$s -m $m --eval -r 8 --n_views 3 --port $port \
      && CUDA_VISIBLE_DEVICES=$g python render.py -m $m -r 8 ) > output/noise/${s}_r${r}.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo "NOISE_DONE"
python - <<'PY'
import re, glob, statistics, os
for s in ["fern","room"]:
    ps=[]
    for r in range(4):
        f=f"output/noise/{s}_r{r}/metrics_10000.txt"
        if os.path.exists(f):
            t=open(f).read(); m=re.search(r"PSNR\s*:\s*([0-9.]+)",t)
            if m: ps.append(float(m.group(1)))
    if len(ps)>1:
        print(f"{s}: runs={['%.3f'%p for p in ps]} mean={statistics.mean(ps):.3f} std={statistics.pstdev(ps):.3f} range={max(ps)-min(ps):.3f}")
PY
