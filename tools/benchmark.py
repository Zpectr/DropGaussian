"""Benchmark render FPS, peak memory and Gaussian count for a trained model.

Usage (inside container):
  python tools/benchmark.py -m output/ablation8/h2/fortress -r 8
"""
import torch, time, os, sys
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int)
    parser.add_argument("--reps", default=100, type=int)
    args = get_combined_args(parser)
    print("loading scene...", flush=True)

    dataset = model.extract(args); pipe = pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras()
        n_gauss = gaussians.get_xyz.shape[0]

        # warmup
        for _ in range(10):
            for c in cams: render(c, gaussians, pipe, bg)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        for _ in range(args.reps):
            for c in cams: render(c, gaussians, pipe, bg)
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_frames = args.reps * len(cams)
        fps = n_frames / dt
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"model={args.model_path}")
        print(f"  #Gaussians={n_gauss}  FPS={fps:.1f}  peak_mem={peak_mem:.1f}MB  (frames={n_frames})")
