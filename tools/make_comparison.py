"""Build qualitative comparison figures (matplotlib, large clear labels).

Usage (inside container):
  python tools/make_comparison.py --scene fortress \
      --configs none original h2 --root output/ablation8 \
      --out results/figs/fortress_compare.png --views 0 1 2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, argparse
from PIL import Image

LABELS = {
    "none": "3DGS (no drop)",
    "original": "DropGaussian",
    "h1": "Ours H1",
    "h2": "Ours (H2)",
    "h1h2": "Ours (H1+H2)",
    "gt": "Ground Truth",
}

def load(path):
    return Image.open(path).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--configs", nargs="+", default=["none", "original", "h2"])
    ap.add_argument("--root", default="output/ablation8")
    ap.add_argument("--iteration", default="10000")
    ap.add_argument("--views", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = ["gt"] + args.configs
    nrows, ncols = len(args.views), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.1))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for r, v in enumerate(args.views):
        for c, cfg in enumerate(cols):
            base = os.path.join(args.root, args.configs[0] if cfg == "gt" else cfg,
                                args.scene, "test", f"ours_{args.iteration}")
            sub = "gt" if cfg == "gt" else "renders"
            im = load(os.path.join(base, sub, f"{v:05d}.png"))
            ax = axes[r, c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(LABELS.get(cfg, cfg), fontsize=15, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(f"view {v}", fontsize=13)

    fig.suptitle(f"LLFF 3-view: {args.scene}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)

if __name__ == "__main__":
    main()
