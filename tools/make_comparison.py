"""Build qualitative comparison figures: GT vs configs, for several test views.

Usage (inside container):
  python tools/make_comparison.py --scene fern \
      --configs none original h1h2 --root output/ablation \
      --out results/figs/fern_compare.png --views 0 2 4
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = {
    "none": "3DGS (no drop)",
    "original": "DropGaussian",
    "h1": "Ours H1",
    "h2": "Ours H2",
    "h1h2": "Ours (H1+H2)",
    "gt": "Ground Truth",
}

def load(path):
    return Image.open(path).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--configs", nargs="+", default=["none", "original", "h1h2"])
    ap.add_argument("--root", default="output/ablation")
    ap.add_argument("--iteration", default="10000")
    ap.add_argument("--views", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # columns: GT + each config ; rows: views
    cols = ["gt"] + args.configs
    pad, lab_h = 4, 22
    rows_imgs = []
    cell_w = cell_h = None
    for v in args.views:
        row = []
        for c in cols:
            base = os.path.join(args.root, args.configs[0] if c == "gt" else c,
                                args.scene, "test", f"ours_{args.iteration}")
            sub = "gt" if c == "gt" else "renders"
            p = os.path.join(base, sub, f"{v:05d}.png")
            im = load(p)
            if cell_w is None:
                cell_w, cell_h = im.size
            row.append(im.resize((cell_w, cell_h)))
        rows_imgs.append(row)

    W = len(cols) * cell_w + (len(cols) + 1) * pad
    H = len(args.views) * (cell_h + lab_h) + (len(args.views) + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for r, row in enumerate(rows_imgs):
        y = pad + r * (cell_h + lab_h + pad)
        for c, im in enumerate(row):
            x = pad + c * (cell_w + pad)
            if r == 0:
                draw.text((x + 2, y), LABELS.get(cols[c], cols[c]), fill=(0, 0, 0), font=font)
            canvas.paste(im, (x, y + lab_h))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
