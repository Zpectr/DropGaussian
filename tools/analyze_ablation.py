"""Aggregate the 8-scene ablation: metrics + Gaussian count, paired comparison.

Usage (inside container): python tools/analyze_ablation.py --root output/ablation8
                          (original may come from output/baseline_llff3)
"""
import os, re, glob, argparse, csv

SCENES = ["fern","flower","fortress","horns","leaves","orchids","room","trex"]
CONFIGS = ["none","original","h2","h1","h1h2"]

def read_metrics(path):
    if not os.path.exists(path): return None
    vals = {}
    for line in open(path):
        m = re.match(r"\s*(PSNR|SSIM|LPIPS)\s*:\s*([0-9.]+)", line)
        if m: vals[m.group(1)] = float(m.group(2))
    return vals if len(vals)==3 else None

def ply_count(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        for _ in range(40):
            line = f.readline().decode("latin1")
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.startswith("end_header"):
                break
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/ablation8")
    ap.add_argument("--orig_fallback", default="output/baseline_llff3")
    ap.add_argument("--iter", default="10000")
    ap.add_argument("--out", default="results/ablation_metrics.csv")
    args = ap.parse_args()

    rows = []
    data = {c: {} for c in CONFIGS}  # config -> scene -> metrics
    for cfg in CONFIGS:
        for s in SCENES:
            base = os.path.join(args.root, cfg, s)
            mp = os.path.join(base, f"metrics_{args.iter}.txt")
            v = read_metrics(mp)
            if v is None and cfg == "original":
                base = os.path.join(args.orig_fallback, s)
                v = read_metrics(os.path.join(base, f"metrics_{args.iter}.txt"))
            n = ply_count(os.path.join(base, "point_cloud", f"iteration_{args.iter}", "point_cloud.ply"))
            if v:
                data[cfg][s] = v
                rows.append([cfg, s, v["PSNR"], v["SSIM"], v["LPIPS"], n])

    # means
    print(f"{'config':10s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'#Gauss':>9s}  (n scenes)")
    means = {}
    for cfg in CONFIGS:
        sc = data[cfg]
        if not sc: continue
        mp = sum(v["PSNR"] for v in sc.values())/len(sc)
        ms = sum(v["SSIM"] for v in sc.values())/len(sc)
        ml = sum(v["LPIPS"] for v in sc.values())/len(sc)
        means[cfg] = (mp, ms, ml)
        print(f"{cfg:10s} {mp:7.2f} {ms:7.3f} {ml:7.3f} {'':>9s}  (n={len(sc)})")

    # paired comparison vs original
    if "original" in means:
        print("\nPaired delta vs original (per scene PSNR):")
        for cfg in ["h2","h1","h1h2"]:
            if not data[cfg]: continue
            deltas = []
            for s in SCENES:
                if s in data[cfg] and s in data["original"]:
                    d = data[cfg][s]["PSNR"] - data["original"][s]["PSNR"]
                    deltas.append(d)
            if deltas:
                pos = sum(1 for d in deltas if d>0)
                mean_d = sum(deltas)/len(deltas)
                print(f"  {cfg:6s}: mean ΔPSNR={mean_d:+.3f} dB, wins {pos}/{len(deltas)} scenes, "
                      f"per-scene=[{', '.join(f'{d:+.2f}' for d in deltas)}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config","scene","PSNR","SSIM","LPIPS","num_gaussians"])
        w.writerows(rows)
        for cfg,(mp,ms,ml) in means.items():
            w.writerow([cfg,"MEAN",f"{mp:.3f}",f"{ms:.3f}",f"{ml:.3f}",""])
    print("\nsaved", args.out)

if __name__ == "__main__":
    main()
