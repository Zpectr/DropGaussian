"""Quantitative metric figures from the real ablation data."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, re

ROOT = "output/ablation8"
SCENES = ["fern","flower","fortress","horns","leaves","orchids","room","trex"]
CFGS = ["none","original","h1","h1h2","h2"]
NAME = {"none":"3DGS","original":"DropGaussian","h1":"+H1","h1h2":"+H1+H2","h2":"+H2 (ours)"}
COL  = {"none":"#999999","original":"#5a7fb0","h1":"#7aa86f","h1h2":"#caa14a","h2":"#c0603a"}

def read(path):
    if not os.path.exists(path): return None
    v={}
    for ln in open(path):
        m=re.match(r"\s*(PSNR|SSIM|LPIPS)\s*:\s*([0-9.]+)",ln)
        if m: v[m.group(1)]=float(m.group(2))
    return v if len(v)==3 else None

data={c:{s:read(f"{ROOT}/{c}/{s}/metrics_10000.txt") for s in SCENES} for c in CFGS}
mean={c:{k:np.mean([data[c][s][k] for s in SCENES if data[c][s]]) for k in ["PSNR","SSIM","LPIPS"]} for c in CFGS}
os.makedirs("results/figs",exist_ok=True)

# Fig A: mean metrics bar (3 subplots)
fig,axs=plt.subplots(1,3,figsize=(11,3.4))
for ax,metric,up in zip(axs,["PSNR","SSIM","LPIPS"],[True,True,False]):
    vals=[mean[c][metric] for c in CFGS]
    bars=ax.bar([NAME[c] for c in CFGS], vals, color=[COL[c] for c in CFGS])
    ax.set_title(f"{metric} {'↑' if up else '↓'} (8-scene mean)", fontweight="bold")
    ax.set_xticklabels([NAME[c] for c in CFGS], rotation=25, ha="right", fontsize=8)
    lo,hi=min(vals),max(vals); pad=(hi-lo)*0.25+1e-6
    ax.set_ylim(lo-pad, hi+pad)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}" if metric!="PSNR" else f"{v:.2f}",
                                       ha="center", va="bottom", fontsize=8)
fig.suptitle("Ablation: 8-scene mean (all variants beat 3DGS; H2 best PSNR, H1/H1+H2 best LPIPS)", fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("results/figs/metrics_bar.png",dpi=150); plt.close(fig)

# Fig B: per-scene PSNR grouped bar (3DGS / DropGaussian / +H2)
sub=["none","original","h2"]; x=np.arange(len(SCENES)); w=0.26
fig,ax=plt.subplots(figsize=(9,3.6))
for i,c in enumerate(sub):
    ax.bar(x+(i-1)*w, [data[c][s]["PSNR"] for s in SCENES], w, label=NAME[c], color=COL[c])
ax.set_xticks(x); ax.set_xticklabels(SCENES, rotation=15)
ax.set_ylabel("PSNR (dB)"); ax.set_ylim(15,25)
ax.set_title("Per-scene PSNR: 3DGS vs DropGaussian vs +H2 (ours)", fontweight="bold")
ax.legend(ncol=3); fig.tight_layout(); fig.savefig("results/figs/perscene_psnr.png",dpi=150); plt.close(fig)

print("saved metrics_bar.png, perscene_psnr.png")
print("means:", {c:{k:round(mean[c][k],3) for k in mean[c]} for c in CFGS})
