"""Extra figures: train/test overfitting gap, and opacity distribution."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, re
from plyfile import PlyData

ROOT="output/ablation8"
SCENES=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
NAME={"none":"3DGS","original":"DropGaussian","h2":"+H2 (ours)"}
COL ={"none":"#9a9a9a","original":"#5a7fb0","h2":"#c0603a"}

def psnr(cfg,s,which):
    log=f"{ROOT}/{cfg}_{s}.log"
    if not os.path.exists(log): return None
    m=re.findall(rf"ITER 10000\] Evaluating {which}.*?PSNR ([0-9.]+)", open(log).read())
    return float(m[-1]) if m else None

# ---- Fig 1: train/test gap (mean over scenes) ----
cfgs=["none","original","h2"]
tr={c:np.mean([v for s in SCENES if (v:=psnr(c,s,"train"))]) for c in cfgs}
te={c:np.mean([v for s in SCENES if (v:=psnr(c,s,"test"))]) for c in cfgs}
gap={c:tr[c]-te[c] for c in cfgs}
x=np.arange(len(cfgs)); w=0.35
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.6))
a1.bar(x-w/2,[tr[c] for c in cfgs],w,label="train",color="#d0a0a0")
a1.bar(x+w/2,[te[c] for c in cfgs],w,label="test",color="#5a7fb0")
a1.set_xticks(x); a1.set_xticklabels([NAME[c] for c in cfgs]); a1.set_ylabel("PSNR (dB)")
a1.set_title("Train vs Test PSNR (8-scene mean)",fontweight="bold"); a1.legend()
for i,c in enumerate(cfgs):
    a1.text(i-w/2,tr[c],f"{tr[c]:.1f}",ha="center",va="bottom",fontsize=8)
    a1.text(i+w/2,te[c],f"{te[c]:.1f}",ha="center",va="bottom",fontsize=8)
bars=a2.bar([NAME[c] for c in cfgs],[gap[c] for c in cfgs],color=[COL[c] for c in cfgs],edgecolor="black",lw=0.4)
a2.set_ylabel("train - test PSNR gap (dB)")
a2.set_title("Overfitting gap (lower = less overfit)",fontweight="bold")
for b,c in zip(bars,cfgs): a2.text(b.get_x()+b.get_width()/2,gap[c],f"{gap[c]:.2f}",ha="center",va="bottom",fontsize=9)
fig.tight_layout(); fig.savefig("results/figs/gap.png",dpi=150); plt.close(fig)
print("gap:",{c:round(gap[c],2) for c in cfgs})

# ---- Fig 2: opacity distribution (original vs h2, pooled over scenes) ----
def opac(cfg):
    vals=[]
    for s in SCENES:
        p=f"{ROOT}/{cfg}/{s}/point_cloud/iteration_10000/point_cloud.ply"
        if os.path.exists(p):
            o=np.asarray(PlyData.read(p)['vertex']['opacity'])
            vals.append(1/(1+np.exp(-o)))  # sigmoid activation
    return np.concatenate(vals)
fig,ax=plt.subplots(figsize=(6,3.6))
for cfg in ["original","h2"]:
    o=opac(cfg)
    ax.hist(o,bins=50,range=(0,1),alpha=0.55,label=f"{NAME[cfg]} (N={len(o)//1000}k)",
            color=COL[cfg],density=True)
ax.set_xlabel("Gaussian opacity"); ax.set_ylabel("density")
ax.set_title("Opacity distribution (pooled 8 scenes)",fontweight="bold"); ax.legend()
fig.tight_layout(); fig.savefig("results/figs/opacity_hist.png",dpi=150); plt.close(fig)
print("saved gap.png, opacity_hist.png")
