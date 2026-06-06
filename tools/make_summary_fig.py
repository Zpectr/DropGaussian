"""One summary figure of the full result table: PSNR/SSIM/LPIPS/#Gaussians."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, re

ROOT="output/ablation8"
SCENES=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
CFGS=["none","original","h1","h1h2","h2"]
NAME={"none":"3DGS","original":"DropGauss.","h1":"+H1","h1h2":"+H1+H2","h2":"+H2 (ours)"}
COL ={"none":"#9a9a9a","original":"#5a7fb0","h1":"#7aa86f","h1h2":"#caa14a","h2":"#c0603a"}

def read(p):
    if not os.path.exists(p): return None
    v={}
    for ln in open(p):
        m=re.match(r"\s*(PSNR|SSIM|LPIPS)\s*:\s*([0-9.]+)",ln)
        if m: v[m.group(1)]=float(m.group(2))
    return v if len(v)==3 else None

def ply_n(p):
    if not os.path.exists(p): return None
    with open(p,"rb") as f:
        for _ in range(40):
            ln=f.readline().decode("latin1")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
            if ln.startswith("end_header"): break
    return None

met={c:{k:np.mean([read(f"{ROOT}/{c}/{s}/metrics_10000.txt")[k] for s in SCENES]) for k in ["PSNR","SSIM","LPIPS"]} for c in CFGS}
gn ={c:np.mean([n for s in SCENES if (n:=ply_n(f"{ROOT}/{c}/{s}/point_cloud/iteration_10000/point_cloud.ply"))]) /1000 for c in CFGS}

panels=[("PSNR (dB) ↑",[met[c]["PSNR"] for c in CFGS],"%.2f",True),
        ("SSIM ↑",[met[c]["SSIM"] for c in CFGS],"%.3f",True),
        ("LPIPS ↓",[met[c]["LPIPS"] for c in CFGS],"%.3f",False),
        ("#Gaussians (k) ↓",[gn[c] for c in CFGS],"%.1f",False)]

fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,(title,vals,fmt,up) in zip(axs.ravel(),panels):
    bars=ax.bar([NAME[c] for c in CFGS],vals,color=[COL[c] for c in CFGS],edgecolor="black",linewidth=0.4)
    # highlight best
    best=max(range(len(vals)),key=lambda i:vals[i]) if up else min(range(len(vals)),key=lambda i:vals[i])
    bars[best].set_edgecolor("red"); bars[best].set_linewidth(2.2)
    ax.set_title(title,fontweight="bold",fontsize=13)
    ax.set_xticklabels([NAME[c] for c in CFGS],rotation=20,ha="right",fontsize=9)
    lo,hi=min(vals),max(vals); pad=(hi-lo)*0.3+1e-9; ax.set_ylim(lo-pad,hi+pad)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v,fmt%v,ha="center",va="bottom",fontsize=9)
fig.suptitle("DropGaussian ablation summary (LLFF 3-view, 8-scene mean)\n"
             "H2: best PSNR & most compact (-10.7% Gaussians); all variants beat 3DGS",
             fontweight="bold",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.93])
os.makedirs("results/figs",exist_ok=True)
fig.savefig("results/figs/summary.png",dpi=150); print("saved results/figs/summary.png")
print("gauss(k):",{c:round(gn[c],1) for c in CFGS})
