"""Generate analysis figures for the report (no GUI)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

os.makedirs("results/figs", exist_ok=True)
scenes = ["fern","flower","fortress","horns","leaves","orchids","room","trex"]

# --- Gaussian count: original vs H2 ---
g_orig = [74727,72400,58002,64963,158319,84151,41133,59439]
g_h2   = [67020,64658,52275,59177,135223,76439,37380,55261]
x = np.arange(len(scenes)); w=0.38
fig, ax = plt.subplots(figsize=(7,3))
ax.bar(x-w/2, np.array(g_orig)/1000, w, label="DropGaussian", color="#5a7fb0")
ax.bar(x+w/2, np.array(g_h2)/1000, w, label="+H2 (ours)", color="#c0603a")
ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=20); ax.set_ylabel("#Gaussians (k)")
ax.set_title("H2 yields a more compact model on all 8 scenes (~-10%)")
ax.legend(); fig.tight_layout(); fig.savefig("results/figs/gauss_bar.png", dpi=150); plt.close(fig)

# --- per-scene delta PSNR (H2 - original) ---
d = [-0.13,+0.33,+0.71,+0.36,+0.09,-0.13,+0.37,+0.18]
fig, ax = plt.subplots(figsize=(7,3))
colors = ["#c0603a" if v>0 else "#888888" for v in d]
ax.bar(scenes, d, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(0.072, color="green", ls="--", lw=0.8, label="fern noise std")
ax.axhline(-0.072, color="green", ls="--", lw=0.8)
ax.set_ylabel(r"$\Delta$PSNR vs orig (dB)"); ax.set_xticklabels(scenes, rotation=20)
ax.set_title("H2 wins 6/8 scenes (paired mean +0.22 dB)")
ax.legend(); fig.tight_layout(); fig.savefig("results/figs/delta_psnr.png", dpi=150); plt.close(fig)

# --- noise floor: 4 repeats of original ---
fern_runs=[22.847,22.783,22.973,22.815]; room_runs=[22.315,21.879,21.788,22.341]
fig, ax = plt.subplots(figsize=(5,3))
ax.scatter([1]*4, fern_runs, color="#5a7fb0", s=40, label=f"fern (std={np.std(fern_runs):.3f})")
ax.scatter([2]*4, room_runs, color="#c0603a", s=40, label=f"room (std={np.std(room_runs):.3f})")
for xx,runs in [(1,fern_runs),(2,room_runs)]:
    ax.hlines(np.mean(runs), xx-0.15, xx+0.15, color="k")
ax.set_xticks([1,2]); ax.set_xticklabels(["fern","room"]); ax.set_xlim(0.5,2.5)
ax.set_ylabel("PSNR (dB)"); ax.set_title("Run-to-run noise (4 identical runs)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("results/figs/noise.png", dpi=150); plt.close(fig)
print("saved gauss_bar.png, delta_psnr.png, noise.png")
