import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "benchmarks/results/"
tor = json.load(open(S + "scaling.json")); jx = json.load(open(S + "scaling_jax.json"))
games = sorted(set(r["game"] for r in jx) & set(r["game"] for r in tor))
sizes = sorted(set(r["batch"] for r in jx) & set(r["batch"] for r in tor))
def mat(rows, key):
    return np.array([[next(r[key] for r in rows if r["game"] == g and r["batch"] == b) for b in sizes] for g in games], dtype=float)
t_ms, j_ms = mat(tor, "ms_per_step"), mat(jx, "ms_per_step")
t_sps, j_sps = mat(tor, "steps_per_s") / 1e6, mat(jx, "steps_per_s") / 1e6
e_rows = [r for r in tor if r["game"] == "brix" and r["eager_steps_per_s"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
for ax in axes:
    ax.set_facecolor("#fcfcfb"); ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.grid(True, color="#e5e4e0", lw=0.8); ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("parallel environments")
def band(ax, m, color, label):
    ax.fill_between(sizes, m.min(0), m.max(0), color=color, alpha=0.18, lw=0)
    ax.plot(sizes, m.mean(0), color=color, lw=2, marker="o", ms=5, label=label)
band(axes[0], t_ms, "#2a78d6", "Octorch (PyTorch, compiled)")
band(axes[0], j_ms, "#eb6834", "Octax (JAX, jit + vmap)")
axes[0].plot([r["batch"] for r in e_rows], [r["batch"] / r["eager_steps_per_s"] * 1e3 for r in e_rows], color="#52514e", lw=2, ls="--", marker="o", ms=5, label="Octorch (PyTorch, eager)")
axes[0].set_ylabel("wall time per environment step (ms)"); axes[0].set_title("Time cost of one batched step", loc="left", fontsize=11)
band(axes[1], t_sps, "#2a78d6", "Octorch (PyTorch, compiled)")
band(axes[1], j_sps, "#eb6834", "Octax (JAX, jit + vmap)")
axes[1].plot([r["batch"] for r in e_rows], [r["eager_steps_per_s"] / 1e6 for r in e_rows], color="#52514e", lw=2, ls="--", marker="o", ms=5, label="Octorch (PyTorch, eager)")
axes[1].set_ylabel("environment steps / second (millions)"); axes[1].set_title("Throughput", loc="left", fontsize=11)
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(f"Octax vs Octorch on one A100-80GB (mean over {len(games)} games, band = min–max; frame_skip 4, 44 CHIP-8 instructions per step)", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("imgs/figure_octax_vs_octorch.png")
print("| envs | Octax ms/step | Octorch ms/step | Octax M steps/s | Octorch M steps/s | speed-up |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
for i, b in enumerate(sizes):
    print(f"| {b} | {j_ms[:, i].mean():.2f} | {t_ms[:, i].mean():.2f} | {j_sps[:, i].mean():.3f} | {t_sps[:, i].mean():.3f} | {t_sps[:, i].mean() / j_sps[:, i].mean():.2f}x |")
print("compile s: jax", np.mean([r["compile_s"] for r in jx]), "torch(cached)", np.mean([r["compile_s"] for r in tor]))
