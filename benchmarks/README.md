# Benchmarks and cross-validation scripts

- `octorch_throughput.py` – compiled Octorch step, several games × batch sizes → `results/scaling.json`
- `octax_jax_throughput.py` – the same for the original JAX Octax (run inside an environment with `jax[cuda12]` and `octax`) → `results/scaling_jax.json`
- `plot_octax_vs_octorch.py` – produces `imgs/figure_octax_vs_octorch.png` and the summary table
- `octax_dump_traces.py <out_dir> <env_id ...>` – dumps 128-step reference traces from JAX Octax (patched to Octorch's PRNG)
- `octax_identity_table.py <trace_dir> <device>` – replays the traces in Octorch and writes `table.md` → `results/identity_table_128_steps.md`

Numbers in `results/` were measured on one NVIDIA A100-80GB (driver 535, CUDA 12.6) with PyTorch 2.14.0+cu126 and JAX 0.6.2.
