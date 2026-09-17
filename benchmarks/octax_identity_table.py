"""Replay JAX Octax reference traces in Octorch and print a markdown identity table."""
import sys, os, glob, numpy as np, torch
from octorch.environments import create_environment
trace_dir, device = sys.argv[1], sys.argv[2]
B = 4
lines = ["| env id | title | steps | observations | registers V | PC / I | PRNG | reward | terminated / truncated | score |", "| --- | --- | ---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |"]
all_ok = True
for path in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
    env_id = os.path.basename(path)[:-4]; ref = np.load(path)
    env, meta = create_environment(env_id, device=device)
    state, obs, info = env.reset(torch.full((B,), 123456789, dtype=torch.int64)); state.rng.fill_(123456789)
    lane = 2
    ok = {k: True for k in ["obs", "V", "pcI", "rng", "reward", "done", "score"]}
    ok["obs"] &= np.array_equal(obs[lane].cpu().numpy(), ref["reset_obs"]); ok["V"] &= np.array_equal(state.V[lane].cpu().numpy(), ref["reset_V"])
    n = 0
    for t, a in enumerate(ref["actions"]):
        state, obs, reward, term, trunc, info = env.step(state, torch.full((B,), int(a), device=device))
        ok["obs"] &= np.array_equal(obs[lane].cpu().numpy(), ref["obs"][t]); ok["V"] &= np.array_equal(state.V[lane].cpu().numpy(), ref["V"][t])
        ok["pcI"] &= int(state.pc[lane]) == int(ref["pc"][t]) and int(state.I[lane]) == int(ref["I"][t]); ok["rng"] &= int(state.rng[lane]) == int(ref["rng"][t])
        ok["reward"] &= float(reward[lane]) == float(ref["reward"][t]); ok["done"] &= bool(term[lane]) == bool(ref["terminated"][t]) and bool(trunc[lane]) == bool(ref["truncated"][t])
        ok["score"] &= float(info["score"][lane]) == float(ref["score"][t]); n += 1
    all_ok &= all(ok.values())
    m = lambda k: "✅" if ok[k] else "❌"
    lines.append(f"| `{env_id}` | {meta['title']} | {n} | {m('obs')} | {m('V')} | {m('pcI')} | {m('rng')} | {m('reward')} | {m('done')} | {m('score')} |")
    print(lines[-1], flush=True)
open(os.path.join(trace_dir, "table.md"), "w").write("\n".join(lines) + "\n")
print("ALL OK" if all_ok else "FAILURES")
