"""Dump reference traces from the original JAX octax, using an xorshift32 PRNG
for CXNN stored in rng[0] so that traces are bit-comparable with octorch."""
import sys, os, numpy as np, jax, jax.numpy as jnp
# run in an environment where `octax` is installed (e.g. `uv sync --extra validate`)
import octax.emulator, octax.instructions.memory
from octax.decode import DecodedInstruction

def execute_random_xorshift(state, instruction: DecodedInstruction):
    s = state.rng[0].astype(jnp.uint32)
    s = s ^ (s << 13); s = s ^ (s >> 17); s = s ^ (s << 5)
    value = ((s ^ (s >> 16)) & 0xFF).astype(jnp.uint8)
    new_rng = state.rng.at[0].set(s)
    return state.replace(V=state.V.at[instruction.x].set(value & instruction.nn), rng=new_rng)
octax.emulator.execute_random = execute_random_xorshift
octax.instructions.memory.execute_random = execute_random_xorshift

import octax.env
from octax.env import OctaxEnv, OctaxEnvState, asdict_non_recursive, run_n_instruction
from octax import create_state, PROGRAM_START

def _reset(self):
    rng = jnp.array([1, 0], dtype=jnp.uint32)   # octorch's create_state(0) PRNG state is 1
    state = create_state(rng)
    rom_array = jnp.array(list(self.rom_data), dtype=jnp.uint8)
    state = state.replace(memory=state.memory.at[PROGRAM_START:PROGRAM_START + len(self.rom_data)].set(rom_array))
    if self.custom_startup:
        state = self.custom_startup(state)
    elif self.startup_instructions > 0:
        state = run_n_instruction(state, self.startup_instructions)
    initial_score = self.score_fn(state) * 1.0
    return OctaxEnvState(**asdict_non_recursive(state), current_score=initial_score, previous_score=initial_score)
OctaxEnv._reset = _reset
from octax.environments import create_environment
out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
env_ids = sys.argv[2:]
NSTEPS = 128
for env_id in env_ids:
    env, meta = create_environment(env_id)
    seed_state = 123456789
    rng = jnp.array([seed_state, 0], dtype=jnp.uint32)
    state, obs, info = env.reset(rng)
    rs = np.random.RandomState(0)
    actions = rs.randint(0, env.num_actions, size=NSTEPS)
    rec = {k: [] for k in ["obs", "reward", "terminated", "truncated", "V", "pc", "I", "score", "rng"]}
    rec["reset_obs"] = np.asarray(obs); rec["reset_V"] = np.asarray(state.V); rec["reset_pc"] = np.asarray(state.pc); rec["reset_memory"] = np.asarray(state.memory)
    for a in actions:
        state, obs, reward, term, trunc, info = env.step(state, int(a))
        rec["obs"].append(np.asarray(obs)); rec["reward"].append(float(reward)); rec["terminated"].append(bool(term)); rec["truncated"].append(bool(trunc))
        rec["V"].append(np.asarray(state.V)); rec["pc"].append(int(state.pc)); rec["I"].append(int(state.I)); rec["score"].append(float(info["score"])); rec["rng"].append(int(state.rng[0]))
    np.savez_compressed(os.path.join(out_dir, env_id + ".npz"), actions=actions, num_actions=env.num_actions, **{k: np.asarray(v) for k, v in rec.items()})
    print(env_id, "done, total reward", sum(rec["reward"]), "terminated", any(rec["terminated"]), flush=True)
