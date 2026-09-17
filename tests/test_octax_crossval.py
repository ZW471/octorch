"""Bit-exact cross-validation against the original JAX Octax.

Requires ``uv sync --extra validate`` (jax + octax). Octax's ``CXNN`` handler is
patched to consume Octorch's xorshift32 stream (stored in ``rng[0]``) and its
start-up PRNG state is aligned, so random games are comparable as well.
"""

import numpy as np
import pytest
import torch

octax = pytest.importorskip("octax")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import octax.emulator  # noqa: E402
import octax.env  # noqa: E402
from octax import PROGRAM_START, create_state as jax_create_state  # noqa: E402
from octax.decode import DecodedInstruction  # noqa: E402
from octax.env import OctaxEnv, OctaxEnvState, asdict_non_recursive, run_n_instruction  # noqa: E402

from octorch.environments import create_environment, ENV_IDS  # noqa: E402


def _execute_random_xorshift(state, instruction: DecodedInstruction):
    s = state.rng[0].astype(jnp.uint32)
    s = s ^ (s << 13)
    s = s ^ (s >> 17)
    s = s ^ (s << 5)
    value = ((s ^ (s >> 16)) & 0xFF).astype(jnp.uint8)
    return state.replace(V=state.V.at[instruction.x].set(value & instruction.nn), rng=state.rng.at[0].set(s))


def _reset(self):
    rng = jnp.array([1, 0], dtype=jnp.uint32)  # octorch.create_state(0).rng == 1
    state = jax_create_state(rng)
    rom_array = jnp.array(list(self.rom_data), dtype=jnp.uint8)
    state = state.replace(memory=state.memory.at[PROGRAM_START:PROGRAM_START + len(self.rom_data)].set(rom_array))
    if self.custom_startup:
        state = self.custom_startup(state)
    elif self.startup_instructions > 0:
        state = run_n_instruction(state, self.startup_instructions)
    initial_score = self.score_fn(state) * 1.0
    return OctaxEnvState(**asdict_non_recursive(state), current_score=initial_score, previous_score=initial_score)


@pytest.fixture(scope="module", autouse=True)
def _patch_octax():
    import octax.environments
    from octorch.environments import get_rom_path
    octax.emulator.execute_random = _execute_random_xorshift
    OctaxEnv._reset = _reset
    octax.environments.get_rom_path = get_rom_path  # use this repository's roms/
    yield


JAX_ENV_IDS = [e for e in ENV_IDS if e not in ("cavern4a", "cavern4b")]  # octax cannot load these ids
NSTEPS = 60
SEED_STATE = 123456789


@pytest.mark.parametrize("env_id", JAX_ENV_IDS)
def test_env_matches_octax(env_id):
    from octax.environments import create_environment as jax_create_environment

    jenv, _ = jax_create_environment(env_id)
    jstate, jobs, jinfo = jenv.reset(jnp.array([SEED_STATE, 0], dtype=jnp.uint32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, _ = create_environment(env_id, device=device)
    assert env.num_actions == jenv.num_actions
    state, obs, info = env.reset(torch.full((2,), SEED_STATE, dtype=torch.int64))
    state.rng.fill_(SEED_STATE)

    assert np.array_equal(state.memory[0].cpu().numpy(), np.asarray(jstate.memory))
    assert np.array_equal(obs[0].cpu().numpy(), np.asarray(jobs))

    actions = np.random.RandomState(0).randint(0, env.num_actions, size=NSTEPS)
    for t, a in enumerate(actions):
        jstate, jobs, jreward, jterm, jtrunc, jinfo = jenv.step(jstate, int(a))
        state, obs, reward, term, trunc, info = env.step(state, torch.full((2,), int(a), device=device))
        assert np.array_equal(obs[1].cpu().numpy(), np.asarray(jobs)), f"{env_id} step {t}: obs"
        assert np.array_equal(state.V[1].cpu().numpy(), np.asarray(jstate.V)), f"{env_id} step {t}: V"
        assert int(state.pc[1]) == int(jstate.pc) and int(state.I[1]) == int(jstate.I), f"{env_id} step {t}: pc/I"
        assert int(state.rng[1]) == int(jstate.rng[0]), f"{env_id} step {t}: rng"
        assert float(reward[1]) == float(jreward), f"{env_id} step {t}: reward"
        assert bool(term[1]) == bool(jterm) and bool(trunc[1]) == bool(jtrunc), f"{env_id} step {t}: done"
        assert float(info["score"][1]) == float(jinfo["score"]), f"{env_id} step {t}: score"
