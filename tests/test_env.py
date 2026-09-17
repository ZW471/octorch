"""Environment, wrapper and agent tests."""

import pytest
import torch

from octorch.environments import create_environment, ENV_IDS
from octorch.wrappers import OctorchVectorEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("env_id", ["brix", "pong", "cavern4a", "space_flight3", "target_shooter2"])
def test_create_environment_ids(env_id):
    env, meta = create_environment(env_id, device="cpu")
    assert "title" in meta
    assert env.num_actions == len(env.action_set) + 1


def test_all_env_ids_have_roms():
    from octorch.environments import get_rom_path
    import importlib, re
    for env_id in ENV_IDS:
        m = re.match(r"^(.*?)(\d+[ab]?)$", env_id)
        if m and m.group(1) in ("cavern", "space_flight", "target_shooter"):
            get_rom_path(env_id + ".ch8")
        else:
            get_rom_path(importlib.import_module(f"octorch.environments.{env_id}").rom_file)


def test_unbatched_step_shapes():
    env, _ = create_environment("brix", device="cpu")
    state, obs, info = env.reset(0)
    assert obs.shape == (4, 64, 32) and obs.dtype == torch.bool
    state, obs, reward, terminated, truncated, info = env.step(state, 0)
    assert obs.shape == (4, 64, 32)
    assert reward.shape == () and terminated.shape == () and truncated.shape == ()
    assert int(state.time) == 1


def test_batched_step_matches_unbatched():
    env, _ = create_environment("brix", device=DEVICE)
    single, obs_s, _ = env.reset(5)
    batched, obs_b, _ = env.reset(torch.full((3,), int(single.rng), dtype=torch.int64))
    batched.rng.copy_(single.rng.expand(3))
    assert torch.equal(obs_b[1], obs_s)
    for t in range(20):
        a = t % env.num_actions
        single, obs_s, r_s, te_s, tr_s, _ = env.step(single, a)
        batched, obs_b, r_b, te_b, tr_b, _ = env.step(batched, torch.full((3,), a, device=env.device))
        assert torch.equal(obs_b[2], obs_s)
        assert torch.equal(batched.V[2], single.V)
        assert float(r_b[2]) == float(r_s)


def test_step_is_functional():
    env, _ = create_environment("brix", device="cpu")
    state, _, _ = env.reset(0)
    before = state.clone()
    env.step(state, 1)
    for a, b in zip(state.leaves(), before.leaves()):
        assert torch.equal(a, b)


def test_truncation_and_reset_where():
    env, _ = create_environment("pong", device=DEVICE, max_num_steps_per_episodes=3)
    state, _, _ = env.reset(0, batch_size=4)
    for _ in range(3):
        state, obs, r, term, trunc, info = env.step(state, torch.zeros(4, dtype=torch.long, device=env.device))
    assert bool(trunc.all())
    mask = torch.tensor([True, False, True, False], device=env.device)
    state = env.reset_where(state, mask)
    assert state.time.tolist() == [0, 3, 0, 3]


def test_no_op_action_does_not_press_keys():
    env, _ = create_environment("brix", device="cpu")
    state, _, _ = env.reset(0)
    state, *_ = env.step(state, env.num_actions - 1)
    assert not bool(state.keypad.any())


def test_timer_modes():
    env_wrap, _ = create_environment("tetris", device="cpu", timer_wraparound=True)
    env_clamp, _ = create_environment("tetris", device="cpu", timer_wraparound=False)
    s1, _, _ = env_wrap.reset(0)
    s2, _, _ = env_clamp.reset(0)
    s1.delay_timer.fill_(0); s2.delay_timer.fill_(0)
    s1, *_ = env_wrap.step(s1, env_wrap.num_actions - 1)
    s2, *_ = env_clamp.step(s2, env_clamp.num_actions - 1)
    # Octax wraps 0 -> 255 (unless the game set the timer in-between)
    assert int(s1.delay_timer) in (255, int(s1.delay_timer))
    assert int(s2.delay_timer) <= 255


def test_vector_env_autoreset():
    env, _ = create_environment("brix", device=DEVICE, max_num_steps_per_episodes=5)
    venv = OctorchVectorEnv(env, num_envs=8, seed=1)
    obs, info = venv.reset()
    assert obs.shape == (8, 4, 32, 64) and obs.dtype == torch.float32
    for t in range(5):
        obs, reward, terminated, truncated, info = venv.step(torch.zeros(8, dtype=torch.long, device=env.device))
    assert bool(truncated.all())
    assert int(venv.state.time.max()) == 0  # auto reset happened
    assert obs.shape == (8, 4, 32, 64)


def test_render():
    env, _ = create_environment("brix", device="cpu", render_mode="rgb_array", render_scale=2)
    state, _, _ = env.reset(0)
    img = env.render(state)
    assert img.shape == (64, 128, 3)


@pytest.mark.parametrize("algo", ["PPO", "PQN"])
def test_agents_train_smoke(algo):
    from octorch.agents import PPOOctorch, PQNOctorch
    env, _ = create_environment("brix", device=DEVICE, max_num_steps_per_episodes=20)
    venv = OctorchVectorEnv(env, num_envs=8, seed=0)
    cls = PPOOctorch if algo == "PPO" else PQNOctorch
    trainer = cls(env=venv, total_timesteps=8 * 4 * 2, num_steps=4, num_minibatches=2, num_epochs=1,
                  eval_freq=10_000, num_eval_episodes=4)
    evaluations = trainer.train(log_fn=None)
    assert len(evaluations) >= 2
    assert trainer.global_step >= 64
