# Octorch Quickstart Guide

Octorch is a PyTorch-based CHIP-8 emulator and reinforcement learning environment suite. This guide covers installation and basic usage.

## What is Octorch?

Octorch provides CHIP-8 games as reinforcement learning environments. The library implements a CHIP-8 emulator with PyTorch tensors, allowing parallel execution of many game instances on a GPU for training RL agents on classic games like Pong, Brix (Breakout), and Tetris. It is a port of [Octax](https://github.com/riiswa/octax) and reproduces its emulation bit-for-bit.

Where Octax relies on `jax.vmap` to batch environments, Octorch makes batching explicit: every tensor of the emulator state has a leading *batch shape*, and one `env.step` call advances every environment of the batch. On a GPU the step can additionally be compiled (`torch.compile` + CUDA graph) for large speed-ups.

## Installation

Octorch is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:ZW471/octorch.git
cd octorch
uv sync                      # torch (CUDA 12.6 wheels), numpy, tqdm
uv sync --extra gui          # optional: pygame, opencv, pillow, matplotlib
```

`pyproject.toml` pins the PyTorch `cu126` wheel index; switch it to `cu128` / `cu130` / `cpu` if your driver requires it, then run `uv sync` again. Check the GPU build with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Your First Octorch Environment

Let's start by running a simple CHIP-8 game. The most straightforward way to experience Octorch is through its environment interface, which wraps the low-level emulator in a familiar API:

```python
import torch
from octorch.environments import create_environment

# Create a Deep8 environment (a shooter game)
env, metadata = create_environment("deep")

print(f"Game: {metadata['title']}")
print(f"Actions: {env.num_actions}")
print(f"Device: {env.device}")
```

This creates an environment for the Deep8 game. The `metadata` tells you about the game, while `env` gives you the actual environment to interact with. The environment lives on CUDA when available (pass `device="cpu"` to `create_environment` otherwise).

Now let's run a simple random agent for one episode:

```python
def random_policy(observation):
    """A simple random policy - just press random buttons!"""
    return int(torch.randint(0, env.num_actions, ()))

def run_episode(seed):
    """Run a complete episode with random actions"""
    # Reset the environment to start fresh (unbatched: one environment)
    state, observation, info = env.reset(seed)

    total_reward = 0.0
    steps = 0
    for _ in range(1000):
        action = random_policy(observation)
        state, observation, reward, terminated, truncated, info = env.step(state, action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return state, total_reward, steps

torch.manual_seed(42)
final_state, total_reward, steps = run_episode(42)
print(f"Episode finished after {steps} steps with total reward: {total_reward}")
```

## Understanding What Just Happened

When you ran that code, several important things happened:

1. **Eager execution**: every `env.step` runs 44 CHIP-8 instructions (11 instructions per frame, 4 frames) as ordinary PyTorch tensor operations. Nothing is compiled, so the first call is as fast as the last one, but a single unbatched environment is slow (hundreds of small kernels per instruction). Octorch shines with large batches and, on GPU, with `env.compile`.

2. **Functional interface**: `env.step` returns a *new* state and leaves the one you passed in untouched, exactly like Octax. There is also an in-place variant, `env.step_`, that updates the state's tensors and avoids a copy.

3. **Batching ready**: the same `reset` / `step` calls accept a `batch_size`, and every tensor then carries a leading batch dimension.

## Trying Different Games

Octorch comes with many built-in games. Here are some favorites:

```python
# Classic Pong - simple and great for learning
pong_env, _ = create_environment("pong")

# Brix (Breakout) - more complex, great for RL training
brix_env, _ = create_environment("brix")

# Tetris - challenging and strategic
tetris_env, _ = create_environment("tetris")
```

Each game has different characteristics. Pong is simple with clear rewards, Brix requires spatial reasoning, and Tetris demands long-term planning. `octorch.environments.ENV_IDS` lists all 39 environment ids.

## Running Many Environments at Once

Pass `batch_size` to `reset` and a tensor of actions to `step`:

```python
num_envs = 256
state, obs, info = env.reset(0, batch_size=num_envs)
print(obs.shape)                       # (256, 4, 64, 32): frame_skip frames of the 64x32 display

total_reward = torch.zeros(num_envs, device=env.device)
for _ in range(200):
    action = torch.randint(0, env.num_actions, (num_envs,), device=env.device)
    state, obs, reward, terminated, truncated, info = env.step(state, action)
    total_reward += reward
    # Reset the environments whose episode ended; the batch shape stays the same
    state = env.reset_where(state, terminated | truncated)

print(f"Mean reward: {total_reward.mean():.2f} +/- {total_reward.std():.2f}")
```

Every environment of the batch owns an independent random stream derived from the seed.

## Visualizing Game Play

Want to see what's happening? Octorch includes rendering capabilities:

```python
from octorch.rendering import create_video

def collect_frames(seed, num_steps=500):
    """Run an episode and collect the last frame of every step"""
    state, obs, info = env.reset(seed)
    frames = []
    for _ in range(num_steps):
        action = random_policy(obs)
        state, obs, reward, terminated, truncated, info = env.step(state, action)
        frames.append(obs[-1])          # obs is (frame_skip, 64, 32); keep the newest frame
    return torch.stack(frames)

frames = collect_frames(123)
print(frames.shape)                     # (500, 64, 32)

# Save a video of the gameplay (needs opencv: uv sync --extra gui); display=True opens a window
create_video(frames, filename="deep.mp4", display=False)
```

`env.render(state)` returns a single RGB frame as a numpy array, and `octorch.rendering.batch_render` tiles a batch of displays into a grid.

## Using the Vectorised / Gymnasium Interface

For training loops, `OctorchVectorEnv` handles automatic resets and returns observations as `(batch, frame_skip, 32, 64)` float tensors (height before width, like the Octax Gymnax wrapper):

```python
from octorch.wrappers import OctorchVectorEnv

venv = OctorchVectorEnv(brix_env, num_envs=64, seed=0)
obs, info = venv.reset()
print(f"Observation batch: {tuple(obs.shape)}, actions: {venv.num_actions}")

for step in range(100):
    action = torch.randint(0, venv.num_actions, (64,), device=venv.device)
    obs, reward, terminated, truncated, info = venv.step(action)
    # finished episodes were already reset; info["done"] tells you which ones
```

If you need the standard `gymnasium.vector.VectorEnv` API with numpy arrays (Stable-Baselines3, RLlib, ...), use `octorch.wrappers.make_gymnasium_env(brix_env, num_envs=64)` (requires `uv sync --extra training`).

## Performance with torch.compile

On a GPU, compile the step once per batch size. Every later `env.step` on a state of that batch shape replays a CUDA graph:

```python
import time

env, _ = create_environment("brix")
num_envs = 4096

env.compile(num_envs)                 # torch.compile + CUDA graph; takes 10 s - 2 min the first time
state, obs, info = env.reset(0, batch_size=num_envs)
action = torch.randint(0, env.num_actions, (num_envs,), device=env.device)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    state, obs, reward, terminated, truncated, info = env.step_(state, action)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"{100 * num_envs / elapsed:,.0f} environment steps per second")
```

On an A100 this reaches roughly a million environment steps per second at 8192 environments. See tutorial 03 for how compilation interacts with training loops.
