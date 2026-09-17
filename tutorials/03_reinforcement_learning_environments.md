# Reinforcement Learning Environments

Octorch transforms classic CHIP-8 games into modern reinforcement learning environments. This tutorial explores how to use Octorch for training RL agents, understanding reward structures, and leveraging batched, compiled execution for large-scale experiments.

## From Games to RL Environments

When you play a CHIP-8 game manually, you see pixels on screen and press buttons based on what you observe. An RL environment formalizes this interaction into the standard observation-action-reward cycle. Octorch wraps the core emulator with additional tracking for scores, termination conditions, and standardized interfaces.

Let's start by understanding how a game becomes an RL environment:

```python
import torch
from octorch.environments import create_environment

# Create the Brix environment (Breakout clone)
env, metadata = create_environment("brix")

print(f"Game: {metadata['title']}")
print(f"Actions available: {env.num_actions}")
print(f"CHIP-8 keys behind the actions: {env.action_set.tolist()} (+ a no-op)")
print(f"Observation shape: {env.observation_shape}")
```

Each game module declares the important game elements:
- **Actions**: Which CHIP-8 keys are actually used (not all 16 keys are needed)
- **Score**: Which register tracks the player's score (`score_fn`)
- **Termination**: When the game ends (`terminated_fn`: lives = 0, game-over flag, ...)
- **Observations**: The 64x32 pixel display as the agent's "vision"

## The Environment Interface

Octorch environments follow a standard interface similar to Gymnasium, but every call can operate on a whole batch of environments:

```python
def explore_environment_interface():
    env, _ = create_environment("pong")

    # Reset gives you initial state, observation and info. An integer seed and no
    # batch_size gives a single, unbatched environment.
    state, observation, info = env.reset(42)

    print(f"Initial observation shape: {tuple(observation.shape)}")
    print(f"Observation is binary: {observation.dtype}")
    print(f"Info keys: {list(info.keys())}")

    # Step takes an action and returns the new state
    action = 1  # second key of the action set
    new_state, new_obs, reward, terminated, truncated, info = env.step(state, action)

    print("After one step:")
    print(f"  Reward: {float(reward)}")
    print(f"  Terminated: {bool(terminated)}")
    print(f"  Truncated: {bool(truncated)}")
    return new_state, new_obs

state, obs = explore_environment_interface()
```

The observation is always the raw CHIP-8 display—`frame_skip` consecutive 64x32 binary frames (4 by default). This gives agents complete visual information about the game state, just like a human player would see.

### Batched environments

Passing `batch_size` (or a tensor of PRNG states from `octorch.make_rng`) to `reset` creates a batch; `step` then expects one action per environment and returns batched observations, rewards and flags:

```python
env, _ = create_environment("pong")
state, obs, info = env.reset(0, batch_size=16)
print(tuple(obs.shape))                                   # (16, 4, 64, 32)

actions = torch.randint(0, env.num_actions, (16,), device=env.device)
state, obs, reward, terminated, truncated, info = env.step(state, actions)
print(tuple(reward.shape), tuple(terminated.shape))       # (16,) (16,)
```

`env.step` is functional (the input state is left untouched); `env.step_` updates the state in place and saves a copy per step. Finished episodes are restarted with `env.reset_where(state, mask)`, which resets only the environments where `mask` is true and keeps the batch shape.

## Understanding Rewards and Scoring

Different games have different reward structures. Let's examine how Octorch detects and uses scores:

```python
def analyze_game_scoring():
    """Compare scoring mechanisms across different games"""
    for game_name in ["pong", "brix", "tetris", "deep"]:
        env, metadata = create_environment(game_name)
        state, obs, info = env.reset(123)

        total_reward = 0.0
        for step in range(100):
            action = int(torch.randint(0, env.num_actions, ()))
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        print(f"{game_name.upper()}:")
        print(f"  Total reward after {step + 1} steps: {total_reward}")
        print(f"  Final score: {float(info['score'])}")
        print(f"  Game ended: {bool(terminated or truncated)}")
        print()

torch.manual_seed(123)
analyze_game_scoring()
```

Most Octorch environments use score-based rewards—the reward at each step is the change in score since the last step. This encourages agents to maximize their game score, which aligns with human objectives.

## Compiling the Environment Step

Eager execution launches hundreds of small kernels per CHIP-8 instruction, so it is launch-bound (tens of thousands of steps per second regardless of the batch size). On a GPU, `env.compile` traces the step with `torch.compile` and captures the whole thing into a CUDA graph:

```python
import time

env, _ = create_environment("brix")
num_envs = 2048
env.compile(num_envs)                     # once per batch size; granularity="frame" by default

state, obs, info = env.reset(0, batch_size=num_envs)
actions = torch.randint(0, env.num_actions, (num_envs,), device=env.device)

torch.cuda.synchronize(); start = time.time()
for _ in range(50):
    state, obs, reward, terminated, truncated, info = env.step_(state, actions)
torch.cuda.synchronize()
print(f"{50 * num_envs / (time.time() - start):,.0f} steps/s")
```

A few things to know:

- `granularity` controls what `torch.compile` sees as one graph: `"instruction"` (fastest compile), `"frame"` (11 instructions, the default) or `"step"` (44 instructions, best throughput, slowest compile). Inductor caches kernels on disk, so later processes compile much faster.
- The compiled step works on static buffers. `env.step` / `env.step_` copy your state through them transparently; `env.static_state(num_envs)` returns the buffers themselves so that a loop can hold its state there and avoid any copies (this is what `OctorchVectorEnv` does).
- The compiled step is checked bit-exact against eager execution in the test-suite.

## Training Your First RL Agent

Let's implement a simple policy gradient agent with PyTorch. This demonstrates how Octorch's batched design makes RL training natural:

```python
import torch
from torch import nn
from torch.distributions import Categorical
from octorch.environments import create_environment

class SimplePolicy(nn.Module):
    """Flatten the frame stack -> hidden layer -> action logits"""
    def __init__(self, observation_shape, num_actions):
        super().__init__()
        input_size = 1
        for s in observation_shape:
            input_size *= s
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_size, 128), nn.Tanh(), nn.Linear(128, num_actions))

    def forward(self, observation):
        return self.net(observation.to(torch.float32))

env, _ = create_environment("pong")
policy = SimplePolicy(env.observation_shape, env.num_actions).to(env.device)
print("Policy parameters:")
for name, p in policy.named_parameters():
    print(f"  {name}: shape {tuple(p.shape)}")
```

Now let's train this policy using a simple REINFORCE algorithm. All environments of the batch are collected at once and finished episodes are reset in place:

```python
def collect_trajectory(policy, env, num_envs, episode_length):
    """Collect `episode_length` steps from `num_envs` parallel environments"""
    state, obs, info = env.reset(0, batch_size=num_envs)
    observations, actions, rewards, dones = [], [], [], []
    for _ in range(episode_length):
        with torch.no_grad():
            action = Categorical(logits=policy(obs)).sample()
        next_state, next_obs, reward, terminated, truncated, info = env.step_(state, action)
        done = terminated | truncated
        observations.append(obs); actions.append(action); rewards.append(reward); dones.append(done)
        state, obs = env.reset_where(next_state, done), next_obs
    return torch.stack(observations), torch.stack(actions), torch.stack(rewards), torch.stack(dones)

def compute_returns(rewards, dones, gamma=0.99):
    """Discounted returns for REINFORCE, computed backwards over time"""
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running * (~dones[t])
        returns[t] = running
    return returns

def policy_gradient_update(policy, optimizer, observations, actions, rewards, dones):
    returns = compute_returns(rewards, dones)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    T, N = actions.shape
    log_probs = Categorical(logits=policy(observations.reshape(T * N, *observations.shape[2:]))).log_prob(actions.reshape(-1))
    loss = -(log_probs * returns.reshape(-1)).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
num_envs = 64
env.compile(num_envs) if env.device.type == "cuda" else None

for iteration in range(10):
    obs, act, rew, done = collect_trajectory(policy, env, num_envs, episode_length=200)
    loss = policy_gradient_update(policy, optimizer, obs, act, rew, done)
    print(f"Iteration {iteration}: mean reward per env = {rew.sum(0).mean():.2f}, loss = {loss:.4f}")

print("Training complete! The agent learned to play through policy gradients.")
```

## Vectorized Training with OctorchVectorEnv

`OctorchVectorEnv` packages the batched environment, automatic resets, the `(batch, frame_skip, 32, 64)` float observation layout expected by convolutional networks, and the compiled static buffers when available:

```python
import time
from octorch.environments import create_environment
from octorch.wrappers import OctorchVectorEnv

env, _ = create_environment("brix")
num_envs = 1024
if env.device.type == "cuda":
    env.compile(num_envs)

venv = OctorchVectorEnv(env, num_envs=num_envs, seed=0)
obs, info = venv.reset()

episode_returns = torch.zeros(num_envs, device=venv.device)
completed = []
start = time.time()
for _ in range(500):
    action = torch.randint(0, venv.num_actions, (num_envs,), device=venv.device)
    obs, reward, terminated, truncated, info = venv.step(action)
    episode_returns += reward
    done = info["done"]                       # environments that were auto-reset this step
    completed.append(episode_returns[done])
    episode_returns[done] = 0
elapsed = time.time() - start
completed = torch.cat(completed)

print(f"{num_envs} parallel environments, 500 steps:")
print(f"  Execution time: {elapsed:.2f}s ({500 * num_envs / elapsed:,.0f} steps/s)")
print(f"  Completed episodes: {len(completed)}")
print(f"  Mean return: {completed.mean():.2f}, std: {completed.std():.2f}")
```

`info["final_observation"]` holds the last observation of the finished episodes, as in Gymnasium.

## The Reference Agents

`octorch.agents` ports the Octax PPO and PQN agents (same convolutional torso, GAE / λ-return computations and hyper-parameters). They train directly on an `OctorchVectorEnv`:

```python
from octorch.agents import PPOOctorch
from octorch.environments import create_environment
from octorch.wrappers import OctorchVectorEnv

env, _ = create_environment("brix")
env.compile(256) if env.device.type == "cuda" else None
venv = OctorchVectorEnv(env, num_envs=256, seed=0)

algo = PPOOctorch(env=venv, total_timesteps=256 * 32 * 4, num_steps=32, num_epochs=4, num_minibatches=8,
                  eval_freq=256 * 32 * 2)
evaluations = algo.train()                  # list of (global_step, episode lengths, returns)
```

The `train.py` script wraps this with a command-line interface (or a YAML config):

```bash
uv run python train.py --env brix --agent PPO --num-envs 512 --total-timesteps 5000000 --compile-env
uv run python train.py --env tetris --agent PQN --num-seeds 3
uv run python train.py --config conf/config.yaml
```

Results are pickled under `results/<env>/` and summarised in `results/results.txt`.

## Environment Customization

Each game environment can be customized for specific research needs:

```python
def explore_environment_customization():
    """Demonstrate environment characteristics across games"""
    games_info = []
    for game in ["pong", "brix", "tetris", "missile", "deep"]:
        env, metadata = create_environment(game, max_num_steps_per_episodes=1000)
        state, obs, info = env.reset(42)

        episode_length, total_reward = 0, 0.0
        for _ in range(1000):
            action = int(torch.randint(0, env.num_actions, ()))
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            episode_length += 1
            total_reward += float(reward)
            if terminated or truncated:
                break

        games_info.append({"name": game, "actions": env.num_actions,
                           "episode_length": episode_length, "final_reward": total_reward})

    print("Game Environment Comparison:")
    print("-" * 60)
    for info in games_info:
        print(f"{info['name'].upper():<12} | Actions: {info['actions']} | "
              f"Episode: {info['episode_length']:4d} steps | Reward: {info['final_reward']:6.1f}")
    return games_info

torch.manual_seed(42)
game_comparison = explore_environment_customization()
```

`create_environment` forwards keyword arguments to `OctorchEnv`: `max_num_steps_per_episodes`, `frame_skip`, `instruction_frequency`, `fps`, `device`, `render_mode`, `render_scale`, `color_scheme`, `modern_mode` and `timer_wraparound` (Octax-compatible timer wrap, see tutorial 02). `env.from_minutes(2.5)` sets the episode length in minutes of gameplay.

## Integration with Popular RL Libraries

`make_gymnasium_env` exposes the batched environment as a standard `gymnasium.vector.VectorEnv` with numpy arrays (requires `uv sync --extra training`):

```python
import time
import numpy as np
from octorch.environments import create_environment
from octorch.wrappers import make_gymnasium_env

env, _ = create_environment("brix")
gym_env = make_gymnasium_env(env, num_envs=8, seed=42)

print("Gymnasium Environment:")
print(f"  Action space: {gym_env.action_space}")
print(f"  Observation space: {gym_env.observation_space}")

obs, info = gym_env.reset(seed=42)
step_count, total_reward = 0, 0.0
start_time = time.time()
while step_count < 200:
    action = gym_env.action_space.sample()
    obs, reward, terminated, truncated, info = gym_env.step(action)
    total_reward += float(np.sum(reward))
    step_count += 1
end_time = time.time()

print("\nGaming Session Results:")
print(f"  Steps: {step_count} x {gym_env.num_envs} envs")
print(f"  Total reward: {total_reward}")
print(f"  FPS: {step_count * gym_env.num_envs / (end_time - start_time):.1f}")
```

Because it is a *vector* environment, finished episodes are reset automatically and libraries that accept `VectorEnv` (CleanRL-style loops, Stable-Baselines3 with a thin adapter, torchrl's `GymWrapper`) can consume it directly.
