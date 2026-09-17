"""Random-policy rollout of a batch of environments and a video of the first one."""

import time

import torch

from octorch.environments import create_environment, print_metadata
from octorch.rendering import create_video

if __name__ == "__main__":
    env, metadata = create_environment("deep")
    print_metadata(metadata)

    num_envs = 1024
    num_steps = 1000
    state, observation, info = env.reset(0, batch_size=num_envs)
    generator = torch.Generator(device=env.device).manual_seed(0)

    frames = []
    total_reward = torch.zeros(num_envs, device=env.device)
    start = time.time()
    for _ in range(num_steps):
        action = torch.randint(0, env.num_actions, (num_envs,), device=env.device, generator=generator)
        state, observation, reward, terminated, truncated, info = env.step(state, action)
        total_reward += reward
        frames.append(observation[0, -1].clone())
        # Reset finished environments (keeps the batch shape constant)
        state = env.reset_where(state, terminated | truncated)
    torch.cuda.synchronize() if env.device.type == "cuda" else None
    elapsed = time.time() - start
    print(f"{num_envs * num_steps / elapsed:,.0f} env steps / second ({elapsed:.2f}s)")
    print(f"Mean return over {num_envs} envs: {total_reward.mean():.2f}")

    try:
        create_video(torch.stack(frames), filename="deep.mp4", display=False)
    except ImportError:
        print("Install opencv (uv sync --extra gui) to write the video")
