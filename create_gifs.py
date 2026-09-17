"""
Generate one GIF per environment using a random policy.
Usage: uv run python create_gifs.py
Output: docs/_static/imgs/<env_name>.gif
"""

import os

import torch
from PIL import Image

from octorch.environments import create_environment, ENV_IDS
from octorch.rendering import chip8_display_to_rgb, create_color_scheme

os.makedirs("docs/_static/imgs", exist_ok=True)

SCALE = 4
NUM_STEPS = 200
COLOR_SCHEME = "octorch"

for env_id in ENV_IDS:
    env, meta = create_environment(env_id)
    state, obs, info = env.reset(42)
    on_color, off_color = create_color_scheme(COLOR_SCHEME)
    frames = []
    for step in range(NUM_STEPS):
        action = int(torch.randint(0, env.num_actions, ()))
        state, obs, reward, terminated, truncated, info = env.step(state, action)
        for frame in obs:
            frames.append(Image.fromarray(chip8_display_to_rgb(frame, SCALE, on_color, off_color)))
        if terminated or truncated:
            break
    out = f"docs/_static/imgs/{env_id}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1::2], duration=33, loop=0)
    print(f"{env_id:18s} {meta['title']:40s} -> {out} ({len(frames)} frames)")
