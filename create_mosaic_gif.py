"""
Generate the mosaic GIF (20 games playing side by side with a random policy), like Octax's
imgs/octax_mosaic.gif but rendered with Octorch.
Usage: uv run python create_mosaic_gif.py [output.gif]
"""

import sys

import numpy as np
import torch
from PIL import Image

from octorch.environments import create_environment
from octorch.rendering import chip8_display_to_rgb, create_color_scheme

GAMES = [
    "airplane", "blinky", "brix", "cavern1", "deep",
    "filter", "flight_runner", "missile", "pong", "rocket",
    "space_flight1", "spacejam", "squash", "submarine", "tank",
    "tetris", "ufo", "vertical_brix", "wipe_off", "worm",
]
COLS, SCALE, PADDING, NUM_FRAMES = 5, 4, 8, 300
COLOR_SCHEME = "octax"


def main(output="imgs/octorch_mosaic.gif"):
    on_color, off_color = create_color_scheme(COLOR_SCHEME)
    envs = [create_environment(g)[0] for g in GAMES]
    states = [env.reset(seed)[0] for seed, env in enumerate(envs)]
    frames_per_game = [[] for _ in GAMES]
    generator = torch.Generator().manual_seed(0)
    for t in range(NUM_FRAMES):
        for i, env in enumerate(envs):
            action = int(torch.randint(0, env.num_actions, (), generator=generator))
            states[i], obs, reward, terminated, truncated, info = env.step(states[i], action)
            frames_per_game[i].append(chip8_display_to_rgb(states[i].display, SCALE, on_color, off_color))
            if terminated or truncated:
                states[i] = env.reset(1000 * i + t)[0]
        if t % 50 == 0:
            print(f"frame {t}/{NUM_FRAMES}", flush=True)

    rows = int(np.ceil(len(GAMES) / COLS))
    h, w = 32 * SCALE, 64 * SCALE
    grid_h, grid_w = rows * h + (rows + 1) * PADDING, COLS * w + (COLS + 1) * PADDING
    images = []
    for t in range(NUM_FRAMES):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        grid[:] = [c // 2 for c in off_color]
        for i in range(len(GAMES)):
            r, c = divmod(i, COLS)
            y, x = PADDING + r * (h + PADDING), PADDING + c * (w + PADDING)
            grid[y:y + h, x:x + w] = frames_per_game[i][t]
        images.append(Image.fromarray(grid).quantize(colors=8))
    images[0].save(output, save_all=True, append_images=images[1:], duration=30, loop=0, optimize=True)
    print(f"saved {output} ({NUM_FRAMES} frames, {grid_w}x{grid_h})")


if __name__ == "__main__":
    main(*sys.argv[1:])
