"""Demonstration of Octorch rendering capabilities."""

import torch

from octorch.environments import create_environment, print_metadata
from octorch.rendering import batch_render, create_color_scheme, chip8_display_to_rgb


def basic_rendering_demo():
    print("🎮 Basic Rendering Demo")
    env, metadata = create_environment("brix", render_mode="rgb_array", render_scale=8)
    print_metadata(metadata)

    state, obs, info = env.reset(42)
    for step in range(10):
        action = int(torch.randint(0, env.num_actions, ()))
        state, obs, reward, terminated, truncated, info = env.step(state, action)
        if terminated or truncated:
            state, obs, info = env.reset(step)

    rgb_frame = env.render(state)
    print(f"✅ Rendered frame shape: {rgb_frame.shape}")
    return rgb_frame


def color_scheme_demo():
    print("\n🎨 Color Scheme Demo")
    env, metadata = create_environment("brix", render_mode="rgb_array")
    state, obs, info = env.reset(123)
    for _ in range(50):
        state, obs, reward, terminated, truncated, info = env.step(state, int(torch.randint(0, env.num_actions, ())))
        if terminated or truncated:
            break
    frames = {}
    for scheme in ["octax", "octorch", "classic", "amber", "white", "blue", "retro"]:
        on, off = create_color_scheme(scheme)
        frames[scheme] = chip8_display_to_rgb(state.display, scale=4, on_color=on, off_color=off)
        print(f"  {scheme:8s}: {frames[scheme].shape}")
    return frames


def batch_rendering_demo():
    print("\n🧱 Batch Rendering Demo")
    env, _ = create_environment("brix")
    state, obs, info = env.reset(0, batch_size=16)
    for _ in range(100):
        action = torch.randint(0, env.num_actions, (16,), device=env.device)
        state, obs, reward, terminated, truncated, info = env.step(state, action)
    grid = batch_render(state.display, scale=2, color_scheme="octorch")
    print(f"✅ Grid image shape: {grid.shape}")
    return grid


if __name__ == "__main__":
    basic_rendering_demo()
    color_scheme_demo()
    grid = batch_rendering_demo()
    try:
        from PIL import Image
        Image.fromarray(grid).save("batch_render.png")
        print("Saved batch_render.png")
    except ImportError:
        pass
