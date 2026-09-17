import importlib
import os.path
import re
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import torch

from octorch import EmulatorState
from octorch.env import OctorchEnv


class EnvDef(ModuleType):
    rom_file: str
    score_fn: Callable[[EmulatorState], torch.Tensor]
    terminated_fn: Callable[[EmulatorState], torch.Tensor]
    action_set: list
    metadata: dict
    startup_instructions: int


ENV_IDS = [
    "airplane", "blinky", "brix", "deep", "filter", "flight_runner", "missile", "pong",
    "rocket", "shooting_stars", "spacejam", "squash", "submarine", "tank", "tetris", "ufo",
    "vertical_brix", "wipe_off", "worm",
    "cavern1", "cavern2", "cavern3", "cavern4a", "cavern4b", "cavern5", "cavern6",
    "space_flight1", "space_flight2", "space_flight3", "space_flight4", "space_flight5",
    "space_flight6", "space_flight7", "space_flight8", "space_flight9", "space_flight10",
    "target_shooter1", "target_shooter2", "target_shooter3",
]


def get_rom_path(rom_filename: str) -> str:
    """Locate ``rom_filename`` in the nearest ``roms`` directory above this package."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        roms_path = current / "roms" / rom_filename
        if roms_path.exists():
            return str(roms_path)
        current = current.parent
    raise FileNotFoundError(
        f"ROM '{rom_filename}' not found. Make sure 'roms' directory exists at repository root."
    )


def create_environment(
    env_id: str,
    render_mode: Optional[str] = None,
    render_scale: int = 8,
    color_scheme: str = "classic",
    **kwargs,
):
    """Create a game environment by id (e.g. ``"brix"``, ``"cavern3"``, ``"space_flight10"``).

    Returns:
        ``(env, metadata)``: an :class:`octorch.env.OctorchEnv` and the game's metadata dict.
    """
    env_id = env_id.replace("-", "_")
    match = re.match(r"^(.*?)(\d+[ab]?)$", env_id)
    if match and match.group(1) in ("cavern", "space_flight", "target_shooter"):
        rom_file = env_id + ".ch8"
        env_id = match.group(1)
        have_level = True
    else:
        have_level = False
    module: EnvDef = importlib.import_module(f"octorch.environments.{env_id}")
    if have_level:
        module.__setattr__("rom_file", rom_file)

    return (
        OctorchEnv(
            rom_path=os.path.join(get_rom_path(module.rom_file)),
            score_fn=getattr(module, "score_fn", lambda _: 0),
            terminated_fn=getattr(module, "terminated_fn", lambda _: False),
            action_set=getattr(module, "action_set", None),
            startup_instructions=getattr(module, "startup_instructions", 0),
            custom_startup=getattr(module, "custom_startup", None),
            disable_delay=getattr(module, "disable_delay", False),
            render_mode=render_mode,
            render_scale=render_scale,
            color_scheme=color_scheme,
            **kwargs,
        ),
        module.metadata,
    )


def print_metadata(program: dict):
    """Print CHIP-8 program metadata with useful information only."""
    title = program.get("title", "Unknown Program")
    release = program.get("release", "Unknown")
    print(f"🎮 {title} ({release})")

    if program.get("authors"):
        print(f"   By: {', '.join(program['authors'])}")

    if program.get("description"):
        desc = program["description"].strip()
        if desc:
            print(f"   {desc}")

    if program.get("roms"):
        for rom_hash, rom_info in program["roms"].items():
            print(f"   📁 {rom_info.get('file', 'Unknown ROM')}")
            if rom_info.get("platforms"):
                print(f"      Platforms: {' → '.join(rom_info['platforms'])}")
            if rom_info.get("keys"):
                keys = rom_info["keys"]
                controls = []
                if any(k in keys for k in ["up", "down", "left", "right"]):
                    dirs = []
                    if "up" in keys: dirs.append(f"↑{keys['up']}")
                    if "down" in keys: dirs.append(f"↓{keys['down']}")
                    if "left" in keys: dirs.append(f"←{keys['left']}")
                    if "right" in keys: dirs.append(f"→{keys['right']}")
                    controls.append(" ".join(dirs))
                if any(k in keys for k in ["a", "b"]):
                    actions = []
                    if "a" in keys: actions.append(f"A={keys['a']}")
                    if "b" in keys: actions.append(f"B={keys['b']}")
                    controls.append(" ".join(actions))
                if controls:
                    print(f"      Controls: {' | '.join(controls)}")
            if rom_info.get("tickrate"):
                print(f"      Speed: {rom_info['tickrate']} cycles/frame")
            if rom_info.get("screenRotation", 0) != 0:
                print(f"      Rotation: {rom_info['screenRotation']}°")
    print()
