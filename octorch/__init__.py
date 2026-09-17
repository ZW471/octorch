"""Octorch: CHIP-8 arcade RL environments in PyTorch (a port of Octax)."""

from octorch.state import EmulatorState, StackState, create_state, empty_state
from octorch.emulator import (
    execute, execute_, fetch, fetch_, load_rom, load_rom_bytes,
    run_instruction, run_instruction_, run_n_instruction,
)
from octorch.decode import DecodedInstruction, decode
from octorch.constants import *  # noqa: F401,F403
from octorch.prng import make_rng, split
from octorch.rendering import chip8_display_to_rgb, create_color_scheme, batch_render

__version__ = "0.1.0"

__all__ = [
    "EmulatorState",
    "StackState",
    "create_state",
    "empty_state",
    "fetch",
    "fetch_",
    "execute",
    "execute_",
    "load_rom",
    "load_rom_bytes",
    "run_instruction",
    "run_instruction_",
    "run_n_instruction",
    "DecodedInstruction",
    "decode",
    "make_rng",
    "split",
    "PROGRAM_START",
    "FONT_START",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "chip8_display_to_rgb",
    "create_color_scheme",
    "batch_render",
]
