"""CHIP-8 memory and register operations."""

import torch

from octorch import prng
from octorch.decode import DecodedInstruction
from octorch.ops import take, put
from octorch.state import EmulatorState


def execute_set(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """6XNN - Set VX = NN."""
    return state.replace(V=put(state.V, instruction.x, instruction.nn))


def execute_add(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """7XNN - Add NN to VX (wraps at 8 bits)."""
    vx = take(state.V, instruction.x).to(torch.int64)
    return state.replace(V=put(state.V, instruction.x, (vx + instruction.nn) & 0xFF))


def execute_set_index(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """ANNN - Set I = NNN."""
    return state.replace(I=instruction.nnn.to(torch.int32))


def execute_random(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """CXNN - Set VX = random & NN."""
    rng, random_value = prng.random_byte(state.rng)
    return state.replace(V=put(state.V, instruction.x, random_value & instruction.nn), rng=rng)
