"""CHIP-8 display operations."""

import torch

from octorch.constants import SCREEN_WIDTH, SCREEN_HEIGHT, MEMORY_SIZE
from octorch.decode import DecodedInstruction
from octorch.ops import take, put, take_many
from octorch.state import EmulatorState

Tensor = torch.Tensor

_grids = {}


def coordinate_grids(device) -> tuple[Tensor, Tensor]:
    """Pre-computed ``(xx, yy)`` int64 coordinate grids of shape ``(64, 32)``."""
    key = str(device)
    if key not in _grids:
        xx, yy = torch.meshgrid(
            torch.arange(SCREEN_WIDTH, device=device),
            torch.arange(SCREEN_HEIGHT, device=device),
            indexing="ij",
        )
        _grids[key] = (xx, yy)
    return _grids[key]


def sprite_mask(state: EmulatorState, instruction: DecodedInstruction) -> Tensor:
    """Compute the ``(..., 64, 32)`` boolean sprite of a DXYN instruction (before XOR)."""
    xx, yy = coordinate_grids(state.pc.device)
    sprite_x = (take(state.V, instruction.x).to(torch.int64) % SCREEN_WIDTH)[..., None, None]
    sprite_y = (take(state.V, instruction.y).to(torch.int64) % SCREEN_HEIGHT)[..., None, None]
    n = instruction.n[..., None, None]

    in_screen = (xx >= sprite_x) & (xx < sprite_x + 8) & (yy >= sprite_y) & (yy < sprite_y + n)
    row_offset = yy - sprite_y
    col_offset = xx - sprite_x

    address = (state.I.to(torch.int64)[..., None, None] + row_offset).clamp(0, MEMORY_SIZE - 1)
    lead = address.shape[:-2]
    sprite_bytes = take_many(state.memory, address.reshape(lead + (-1,))).reshape(address.shape).to(torch.int64)
    shift = (7 - col_offset).clamp(0, 7)
    return (((sprite_bytes >> shift) & 1) != 0) & in_screen


def execute_display(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """DXYN - Draw sprite at (VX, VY) with height N."""
    sprite = sprite_mask(state, instruction)
    collision = (state.display & sprite).flatten(-2).any(-1)
    return state.replace(
        display=state.display ^ sprite,
        V=put(state.V, torch.full_like(instruction.x, 15), collision),
    )
