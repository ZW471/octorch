"""CHIP-8 instruction decoding."""

from __future__ import annotations

import dataclasses
from typing import Union

import torch

from octorch.struct import Struct

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class DecodedInstruction(Struct):
    """Decoded CHIP-8 instruction with extracted operands (all int64 tensors)."""
    raw: Tensor
    opcode: Tensor  # First nibble
    x: Tensor       # Second nibble (VX register)
    y: Tensor       # Third nibble (VY register)
    n: Tensor       # Fourth nibble (4-bit immediate)
    nn: Tensor      # Last byte (8-bit immediate)
    nnn: Tensor     # Last 12 bits (12-bit address)


def decode(instruction: Union[int, Tensor], device=None) -> DecodedInstruction:
    """Decode 16-bit instruction(s) into components."""
    if not isinstance(instruction, torch.Tensor):
        instruction = torch.tensor(int(instruction), dtype=torch.int64, device=device)
    instruction = instruction.to(torch.int64) & 0xFFFF
    return DecodedInstruction(
        raw=instruction,
        opcode=(instruction & 0xF000) >> 12,
        x=(instruction & 0x0F00) >> 8,
        y=(instruction & 0x00F0) >> 4,
        n=instruction & 0x000F,
        nn=instruction & 0x00FF,
        nnn=instruction & 0x0FFF,
    )
