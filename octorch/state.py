"""CHIP-8 emulator state structures."""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Union

import torch

from octorch.constants import (
    PROGRAM_START, FONT_START, FONT_DATA, SCREEN_WIDTH, SCREEN_HEIGHT, STACK_SIZE, MEMORY_SIZE,
)
from octorch.prng import make_rng
from octorch.struct import Struct, static_field

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class StackState(Struct):
    """Stack state for subroutine calls.

    Attributes:
        data: ``(..., 16)`` int32 return addresses.
        pointer: ``(...,)`` int32 stack pointer.
    """
    data: Tensor
    pointer: Tensor


@dataclasses.dataclass(frozen=True)
class EmulatorState(Struct):
    """Main CHIP-8 emulator state.

    Every tensor has the same leading *batch shape* ``(...)`` (possibly empty for a
    single emulator) followed by the field's own shape:

    * ``rng``: ``(...)`` int64 – xorshift32 PRNG state (see :mod:`octorch.prng`)
    * ``memory``: ``(..., 4096)`` uint8
    * ``pc``: ``(...)`` int32 – program counter
    * ``display``: ``(..., 64, 32)`` bool – indexed ``[x, y]`` like Octax
    * ``stack``: :class:`StackState`
    * ``delay_timer`` / ``sound_timer``: ``(...)`` uint8
    * ``keypad``: ``(..., 16)`` bool
    * ``V``: ``(..., 16)`` uint8 – general purpose registers
    * ``I``: ``(...)`` int32 – index register
    * ``modern_mode``: static python bool selecting modern vs legacy quirks
    """
    rng: Tensor
    memory: Tensor
    pc: Tensor
    display: Tensor
    stack: StackState
    delay_timer: Tensor
    sound_timer: Tensor
    keypad: Tensor
    V: Tensor
    I: Tensor
    modern_mode: bool = static_field(default=True)

    @property
    def batch_shape(self) -> torch.Size:
        return self.pc.shape

    @property
    def batch_size(self) -> int:
        return self.pc.numel()

    def expand(self, batch_shape: Union[int, Sequence[int]]) -> "EmulatorState":
        """Broadcast an unbatched state to ``batch_shape`` (materialised copy)."""
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        batch_shape = tuple(batch_shape)
        lead = len(self.batch_shape)

        def _expand(t: Tensor) -> Tensor:
            trailing = t.shape[lead:]
            t = t.reshape((1,) * len(batch_shape) + tuple(t.shape[lead:])) if lead == 0 else t
            return t.expand(batch_shape + tuple(trailing)).clone()

        from octorch.struct import tree_map
        return tree_map(_expand, self)


def empty_state(
    batch_shape: Sequence[int] = (),
    device=None,
    modern_mode: bool = True,
    rng: Optional[Tensor] = None,
) -> EmulatorState:
    """Allocate a zero-filled emulator state (no font data)."""
    bs = tuple(batch_shape)
    if rng is None:
        rng = make_rng(0, bs, device)
    return EmulatorState(
        rng=rng,
        memory=torch.zeros(bs + (MEMORY_SIZE,), dtype=torch.uint8, device=device),
        pc=torch.full(bs, PROGRAM_START, dtype=torch.int32, device=device),
        display=torch.zeros(bs + (SCREEN_WIDTH, SCREEN_HEIGHT), dtype=torch.bool, device=device),
        stack=StackState(
            data=torch.zeros(bs + (STACK_SIZE,), dtype=torch.int32, device=device),
            pointer=torch.zeros(bs, dtype=torch.int32, device=device),
        ),
        delay_timer=torch.zeros(bs, dtype=torch.uint8, device=device),
        sound_timer=torch.zeros(bs, dtype=torch.uint8, device=device),
        keypad=torch.zeros(bs + (16,), dtype=torch.bool, device=device),
        V=torch.zeros(bs + (16,), dtype=torch.uint8, device=device),
        I=torch.zeros(bs, dtype=torch.int32, device=device),
        modern_mode=modern_mode,
    )


def create_state(
    rng: Union[int, Tensor] = 0,
    batch_shape: Sequence[int] = (),
    device=None,
    modern_mode: bool = True,
) -> EmulatorState:
    """Create initial emulator state with font data loaded.

    Args:
        rng: integer seed or a tensor of PRNG states with shape ``batch_shape``.
        batch_shape: leading batch shape (``()`` for a single emulator).
        device: torch device.
        modern_mode: use modern (True) or legacy (False) CHIP-8 quirks.
    """
    if isinstance(rng, torch.Tensor) and rng.numel() > 1:
        batch_shape = tuple(rng.shape)
        rng = rng.to(device=device, dtype=torch.int64)
    else:
        rng = make_rng(rng, batch_shape, device)
    state = empty_state(batch_shape, device, modern_mode, rng)
    memory = state.memory
    memory[..., FONT_START:FONT_START + len(FONT_DATA)] = FONT_DATA.to(memory.device)
    return state
