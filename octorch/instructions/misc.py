"""CHIP-8 miscellaneous instructions (Fxxx)."""

import torch

from octorch.constants import FONT_START, MEMORY_SIZE
from octorch.decode import DecodedInstruction
from octorch.ops import take, put, take_many, bmask
from octorch.state import EmulatorState

Tensor = torch.Tensor


def _vx(state, inst):
    return take(state.V, inst.x).to(torch.int64)


def execute_get_delay_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX07 - Set VX to delay timer value."""
    return state.replace(V=put(state.V, instruction.x, state.delay_timer))


def execute_set_delay_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX15 - Set delay timer to VX."""
    return state.replace(delay_timer=take(state.V, instruction.x))


def execute_set_sound_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX18 - Set sound timer to VX."""
    return state.replace(sound_timer=take(state.V, instruction.x))


def add_to_index(state: EmulatorState, instruction: DecodedInstruction) -> tuple[Tensor, Tensor]:
    """FX1E helper: returns ``(new_I, overflow_flag)`` as int64."""
    new_i = state.I.to(torch.int64) + _vx(state, instruction)
    return new_i & 0xFFF, (new_i > 0xFFF).to(torch.int64)


def execute_add_to_index(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX1E - Add VX to I register."""
    new_i, overflow = add_to_index(state, instruction)
    return state.replace(I=new_i.to(torch.int32), V=put(state.V, torch.full_like(instruction.x, 15), overflow))


def pressed_key(state: EmulatorState) -> tuple[Tensor, Tensor]:
    """Returns ``(any_key_pressed, index_of_first_pressed_key)``."""
    return state.keypad.any(-1), state.keypad.to(torch.int8).argmax(-1)


def execute_wait_for_key(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX0A - Wait for key press (blocking: rewinds PC while no key is pressed)."""
    any_key, key = pressed_key(state)
    return state.replace(
        V=torch.where(bmask(any_key, state.V), put(state.V, instruction.x, key), state.V),
        pc=torch.where(any_key, state.pc, state.pc - 2),
    )


def font_address(state: EmulatorState, instruction: DecodedInstruction) -> Tensor:
    return FONT_START + _vx(state, instruction) * 5


def execute_font_character(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX29 - Set I to location of sprite for digit VX."""
    return state.replace(I=font_address(state, instruction).to(torch.int32))


def bcd_digits(value: Tensor) -> Tensor:
    """``(..., 3)`` hundreds / tens / ones digits of ``value``."""
    return torch.stack([value // 100, (value // 10) % 10, value % 10], dim=-1)


def block_indices(state: EmulatorState) -> tuple[Tensor, Tensor]:
    """``(indices, valid)`` for the 16 bytes starting at I (``jnp`` style: OOB reads clamp, writes drop)."""
    idx = state.I.to(torch.int64)[..., None] + torch.arange(16, device=state.I.device)
    return idx.clamp(0, MEMORY_SIZE - 1), idx < MEMORY_SIZE


def execute_bcd_conversion(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX33 - Store BCD representation of VX at I, I+1, I+2."""
    idx, valid = block_indices(state)
    idx, valid = idx[..., :3], valid[..., :3]
    digits = bcd_digits(_vx(state, instruction))
    current = take_many(state.memory, idx)
    new_values = torch.where(valid, digits.to(state.memory.dtype), current)
    return state.replace(memory=state.memory.scatter(-1, idx, new_values))


def _register_mask(instruction: DecodedInstruction) -> Tensor:
    return torch.arange(16, device=instruction.x.device) <= instruction.x[..., None]


def execute_store_registers(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX55 - Store V0 through VX in memory starting at I."""
    idx, valid = block_indices(state)
    current = take_many(state.memory, idx)
    new_values = torch.where(_register_mask(instruction) & valid, state.V, current)
    new_memory = state.memory.scatter(-1, idx, new_values)
    if state.modern_mode:
        return state.replace(memory=new_memory)
    return state.replace(memory=new_memory, I=(state.I.to(torch.int64) + instruction.x + 1).to(torch.int32))


def execute_load_registers(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX65 - Load V0 through VX from memory starting at I."""
    idx, _ = block_indices(state)
    memory_values = take_many(state.memory, idx)
    new_V = torch.where(_register_mask(instruction), memory_values, state.V)
    if state.modern_mode:
        return state.replace(V=new_V)
    return state.replace(V=new_V, I=(state.I.to(torch.int64) + instruction.x + 1).to(torch.int32))


MISC_HANDLERS = {
    0x07: execute_get_delay_timer,
    0x0A: execute_wait_for_key,
    0x15: execute_set_delay_timer,
    0x18: execute_set_sound_timer,
    0x1E: execute_add_to_index,
    0x29: execute_font_character,
    0x33: execute_bcd_conversion,
    0x55: execute_store_registers,
    0x65: execute_load_registers,
}
