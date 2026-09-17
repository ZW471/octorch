"""CHIP-8 control flow instructions."""

import torch

from octorch.decode import DecodedInstruction
from octorch.ops import take
from octorch.stack import push
from octorch.state import EmulatorState

Tensor = torch.Tensor


def execute_jump(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """1NNN - Jump to address NNN."""
    return state.replace(pc=instruction.nnn.to(torch.int32))


def execute_call(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """2NNN - Call subroutine at NNN."""
    state = state.replace(stack=push(state.stack, state.pc))
    return execute_jump(state, instruction)


def make_skip_instruction(condition_fn):
    """Factory for skip instructions."""
    def skip_instruction(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
        condition = condition_fn(state, instruction)
        return state.replace(pc=torch.where(condition, state.pc + 2, state.pc))
    return skip_instruction


def _vx(state, inst):
    return take(state.V, inst.x).to(torch.int64)


def _vy(state, inst):
    return take(state.V, inst.y).to(torch.int64)


execute_skip_if_equal_immediate = make_skip_instruction(lambda s, i: _vx(s, i) == i.nn)
execute_skip_if_not_equal_immediate = make_skip_instruction(lambda s, i: _vx(s, i) != i.nn)
execute_skip_if_equal_register = make_skip_instruction(lambda s, i: _vx(s, i) == _vy(s, i))
execute_skip_if_not_equal_register = make_skip_instruction(lambda s, i: _vx(s, i) != _vy(s, i))


def jump_with_offset_target(state: EmulatorState, instruction: DecodedInstruction) -> Tensor:
    """Target of BXNN / BNNN depending on ``modern_mode`` (int64)."""
    if state.modern_mode:
        return (instruction.nn + _vx(state, instruction)) & 0xFFF
    return (instruction.nnn + state.V[..., 0].to(torch.int64)) & 0xFFF


def execute_jump_with_offset_modern(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """BXNN - Jump to address NN + VX (modern behavior)."""
    return state.replace(pc=((instruction.nn + _vx(state, instruction)) & 0xFFF).to(torch.int32))


def execute_jump_with_offset_legacy(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """BNNN - Jump to address NNN + V0 (legacy behavior)."""
    return state.replace(pc=((instruction.nnn + state.V[..., 0].to(torch.int64)) & 0xFFF).to(torch.int32))


def skip_if_key_condition(state: EmulatorState, instruction: DecodedInstruction) -> Tensor:
    """Condition of EX9E/EXA1 (any other EXNN behaves like EX9E, as in Octax)."""
    key_index = _vx(state, instruction) & 0xF
    key_pressed = take(state.keypad, key_index)
    is_not_instruction = instruction.nn == 0xA1
    return key_pressed ^ is_not_instruction


execute_skip_if_key = make_skip_instruction(skip_if_key_condition)
