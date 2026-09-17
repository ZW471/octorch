"""CHIP-8 system instructions (0x0xxx)."""

import torch

from octorch.decode import DecodedInstruction
from octorch.stack import pop
from octorch.state import EmulatorState


def no_op(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """No operation."""
    return state


def execute_clear_screen(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """00E0 - Clear display."""
    return state.replace(display=torch.zeros_like(state.display))


def execute_return(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """00EE - Return from subroutine."""
    stack, address = pop(state.stack)
    return state.replace(stack=stack, pc=address.to(torch.int32))


def execute_system_instruction(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """Dispatch system instructions (functional, single-opcode version)."""
    is_cls = instruction.raw == 0x00E0
    is_ret = instruction.raw == 0x00EE
    cleared = execute_clear_screen(state, instruction)
    returned = execute_return(state, instruction)
    from octorch.ops import bmask
    return state.replace(
        display=torch.where(bmask(is_cls, state.display), cleared.display, state.display),
        pc=torch.where(is_ret, returned.pc, state.pc),
        stack=state.stack.replace(
            data=torch.where(bmask(is_ret, state.stack.data), returned.stack.data, state.stack.data),
            pointer=torch.where(is_ret, returned.stack.pointer, state.stack.pointer),
        ),
    )
