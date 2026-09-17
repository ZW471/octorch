"""CHIP-8 ALU operations (8xxx).

Each ``alu_*`` function mirrors the corresponding Octax helper and works on
int64 tensors holding 8-bit values; :func:`execute_alu_operation` evaluates all
of them and selects by the instruction's last nibble.
"""

import torch

from octorch.decode import DecodedInstruction
from octorch.ops import take, put
from octorch.state import EmulatorState

Tensor = torch.Tensor


def alu_set(vx: Tensor, vy: Tensor):
    """8XY0 - Set: VX = VY."""
    return vy, torch.zeros_like(vx)


def alu_or(vx: Tensor, vy: Tensor):
    """8XY1 - Binary OR: VX |= VY."""
    return vx | vy, torch.zeros_like(vx)


def alu_and(vx: Tensor, vy: Tensor):
    """8XY2 - Binary AND: VX &= VY."""
    return vx & vy, torch.zeros_like(vx)


def alu_xor(vx: Tensor, vy: Tensor):
    """8XY3 - Logical XOR: VX ^= VY."""
    return vx ^ vy, torch.zeros_like(vx)


def alu_add(vx: Tensor, vy: Tensor):
    """8XY4 - Add: VX += VY, set carry flag."""
    result = vx + vy
    return result & 0xFF, (result > 255).to(vx.dtype)


def alu_sub_xy(vx: Tensor, vy: Tensor):
    """8XY5 - Subtract: VX -= VY, set borrow flag."""
    return (vx - vy) & 0xFF, (vx >= vy).to(vx.dtype)


def alu_shift_right(vx: Tensor, vy: Tensor):
    """8XY6 - Shift right: VX >>= 1."""
    return vx >> 1, vx & 1


def alu_shift_left(vx: Tensor, vy: Tensor):
    """8XYE - Shift left: VX <<= 1."""
    return (vx << 1) & 0xFF, (vx & 0x80) >> 7


def alu_sub_yx(vx: Tensor, vy: Tensor):
    """8XY7 - Subtract: VX = VY - VX, set borrow flag."""
    return (vy - vx) & 0xFF, (vy >= vx).to(vx.dtype)


def alu_undefined(vx: Tensor, vy: Tensor):
    """Undefined ALU operation."""
    return vx, torch.zeros_like(vx)


def alu_results(vx: Tensor, vy: Tensor, n: Tensor, modern_mode: bool) -> tuple[Tensor, Tensor]:
    """Evaluate every ALU op and select by ``n``. Returns ``(result, vf)`` int64 tensors."""
    shift_src = vx if modern_mode else vy
    ops = [
        alu_set(vx, vy), alu_or(vx, vy), alu_and(vx, vy), alu_xor(vx, vy),
        alu_add(vx, vy), alu_sub_xy(vx, vy), alu_shift_right(shift_src, vy), alu_sub_yx(vx, vy),
    ]
    undefined = alu_undefined(vx, vy)
    ops += [undefined] * 6          # 8..D
    ops.append(alu_shift_left(shift_src, vy))  # E
    ops.append(undefined)           # F
    results = torch.stack([r for r, _ in ops], dim=-1)
    flags = torch.stack([f for _, f in ops], dim=-1)
    return take(results, n), take(flags, n)


def execute_alu_operation(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """8XYN - ALU operations dispatcher (functional, single-opcode version)."""
    vx = take(state.V, instruction.x).to(torch.int64)
    vy = take(state.V, instruction.y).to(torch.int64)
    result, vf = alu_results(vx, vy, instruction.n, state.modern_mode)
    new_V = put(state.V, instruction.x, result)
    new_V = put(new_V, torch.full_like(instruction.x, 15), vf)
    return state.replace(V=new_V)
