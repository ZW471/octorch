"""Main CHIP-8 emulator execution engine.

Octax dispatches one instruction through ``jax.lax.switch`` and relies on
``jax.vmap`` to batch emulators; under ``vmap`` XLA evaluates every branch and
selects per lane.  Octorch does exactly that explicitly: :func:`execute`
evaluates the effect of all opcode families for the whole batch and merges them
with masks, so a single call advances any number of emulators at once.

Two flavours are provided:

* :func:`execute` / :func:`fetch` are functional (return a new state, leave the
  input untouched) like their Octax counterparts.
* :func:`execute_` / :func:`fetch_` mutate the state's tensors in place; they
  avoid one copy of memory / display per instruction and are used by
  :class:`octorch.env.OctorchEnv` (which clones the incoming state once per
  environment step).
"""

from __future__ import annotations

from typing import Union

import torch

from octorch import prng
from octorch.constants import PROGRAM_START, FONT_START, MEMORY_SIZE, STACK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
from octorch.decode import DecodedInstruction, decode
from octorch.instructions.alu import alu_results
from octorch.ops import take, take_many, bmask
from octorch.state import EmulatorState

Tensor = torch.Tensor


def _pack_u16(high: Tensor, low: Tensor) -> Tensor:
    """Pack two bytes into a 16-bit value (int64)."""
    return (high.to(torch.int64) << 8) | low.to(torch.int64)


def _unpack_u16(value: Tensor) -> tuple[Tensor, Tensor]:
    """Unpack a 16-bit value into two bytes."""
    return ((value >> 8) & 0xFF).to(torch.uint8), (value & 0xFF).to(torch.uint8)


def fetch(state: EmulatorState) -> tuple[EmulatorState, Tensor]:
    """Fetch next instruction from memory (functional)."""
    pc = state.pc.to(torch.int64)
    instruction = _pack_u16(
        take(state.memory, pc.clamp(0, MEMORY_SIZE - 1)),
        take(state.memory, (pc + 1).clamp(0, MEMORY_SIZE - 1)),
    )
    return state.replace(pc=state.pc + 2), instruction


def fetch_(state: EmulatorState) -> tuple[EmulatorState, Tensor]:
    """Fetch next instruction from memory, advancing ``pc`` in place."""
    pc = state.pc.to(torch.int64)
    instruction = _pack_u16(
        take(state.memory, pc.clamp(0, MEMORY_SIZE - 1)),
        take(state.memory, (pc + 1).clamp(0, MEMORY_SIZE - 1)),
    )
    state.pc.add_(2)
    return state, instruction


def execute(state: EmulatorState, instruction: Union[int, Tensor]) -> EmulatorState:
    """Execute a single CHIP-8 instruction per batch element (functional)."""
    return execute_(state.clone(), instruction)


def execute_(state: EmulatorState, instruction: Union[int, Tensor]) -> EmulatorState:
    """Execute a single CHIP-8 instruction per batch element, mutating ``state`` in place.

    Args:
        state: emulator state with batch shape ``(...)``.
        instruction: int or int tensor broadcastable to the batch shape.
    Returns:
        The same ``state`` object (for chaining).
    """
    device = state.pc.device
    if not isinstance(instruction, torch.Tensor):
        instruction = torch.tensor(int(instruction), dtype=torch.int64, device=device)
    instruction = torch.broadcast_to(instruction.to(torch.int64), state.pc.shape)
    ins: DecodedInstruction = decode(instruction)
    modern = state.modern_mode

    op, x, y, n, nn, nnn, raw = ins.opcode, ins.x, ins.y, ins.n, ins.nn, ins.nnn, ins.raw
    V, memory, display, keypad = state.V, state.memory, state.display, state.keypad
    pc = state.pc.to(torch.int64)
    I = state.I.to(torch.int64)
    vx = take(V, x).to(torch.int64)
    vy = take(V, y).to(torch.int64)

    # ---- opcode masks -------------------------------------------------------
    is_cls = raw == 0x00E0
    is_ret = raw == 0x00EE
    is_jp, is_call = op == 0x1, op == 0x2
    is_se_b, is_sne_b, is_se_r = op == 0x3, op == 0x4, op == 0x5
    is_ld_b, is_add_b, is_alu = op == 0x6, op == 0x7, op == 0x8
    is_sne_r, is_ld_i, is_jp_v0 = op == 0x9, op == 0xA, op == 0xB
    is_rnd, is_drw, is_key, is_misc = op == 0xC, op == 0xD, op == 0xE, op == 0xF
    f07 = is_misc & (nn == 0x07)
    f0A = is_misc & (nn == 0x0A)
    f15 = is_misc & (nn == 0x15)
    f18 = is_misc & (nn == 0x18)
    f1E = is_misc & (nn == 0x1E)
    f29 = is_misc & (nn == 0x29)
    f33 = is_misc & (nn == 0x33)
    f55 = is_misc & (nn == 0x55)
    f65 = is_misc & (nn == 0x65)

    # ---- keypad dependent values --------------------------------------------
    key_pressed = take(keypad, vx & 0xF)
    key_skip = key_pressed ^ (nn == 0xA1)
    any_key = keypad.any(-1)
    first_key = keypad.to(torch.int8).argmax(-1)

    # ---- stack ----------------------------------------------------------------
    stack_data, pointer = state.stack.data, state.stack.pointer
    ptr = pointer.to(torch.int64)
    push_idx = torch.where(ptr < 0, ptr + STACK_SIZE, ptr)
    push_valid = is_call & (push_idx >= 0) & (push_idx < STACK_SIZE)
    push_idx = push_idx.clamp(0, STACK_SIZE - 1)
    pop_idx = ptr - 1
    pop_idx = torch.where(pop_idx < 0, pop_idx + STACK_SIZE, pop_idx)
    pop_valid = is_ret & (pop_idx >= 0) & (pop_idx < STACK_SIZE)
    pop_idx = pop_idx.clamp(0, STACK_SIZE - 1)
    popped = take(stack_data, pop_idx).to(torch.int64)

    # ---- program counter ------------------------------------------------------
    skip = (
        (is_se_b & (vx == nn)) | (is_sne_b & (vx != nn)) | (is_se_r & (vx == vy))
        | (is_sne_r & (vx != vy)) | (is_key & key_skip)
    )
    new_pc = torch.where(skip, pc + 2, pc)
    new_pc = torch.where(f0A & ~any_key, pc - 2, new_pc)
    new_pc = torch.where(is_jp | is_call, nnn, new_pc)
    if modern:
        jump_target = (nn + vx) & 0xFFF
    else:
        jump_target = (nnn + V[..., 0].to(torch.int64)) & 0xFFF
    new_pc = torch.where(is_jp_v0, jump_target, new_pc)
    new_pc = torch.where(is_ret, popped, new_pc)

    # ---- ALU / random ---------------------------------------------------------
    alu_res, alu_vf = alu_results(vx, vy, n, modern)
    rng_next, rand = prng.random_byte(state.rng)

    # ---- 16-byte memory window at I (FX33 / FX55 / FX65) ----------------------
    ar16 = torch.arange(16, device=device)
    idx16 = I[..., None] + ar16
    idx_valid = idx16 < MEMORY_SIZE
    mem16 = take_many(memory, idx16.clamp(0, MEMORY_SIZE - 1))
    reg_mask = ar16 <= x[..., None]
    digits = torch.stack([vx // 100, (vx // 10) % 10, vx % 10], dim=-1).to(torch.uint8)
    digits = torch.nn.functional.pad(digits, (0, 13))
    bcd_mask = ar16 < 3
    write_mask = ((bmask(f55, mem16) & reg_mask) | (bmask(f33, mem16) & bcd_mask)) & idx_valid
    write_vals = torch.where(bmask(f33, mem16), digits, V)
    # Invalid (out of range) lanes are redirected to address 0 and rewrite the
    # current byte, which is a deterministic no-op (only I > 4080 has such lanes).
    idx_w = torch.where(idx_valid, idx16, torch.zeros_like(idx16))
    cur_w = torch.where(idx_valid, mem16, memory[..., :1].expand(mem16.shape))
    write_vals = torch.where(write_mask, write_vals, cur_w)

    # ---- display (DXYN / 00E0) -------------------------------------------------
    # Only the 16x8 sprite patch is touched. Every patch lane maps to a unique
    # screen pixel (positions wrap modulo the screen size); lanes that are not
    # drawn (row >= n, off-screen, or not a DRW) write back the pixel unchanged.
    rows = torch.arange(16, device=device)
    cols = torch.arange(8, device=device)
    sx = (vx % SCREEN_WIDTH)[..., None, None]
    sy = (vy % SCREEN_HEIGHT)[..., None, None]
    px = sx + cols                                    # (..., 1, 8)
    py = sy + rows[:, None]                           # (..., 16, 1)
    drawn = (px < SCREEN_WIDTH) & (py < SCREEN_HEIGHT) & (rows[:, None] < n[..., None, None]) & bmask(is_drw, py)
    bits = ((mem16.to(torch.int64)[..., :, None] >> (7 - cols)) & 1) != 0   # (..., 16, 8)
    bits = bits & drawn
    pix = ((px % SCREEN_WIDTH) * SCREEN_HEIGHT + (py % SCREEN_HEIGHT)).reshape(lead_shape := vx.shape + (128,))
    disp_flat = display.reshape(vx.shape + (SCREEN_WIDTH * SCREEN_HEIGHT,))
    old_pix = disp_flat.gather(-1, pix)
    bits = bits.reshape(lead_shape)
    collision = (old_pix & bits).any(-1)
    new_pix = old_pix ^ bits

    # ---- registers ---------------------------------------------------------------
    vx_new = torch.where(is_ld_b, nn, vx)
    vx_new = torch.where(is_add_b, (vx + nn) & 0xFF, vx_new)
    vx_new = torch.where(is_alu, alu_res, vx_new)
    vx_new = torch.where(is_rnd, rand & nn, vx_new)
    vx_new = torch.where(f07, state.delay_timer.to(torch.int64), vx_new)
    vx_new = torch.where(f0A & any_key, first_key, vx_new)
    load_V = torch.where(reg_mask, mem16, V)
    V_new = V.scatter(-1, x.unsqueeze(-1), vx_new.to(torch.uint8).unsqueeze(-1))
    V_new = torch.where(bmask(f65, V), load_V, V_new)
    i_add = I + vx
    vf_new = V_new[..., 15].to(torch.int64)
    vf_new = torch.where(is_alu, alu_vf, vf_new)
    vf_new = torch.where(is_drw, collision.to(torch.int64), vf_new)
    vf_new = torch.where(f1E, (i_add > 0xFFF).to(torch.int64), vf_new)
    V_new = V_new.clone()
    V_new[..., 15] = vf_new.to(torch.uint8)

    # ---- index register -----------------------------------------------------------
    I_new = torch.where(is_ld_i, nnn, I)
    I_new = torch.where(f1E, i_add & 0xFFF, I_new)
    I_new = torch.where(f29, FONT_START + vx * 5, I_new)
    if not modern:
        I_new = torch.where(f55 | f65, I + x + 1, I_new)

    # ---- commit (in place) ----------------------------------------------------------
    memory.scatter_(-1, idx_w, write_vals)
    disp_flat.scatter_(-1, pix, new_pix)
    display.masked_fill_(bmask(is_cls, display), False)
    V.copy_(V_new)
    state.pc.copy_(new_pc.to(state.pc.dtype))
    state.I.copy_(I_new.to(state.I.dtype))
    state.rng.copy_(torch.where(is_rnd, rng_next, state.rng))
    state.delay_timer.copy_(torch.where(f15, vx, state.delay_timer.to(torch.int64)).to(torch.uint8))
    state.sound_timer.copy_(torch.where(f18, vx, state.sound_timer.to(torch.int64)).to(torch.uint8))
    push_val = torch.where(push_valid, pc & 0xFFF, take(stack_data, push_idx).to(torch.int64))
    stack_data.scatter_(-1, push_idx.unsqueeze(-1), push_val.to(stack_data.dtype).unsqueeze(-1))
    pop_val = torch.where(pop_valid, torch.zeros_like(popped), take(stack_data, pop_idx).to(torch.int64))
    stack_data.scatter_(-1, pop_idx.unsqueeze(-1), pop_val.to(stack_data.dtype).unsqueeze(-1))
    pointer.copy_((ptr + is_call.to(torch.int64) - is_ret.to(torch.int64)).to(pointer.dtype))
    return state


def run_instruction(state: EmulatorState) -> EmulatorState:
    """Fetch-decode-execute one instruction (functional)."""
    state, instruction = fetch(state)
    return execute(state, instruction)


def run_instruction_(state: EmulatorState) -> EmulatorState:
    """Fetch-decode-execute one instruction in place."""
    state, instruction = fetch_(state)
    return execute_(state, instruction)


def run_n_instruction(state: EmulatorState, n: int, scalar: bool = True) -> EmulatorState:
    """Run ``n`` instructions (functional: the input state is not modified).

    Unbatched states are executed by the scalar reference interpreter (much
    faster for a single machine, identical semantics) unless ``scalar=False``.
    """
    if scalar and state.pc.dim() == 0:
        from octorch.interpreter import run_n_instruction_scalar
        return run_n_instruction_scalar(state, n)
    state = state.clone()
    for _ in range(int(n)):
        run_instruction_(state)
    return state


def load_rom(state: EmulatorState, filename: str) -> EmulatorState:
    """Load ROM data into CHIP-8 memory starting at 0x200."""
    with open(filename, "rb") as f:
        rom_data = f.read()
    return load_rom_bytes(state, rom_data)


def load_rom_bytes(state: EmulatorState, rom_data: bytes) -> EmulatorState:
    """Load raw ROM bytes into CHIP-8 memory starting at 0x200."""
    rom = torch.frombuffer(bytearray(rom_data), dtype=torch.uint8).to(state.memory.device)
    memory = state.memory.clone()
    memory[..., PROGRAM_START:PROGRAM_START + len(rom_data)] = rom
    return state.replace(memory=memory)
