"""Cross-checks between the batched tensor engine, the scalar reference
interpreter, the standalone instruction handlers and batched/unbatched runs."""

import random

import pytest
import torch

from octorch import create_state, execute, fetch, load_rom, run_n_instruction, decode
from octorch.emulator import run_instruction_
from octorch.environments import get_rom_path
from octorch.interpreter import ScalarChip8
from octorch.instructions import alu, control_flow, display, memory, misc, system

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _leaves(s):
    return {
        "rng": s.rng, "memory": s.memory, "pc": s.pc, "display": s.display,
        "stack": s.stack.data, "sp": s.stack.pointer, "delay": s.delay_timer,
        "sound": s.sound_timer, "keypad": s.keypad, "V": s.V, "I": s.I,
    }


def assert_states_equal(a, b, msg=""):
    la, lb = _leaves(a), _leaves(b)
    for k in la:
        assert torch.equal(la[k].cpu(), lb[k].cpu()), f"{msg} field {k} differs"


def random_instruction(rng: random.Random) -> int:
    """Random instruction biased towards defined opcodes and small addresses."""
    op = rng.randrange(16)
    x, y, n = rng.randrange(16), rng.randrange(16), rng.randrange(16)
    if op == 0:
        return rng.choice([0x00E0, 0x00EE, 0x0123])
    if op in (1, 2, 0xA, 0xB):
        return (op << 12) | rng.randrange(0x200, 0x300)
    if op == 0xF:
        nn = rng.choice([0x07, 0x0A, 0x15, 0x18, 0x1E, 0x29, 0x33, 0x55, 0x65, 0x99])
        return (op << 12) | (x << 8) | nn
    if op == 0xE:
        return (op << 12) | (x << 8) | rng.choice([0x9E, 0xA1])
    return (op << 12) | (x << 8) | (y << 4) | n


def randomized_state(rng: random.Random, device, modern=True):
    state = create_state(rng.randrange(1 << 30), device=device, modern_mode=modern)
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    state.V.copy_(torch.randint(0, 256, (16,), generator=g, dtype=torch.uint8).to(device))
    state.I.fill_(rng.randrange(0x200, 0x300))
    state.memory[0x200:0x300] = torch.randint(0, 256, (0x100,), generator=g, dtype=torch.uint8).to(device)
    state.keypad.copy_((torch.rand(16, generator=g) < 0.2).to(device))
    state.display.copy_((torch.rand(64, 32, generator=g) < 0.3).to(device))
    state.stack.pointer.fill_(rng.randrange(0, 4))
    state.stack.data[:4] = torch.tensor([0x210, 0x220, 0x230, 0x240], dtype=torch.int32, device=device)
    state.delay_timer.fill_(rng.randrange(256))
    state.sound_timer.fill_(rng.randrange(256))
    return state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("modern", [True, False])
def test_tensor_engine_matches_scalar_interpreter(device, modern):
    rng = random.Random(1234 + int(modern))
    for trial in range(40):
        state = randomized_state(rng, device, modern)
        machine = ScalarChip8(state)
        for _ in range(25):
            instruction = random_instruction(rng)
            state = execute(state, instruction)
            machine.execute(instruction)
        assert_states_equal(state, machine.to_state(state), f"trial {trial}")


@pytest.mark.parametrize("device", DEVICES)
def test_batched_matches_unbatched(device):
    rng = random.Random(7)
    singles = [randomized_state(rng, device) for _ in range(6)]
    from octorch.struct import stack
    batched = stack(singles)
    for _ in range(30):
        ins = torch.tensor([random_instruction(rng) for _ in singles], device=device)
        batched = execute(batched, ins)
        singles = [execute(s, int(i)) for s, i in zip(singles, ins)]
    for b, s in enumerate(singles):
        assert_states_equal(batched[b], s, f"lane {b}")


@pytest.mark.parametrize("device", DEVICES)
def test_multi_dim_batch(device):
    state = load_rom(create_state(batch_shape=(2, 3), device=device), get_rom_path("test_opcode.ch8"))
    ref = load_rom(create_state(device=device), get_rom_path("test_opcode.ch8"))
    state.rng.fill_(int(ref.rng))
    out = run_n_instruction(state, 200)
    ref = run_n_instruction(ref, 200)
    for i in range(2):
        for j in range(3):
            assert_states_equal(out[i, j], ref)


def test_scalar_and_tensor_run_n_instruction_agree():
    state = load_rom(create_state(), get_rom_path("test_opcode.ch8"))
    fast = run_n_instruction(state, 500, scalar=True)
    slow = run_n_instruction(state, 500, scalar=False)
    assert_states_equal(fast, slow)


def test_test_opcode_rom_passes():
    """The classic corax89 test ROM draws 'OK' next to every instruction group."""
    state = load_rom(create_state(), get_rom_path("test_opcode.ch8"))
    state = run_n_instruction(state, 600)
    display = state.display
    # Sprite "OK" glyph columns are drawn at x=... check no 'ERR' style output: total on pixels known.
    assert int(display.sum()) == 626


HANDLERS = {
    0x0: system.execute_system_instruction,
    0x1: control_flow.execute_jump,
    0x2: control_flow.execute_call,
    0x3: control_flow.execute_skip_if_equal_immediate,
    0x4: control_flow.execute_skip_if_not_equal_immediate,
    0x5: control_flow.execute_skip_if_equal_register,
    0x6: memory.execute_set,
    0x7: memory.execute_add,
    0x8: alu.execute_alu_operation,
    0x9: control_flow.execute_skip_if_not_equal_register,
    0xA: memory.execute_set_index,
    0xC: memory.execute_random,
    0xD: display.execute_display,
    0xE: control_flow.execute_skip_if_key,
}


def test_standalone_handlers_match_fused_engine():
    rng = random.Random(99)
    for trial in range(200):
        state = randomized_state(rng, "cpu", modern=bool(trial % 2))
        instruction = random_instruction(rng)
        decoded = decode(instruction)
        op = int(decoded.opcode)
        if op == 0xB:
            handler = (control_flow.execute_jump_with_offset_modern if state.modern_mode
                       else control_flow.execute_jump_with_offset_legacy)
        elif op == 0xF:
            handler = misc.MISC_HANDLERS.get(int(decoded.nn), system.no_op)
        else:
            handler = HANDLERS[op]
        assert_states_equal(execute(state, instruction), handler(state, decoded), f"{instruction:04X}")


def test_execute_is_functional_and_execute_inplace_is_not():
    state = create_state()
    new = execute(state, 0x6A42)
    assert int(state.V[10]) == 0 and int(new.V[10]) == 0x42
    from octorch import execute_
    same = execute_(state, 0x6A42)
    assert same is state and int(state.V[10]) == 0x42


def test_fetch_advances_pc():
    state = load_rom(create_state(), get_rom_path("test_opcode.ch8"))
    new, instruction = fetch(state)
    assert int(new.pc) == int(state.pc) + 2
    assert int(instruction) == (int(state.memory[0x200]) << 8) | int(state.memory[0x201])
