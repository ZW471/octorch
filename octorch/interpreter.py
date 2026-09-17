"""Scalar (pure Python) reference CHIP-8 interpreter.

This is a straightforward, readable implementation of exactly the same semantics
as the batched tensor engine in :mod:`octorch.emulator` (including PRNG,
quirk modes and out-of-bounds behaviour).  It serves two purposes:

* it is the oracle used by the test-suite to validate the tensor engine, and
* it is much faster than the tensor engine for a *single* emulator, so
  :func:`octorch.emulator.run_n_instruction` uses it for unbatched states
  (e.g. the start-up instructions executed once at environment creation).
"""

from __future__ import annotations

import numpy as np
import torch

from octorch.constants import FONT_START, MEMORY_SIZE, STACK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
from octorch.state import EmulatorState, StackState

_M32 = 0xFFFFFFFF


class ScalarChip8:
    """Mutable, unbatched CHIP-8 machine backed by numpy arrays / python ints."""

    def __init__(self, state: EmulatorState):
        if state.pc.dim() != 0:
            raise ValueError("ScalarChip8 only supports unbatched states")
        self.rng = int(state.rng)
        self.memory = state.memory.cpu().numpy().copy()
        self.pc = int(state.pc)
        self.display = state.display.cpu().numpy().copy()
        self.stack = state.stack.data.cpu().numpy().astype(np.int64).copy()
        self.sp = int(state.stack.pointer)
        self.delay_timer = int(state.delay_timer)
        self.sound_timer = int(state.sound_timer)
        self.keypad = state.keypad.cpu().numpy().copy()
        self.V = state.V.cpu().numpy().astype(np.int64).copy()
        self.I = int(state.I)
        self.modern_mode = state.modern_mode

    def to_state(self, template: EmulatorState) -> EmulatorState:
        dev = template.pc.device
        return template.replace(
            rng=torch.tensor(self.rng, dtype=torch.int64, device=dev),
            memory=torch.from_numpy(self.memory.copy()).to(dev),
            pc=torch.tensor(self.pc, dtype=torch.int32, device=dev),
            display=torch.from_numpy(self.display.copy()).to(dev),
            stack=StackState(
                data=torch.from_numpy(self.stack.astype(np.int32)).to(dev),
                pointer=torch.tensor(self.sp, dtype=torch.int32, device=dev),
            ),
            delay_timer=torch.tensor(self.delay_timer, dtype=torch.uint8, device=dev),
            sound_timer=torch.tensor(self.sound_timer, dtype=torch.uint8, device=dev),
            keypad=torch.from_numpy(self.keypad.copy()).to(dev),
            V=torch.from_numpy(self.V.astype(np.uint8)).to(dev),
            I=torch.tensor(self.I, dtype=torch.int32, device=dev),
        )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _clamp(i: int) -> int:
        return min(max(i, 0), MEMORY_SIZE - 1)

    @staticmethod
    def _stack_index(i: int):
        if i < 0:
            i += STACK_SIZE
        return min(max(i, 0), STACK_SIZE - 1), 0 <= i < STACK_SIZE

    def fetch(self) -> int:
        instruction = (int(self.memory[self._clamp(self.pc)]) << 8) | int(self.memory[self._clamp(self.pc + 1)])
        self.pc += 2
        return instruction

    def random_byte(self) -> int:
        s = self.rng & _M32
        s = (s ^ (s << 13)) & _M32
        s ^= s >> 17
        s = (s ^ (s << 5)) & _M32
        self.rng = s
        return (s ^ (s >> 16)) & 0xFF

    def execute(self, instruction: int) -> None:
        op = (instruction & 0xF000) >> 12
        x = (instruction & 0x0F00) >> 8
        y = (instruction & 0x00F0) >> 4
        n = instruction & 0x000F
        nn = instruction & 0x00FF
        nnn = instruction & 0x0FFF
        V = self.V
        vx, vy = int(V[x]), int(V[y])

        if op == 0x0:
            if instruction == 0x00E0:
                self.display[:] = False
            elif instruction == 0x00EE:
                self.sp -= 1
                idx, valid = self._stack_index(self.sp)
                self.pc = int(self.stack[idx])
                if valid:
                    self.stack[idx] = 0
        elif op == 0x1:
            self.pc = nnn
        elif op == 0x2:
            idx, valid = self._stack_index(self.sp)
            if valid:
                self.stack[idx] = self.pc & 0xFFF
            self.sp += 1
            self.pc = nnn
        elif op == 0x3:
            if vx == nn:
                self.pc += 2
        elif op == 0x4:
            if vx != nn:
                self.pc += 2
        elif op == 0x5:
            if vx == vy:
                self.pc += 2
        elif op == 0x6:
            V[x] = nn
        elif op == 0x7:
            V[x] = (vx + nn) & 0xFF
        elif op == 0x8:
            src = vx if self.modern_mode else vy
            if n == 0x0:
                res, vf = vy, 0
            elif n == 0x1:
                res, vf = vx | vy, 0
            elif n == 0x2:
                res, vf = vx & vy, 0
            elif n == 0x3:
                res, vf = vx ^ vy, 0
            elif n == 0x4:
                res, vf = (vx + vy) & 0xFF, int(vx + vy > 255)
            elif n == 0x5:
                res, vf = (vx - vy) & 0xFF, int(vx >= vy)
            elif n == 0x6:
                res, vf = src >> 1, src & 1
            elif n == 0x7:
                res, vf = (vy - vx) & 0xFF, int(vy >= vx)
            elif n == 0xE:
                res, vf = (src << 1) & 0xFF, (src & 0x80) >> 7
            else:
                res, vf = vx, 0
            V[x] = res
            V[15] = vf
        elif op == 0x9:
            if vx != vy:
                self.pc += 2
        elif op == 0xA:
            self.I = nnn
        elif op == 0xB:
            self.pc = ((nn + vx) & 0xFFF) if self.modern_mode else ((nnn + int(V[0])) & 0xFFF)
        elif op == 0xC:
            V[x] = self.random_byte() & nn
        elif op == 0xD:
            sx, sy = vx % SCREEN_WIDTH, vy % SCREEN_HEIGHT
            collision = False
            for r in range(n):
                py = sy + r
                if py >= SCREEN_HEIGHT:
                    break
                byte = int(self.memory[self._clamp(self.I + r)])
                for c in range(8):
                    px = sx + c
                    if px >= SCREEN_WIDTH:
                        break
                    if (byte >> (7 - c)) & 1:
                        if self.display[px, py]:
                            collision = True
                        self.display[px, py] ^= True
            V[15] = int(collision)
        elif op == 0xE:
            pressed = bool(self.keypad[vx & 0xF])
            if pressed ^ (nn == 0xA1):
                self.pc += 2
        elif op == 0xF:
            if nn == 0x07:
                V[x] = self.delay_timer
            elif nn == 0x0A:
                if self.keypad.any():
                    V[x] = int(np.argmax(self.keypad))
                else:
                    self.pc -= 2
            elif nn == 0x15:
                self.delay_timer = vx
            elif nn == 0x18:
                self.sound_timer = vx
            elif nn == 0x1E:
                new_i = self.I + vx
                V[15] = int(new_i > 0xFFF)
                self.I = new_i & 0xFFF
            elif nn == 0x29:
                self.I = FONT_START + vx * 5
            elif nn == 0x33:
                for k, d in enumerate((vx // 100, (vx // 10) % 10, vx % 10)):
                    if self.I + k < MEMORY_SIZE:
                        self.memory[self.I + k] = d
            elif nn == 0x55:
                for k in range(x + 1):
                    if self.I + k < MEMORY_SIZE:
                        self.memory[self.I + k] = V[k]
                if not self.modern_mode:
                    self.I = self.I + x + 1
            elif nn == 0x65:
                for k in range(x + 1):
                    V[k] = int(self.memory[self._clamp(self.I + k)])
                if not self.modern_mode:
                    self.I = self.I + x + 1

    def step(self) -> None:
        self.execute(self.fetch())

    def run(self, n: int) -> None:
        for _ in range(n):
            self.execute(self.fetch())


def run_n_instruction_scalar(state: EmulatorState, n: int) -> EmulatorState:
    """Run ``n`` instructions on an unbatched state using the scalar interpreter."""
    machine = ScalarChip8(state)
    machine.run(int(n))
    return machine.to_state(state)
