# Core Emulator Concepts

This tutorial covers the CHIP-8 emulator implementation in Octorch. Understanding these concepts helps with troubleshooting, performance optimization, and extending functionality.

## The CHIP-8 Architecture

CHIP-8 isn't a real computer—it's a virtual machine specification from the 1970s designed to make game programming easier. Think of it as the "assembly language" of simple games. When you run a CHIP-8 game, you're actually running a tiny program written in CHIP-8 instructions on Octorch's virtual processor.

The CHIP-8 system has several key components:
- **Memory**: 4KB of RAM where both the program and data live
- **Registers**: 16 general-purpose registers (V0-VF) for temporary storage
- **Stack**: For remembering where to return from subroutines
- **Timers**: For controlling game speed and sound
- **Display**: A 64x32 pixel monochrome screen

## The Emulator State

Everything in Octorch revolves around the `EmulatorState`. This is a snapshot of the entire CHIP-8 system at any moment:

```python
import octorch

# Create a fresh CHIP-8 system (on CPU here; device="cuda" works the same)
state = octorch.create_state(device="cpu")

print(f"Program counter starts at: 0x{int(state.pc):03X}")
print(f"Memory size: {state.memory.shape[-1]} bytes")
print(f"Number of registers: {state.V.shape[-1]}")
print(f"Display dimensions: {tuple(state.display.shape)}")
print(f"Batch shape: {tuple(state.batch_shape)}")
```

`EmulatorState` is a frozen dataclass of tensors. Executing an instruction with `octorch.execute` creates a new state rather than modifying the existing one (like Octax); `octorch.execute_` is the in-place variant used by the environment for speed.

### Batched States

The distinctive feature of Octorch is that every tensor of the state has a leading *batch shape*. `create_state()` gives an unbatched state (batch shape `()`), `create_state(batch_shape=(8,))` gives eight independent machines, and every function in the package accepts either:

```python
batch = octorch.create_state(batch_shape=(8,), device="cpu")
print(f"Registers: {tuple(batch.V.shape)}, display: {tuple(batch.display.shape)}")

# Index the batch like a tensor
first = batch[0]
print(f"One machine: {tuple(first.V.shape)}")
```

This is why game definitions index registers with an ellipsis: `state.V[..., 5]` is register V5 of every machine in the batch, whatever the batch shape.

### Random Numbers

The `CXNN` instruction needs randomness. Octax threads a `jax.random.PRNGKey` through the state; Octorch stores a 32-bit xorshift state per machine in `state.rng` and advances it with tensor arithmetic, so no host synchronisation is needed:

```python
from octorch.prng import make_rng

rngs = make_rng(seed=0, shape=(4,))
print(rngs)                       # four independent, non-zero 32-bit states
state = octorch.create_state(rngs, device="cpu")
print(tuple(state.batch_shape))   # (4,)
```

## Loading and Running Programs

Every CHIP-8 game is stored as a ROM file—essentially a sequence of bytes that represent instructions and data. Let's see how to load and run a simple program:

```python
import octorch

# Start with a fresh state
state = octorch.create_state(device="cpu")

# Load a ROM into memory
state = octorch.load_rom(state, "roms/Pong (1 player).ch8")

print(f"ROM loaded! First few bytes: {state.memory[0x200:0x210].tolist()}")
print(f"Program counter: 0x{int(state.pc):03X}")
```

CHIP-8 programs always load at address 0x200 (512 in decimal). The first 512 bytes are reserved for the interpreter and font data. This is why `state.pc` starts at 0x200—it's pointing to the first instruction of your program.

## The Execution Cycle

The heart of any emulator is the execution cycle: fetch, decode, execute, repeat. Let's break this down:

```python
def emulate_steps(state, num_steps):
    """Run the emulator for a specific number of steps"""
    for _ in range(num_steps):
        # 1. Fetch: Get the instruction at the current program counter
        state, instruction_word = octorch.fetch(state)

        # 2. Decode: Figure out what this instruction means
        decoded = octorch.decode(instruction_word)

        # 3. Execute: Carry out the instruction
        state = octorch.execute(state, instruction_word)

        print(f"PC: 0x{int(state.pc):03X}, Instruction: 0x{int(instruction_word):04X}, "
              f"Opcode: {int(decoded.opcode)}")
    return state

# Run a few steps to see what happens
state = emulate_steps(state, 5)
```

This three-step cycle is fundamental to computer emulation. The fetch step reads the instruction from memory and advances the program counter. The decode step interprets the raw bytes as a specific operation. The execute step actually performs that operation, potentially changing registers, memory, or display. `octorch.run_instruction` / `octorch.run_n_instruction` bundle the cycle for convenience.

### How execution is batched

Octax dispatches on the opcode with `jax.lax.switch`; under `jax.vmap`, XLA evaluates every branch and selects the right one per lane. Octorch does the same thing explicitly: `execute` computes the effect of *all* opcode families for the whole batch and merges them with masks. A batch of 8192 machines therefore executes an instruction each in a fixed number of tensor operations, whatever mix of opcodes they are running. The price is that a single unbatched machine pays the same cost as a batch, which is why the package also ships a plain-Python reference interpreter:

```python
from octorch.interpreter import ScalarChip8

machine = ScalarChip8(octorch.load_rom(octorch.create_state(device="cpu"), "roms/Pong (1 player).ch8"))
machine.run(1000)                      # ~100x faster than the tensor engine for one machine
print(f"PC after 1000 instructions: 0x{machine.pc:03X}")

# Convert back to a tensor state
state = machine.to_state(octorch.create_state(device="cpu"))
```

`ScalarChip8` implements exactly the same semantics (the test-suite checks both engines against each other on random programs), and `run_n_instruction` uses it automatically for unbatched states, for instance when environments run their start-up instructions.

## Understanding Instructions

CHIP-8 has 35 different instruction types, each with its own behavior. Let's explore a few common ones:

```python
def analyze_instruction(instruction_word):
    d = octorch.decode(instruction_word)
    opcode, x, y, n, nn, nnn = (int(v) for v in (d.opcode, d.x, d.y, d.n, d.nn, d.nnn))

    if opcode == 0x6:  # Load immediate
        print(f"LD V{x:X}, 0x{nn:02X} - Load {nn} into register V{x:X}")
    elif opcode == 0x8 and n == 0x0:  # Copy register
        print(f"LD V{x:X}, V{y:X} - Copy V{y:X} to V{x:X}")
    elif opcode == 0x8 and n == 0x4:  # Add registers
        print(f"ADD V{x:X}, V{y:X} - Add V{y:X} to V{x:X}")
    elif opcode == 0xA:  # Set index register
        print(f"LD I, 0x{nnn:03X} - Set index register to 0x{nnn:03X}")
    elif opcode == 0xD:  # Draw sprite
        print(f"DRW V{x:X}, V{y:X}, {n} - Draw sprite at (V{x:X}, V{y:X})")
    else:
        print(f"Unknown or complex instruction: 0x{instruction_word:04X}")

analyze_instruction(0x6A42)  # Load 0x42 into VA
analyze_instruction(0x8AB0)  # Copy VB to VA
analyze_instruction(0x8AB4)  # Add VB to VA
analyze_instruction(0xA123)  # Set I to 0x123
analyze_instruction(0xDAB5)  # Draw 5-byte sprite at (VA, VB)
```

`decode` returns tensors (so that it works on a batch of instructions); `int(...)` converts the scalar case. The instruction format follows consistent patterns:
- `6XNN`: Load immediate value NN into register VX
- `8XY0`: Copy register VY to register VX
- `ANNN`: Set the index register I to address NNN
- `DXYN`: Draw N-byte sprite from memory[I] at coordinates (VX, VY)

## Working with Registers and Memory

The CHIP-8 has 16 8-bit registers (V0 through VF) and one 16-bit index register (I). Register VF is special—it's used for flags like carry bits and collision detection:

```python
from octorch.ops import put

state = octorch.create_state(device="cpu")

# Functional update of a register (what the 6XNN instruction does)
state = state.replace(V=put(state.V, 0, 100))   # V0 = 100
state = state.replace(V=put(state.V, 1, 50))    # V1 = 50
print(f"V0: {int(state.V[0])}, V1: {int(state.V[1])}")

# Or simply run the instructions
state = octorch.execute(state, 0x8014)          # V0 += V1
print(f"V0 after ADD: {int(state.V[0])}, VF (carry): {int(state.V[15])}")

# The index register is used for memory operations
state = octorch.execute(state, 0xA300)          # I = 0x300
print(f"Index register I: 0x{int(state.I):03X}")
print(f"Memory at I: {int(state.memory[int(state.I)])}")
```

Tensors inside a state can also be mutated in place (`state.V[0] = 100`); this is what the environment does internally for speed, but it modifies every state that shares those tensors.

## The Display System

One of the most interesting aspects of CHIP-8 is its display system. Instead of setting individual pixels, you draw sprites using XOR operations:

```python
import torch

state = octorch.create_state(device="cpu")
print(f"Display shape: {tuple(state.display.shape)}")       # (64, 32): indexed [x, y]
print(f"Display is all off initially: {not bool(state.display.any())}")
print(f"Display type: {state.display.dtype}")

# Draw the font glyph '0' at (10, 5): FX29 sets I to the glyph, DXYN draws 5 rows
state = octorch.execute(state, 0x6000)     # V0 = 0 (the digit)
state = octorch.execute(state, 0xF029)     # I = address of glyph for V0
state = octorch.execute(state, 0x610A)     # V1 = 10
state = octorch.execute(state, 0x6205)     # V2 = 5
state = octorch.execute(state, 0xD125)     # draw at (V1, V2), 5 rows
print(f"Pixels on: {int(state.display.sum())}, collision flag VF: {int(state.V[15])}")

# Drawing the same sprite again erases it (XOR) and sets the collision flag
state = octorch.execute(state, 0xD125)
print(f"Pixels on: {int(state.display.sum())}, collision flag VF: {int(state.V[15])}")
```

The XOR drawing system enables sprite animation (draw, move, erase, redraw) and collision detection (XOR turning a pixel off indicates collision). Note the `[x, y]` indexing convention (width first) inherited from Octax; `octorch.rendering.chip8_display_to_rgb` transposes it to the usual image layout.

## Timers and Game Flow

CHIP-8 has two timers that count down at 60 Hz:

```python
state = octorch.create_state(device="cpu")
print(f"Delay timer: {int(state.delay_timer)}, sound timer: {int(state.sound_timer)}")

# Games set these timers with FX15 / FX18; setting delay_timer = 60 creates a 1-second delay
state = octorch.execute(state, 0x603C)     # V0 = 60
state = octorch.execute(state, 0xF015)     # delay timer = V0
state = octorch.execute(state, 0x601E)     # V0 = 30
state = octorch.execute(state, 0xF018)     # sound timer = V0
print(f"Timers set - Delay: {int(state.delay_timer)}, Sound: {int(state.sound_timer)}")
```

The delay timer is used for game timing (like waiting between moves in Tetris), while the sound timer controls a simple beep sound. The emulator core never decrements them; the environment does it once per step (unless the game definition sets `disable_delay = True`, in which case both timers are forced to zero).

**Octax compatibility quirk.** Octax decrements its `uint8` timers with `maximum(timer - 1, 0)`, so a timer at 0 wraps to 255 and games with timers enabled see a 255-count cycle. Octorch reproduces this by default so that results stay comparable with the Octax paper; pass `timer_wraparound=False` to `create_environment` / `OctorchEnv` to clamp at zero like real CHIP-8 hardware.

## Font Data and Text Rendering

CHIP-8 includes a built-in font for hexadecimal digits (0-F). Let's see how this works:

```python
from octorch.constants import FONT_START

# The font data is loaded at startup
state = octorch.create_state(device="cpu")

# Each character is 5 bytes tall and 4 pixels wide. Let's look at the digit '0'
font_0 = state.memory[FONT_START:FONT_START + 5].tolist()
print(f"Font data for '0': {font_0}")

# Convert to binary to see the pattern
for i, byte in enumerate(font_0):
    binary = format(byte, '08b')[:4]  # Only 4 bits are used
    print(f"Row {i}: {binary.replace('0', ' ').replace('1', '█')}")
```

This font system allows games to display scores and text. The index register is set to point to the desired character (`FX29`), then a draw instruction renders it to the screen.

## Putting It All Together

Now let's create a simple program that demonstrates these concepts:

```python
import torch

def step_by_step_execution(state, num_steps=10):
    """Execute instructions one by one with detailed output"""
    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")
        print(f"PC: 0x{int(state.pc):03X}")

        if int(state.pc) >= state.memory.shape[-1] - 1:
            print("End of memory reached!")
            break

        # Fetch
        old_pc = int(state.pc)
        state, instruction_word = octorch.fetch(state)
        print(f"Fetched: 0x{int(instruction_word):04X} (PC: 0x{old_pc:03X} -> 0x{int(state.pc):03X})")

        # Decode
        d = octorch.decode(instruction_word)
        print(f"Decoded - Opcode: 0x{int(d.opcode):X}, x: {int(d.x)}, y: {int(d.y)}, n: {int(d.n)}")

        # Execute
        old_registers = state.V.clone()
        state = octorch.execute(state, instruction_word)

        # Show what changed
        for reg in torch.nonzero(old_registers != state.V).flatten().tolist():
            print(f"V{reg:X}: {int(old_registers[reg])} -> {int(state.V[reg])}")
    return state

# Load a ROM and watch it execute
state = octorch.load_rom(octorch.create_state(device="cpu"), "roms/test_opcode.ch8")
final_state = step_by_step_execution(state, 5)
```

## Modern vs Legacy Mode

Octorch supports both modern and legacy CHIP-8 behavior. Some instructions behave differently between the original CHIP-8 interpreter and later implementations:

| Instruction | Modern mode (default) | Legacy mode |
|---|---|---|
| `8XY6` / `8XYE` (shifts) | shift VX in place, VY ignored | VX = VY shifted |
| `BNNN` / `BXNN` (jump with offset) | jump to `XNN + VX` | jump to `NNN + V0` |
| `FX55` / `FX65` (store / load registers) | I unchanged | I += X + 1 |

The mode is a static flag on the state:

```python
modern = octorch.create_state(device="cpu")
legacy = modern.replace(modern_mode=False)

for label, s in (("modern", modern), ("legacy", legacy)):
    s = octorch.execute(s, 0x6108)      # V1 = 8
    s = octorch.execute(s, 0x6203)      # V2 = 3
    s = octorch.execute(s, 0x8126)      # V1 >>= 1 (modern) / V1 = V2 >> 1 (legacy)
    print(f"{label}: V1 = {int(s.V[1])}, VF = {int(s.V[15])}")
```

All shipped games run in modern mode, which is what Octax uses; `create_environment(..., modern_mode=False)` switches a whole environment to legacy behaviour.
