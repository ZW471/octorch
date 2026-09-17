# Creating Custom Game Environments

Octorch supports adding any CHIP-8 game as an RL environment. The process has three stages: playing the game to understand its mechanics, identifying the right registers, and writing a short Python module. This tutorial walks through all three.

## Understanding Game Environment Structure

Every Octorch environment is a small Python module inside `octorch/environments/`. Let's read one of the simplest to understand the pattern:

```python
# octorch/environments/missile.py

from octorch import EmulatorState

rom_file = "Missile [David Winter].ch8"

def score_fn(state: EmulatorState):
    return state.V[..., 7]          # Score lives in register V7

def terminated_fn(state: EmulatorState):
    return state.V[..., 6] == 0     # Game over when V6 (missiles) reaches 0

action_set = [8]                    # Only key 8 (fire) is used

startup_instructions = 50           # Skip past the title screen

metadata = {
    "title": "Missile Command",
    "release": "1996",
    "authors": ["David Winter"],
    ...
}
```

That is the complete environment definition. Everything else — step logic, observation stacking, reward calculation, batching, compilation — is handled automatically by `OctorchEnv`.

The one thing to remember: `score_fn` and `terminated_fn` receive a **batched** state, so registers must be indexed with an ellipsis (`state.V[..., 7]`, not `state.V[7]`). The functions then work for a single environment, a batch of 8192, or a `(seeds, envs)` batch alike, and return a tensor with the batch shape.

## Stage 1 — Play the Game

The most reliable way to understand a CHIP-8 game is to run it in Octorch's interactive emulator (`play.py`, needs `uv sync --extra gui`). This displays all 16 registers in real time and prints a `🎯 BCD!` marker whenever a BCD-conversion instruction fires, which is the strongest indicator of a score register.

```bash
uv run python play.py "roms/YourGame.ch8"
```

**Controls in the emulator:**
- `D` — toggle the debug overlay (register values, BCD markers)
- `P` — pause / unpause
- `R` — reset
- `+` / `-` — increase / decrease CPU speed
- `ESC` — quit

While playing, look for:

| Signal | What it means |
|---|---|
| BCD marker `🎯` on register `Vx` | `Vx` holds the displayed score |
| Register that starts high and decreases | Likely a lives / ammo counter |
| Register that only increases | Likely a score or progress counter |
| Register that snaps to `0` or `255` at game over | Likely a game-state flag |

### Headless analysis with the scalar interpreter

Without a display, the plain-Python `ScalarChip8` interpreter is convenient for scripted exploration (it runs about a million instructions per second):

```python
import octorch
from octorch.interpreter import ScalarChip8

state = octorch.load_rom(octorch.create_state(device="cpu"), "roms/UFO [Lutz V, 1992].ch8")
machine = ScalarChip8(state)

seen_bcd = {}
for i in range(300000):
    instruction = machine.fetch()
    if (instruction & 0xF0FF) == 0xF033:          # FX33: BCD of VX -> a displayed number
        register = (instruction & 0x0F00) >> 8
        seen_bcd[register] = seen_bcd.get(register, 0) + 1
    machine.execute(instruction)
    if i % 3000 == 2999:
        machine.keypad[5] = not machine.keypad[5]  # toggle the "fire straight up" key now and then

print(f"Registers converted to BCD: { {f'V{r:X}': n for r, n in seen_bcd.items()} }")
print(f"Register values: {machine.V.tolist()}")
```

For UFO this reports `V7` (the score, drawn on the left) and `V8` (the remaining missiles, drawn on the right), which is exactly the pair used by `octorch/environments/ufo.py`.

## Stage 2 — Identify the Key Registers

Open the debug overlay (`D`) and play until you score points, lose a life, and reach game over. Note down:

1. **Score register** — which `Vx` tracks your score
2. **Termination register** — which `Vx` signals game over and what value it takes
3. **Used keys** — which keys you actually pressed (shown in the overlay)
4. **Startup skip** — how many instructions pass before the game loop begins (skip menus / splash screens)

### Tip: Check the `.8o` Source

If a decompiled source file exists in `roms/`, it provides the definitive register map. Look for:

```
# register comments at the top of the file
:alias score v7
:alias lives v6
```

### Tip: Using `print_all_registers`

Run `play.py` for 30 seconds and wait for the automatic register dump. The registers marked with `📈` (monotonically increasing) are strong score candidates; `📉` (decreasing) are likely countdown timers.

## Stage 3 — Write the Environment Module

Create a new file in `octorch/environments/`:

```python
# octorch/environments/my_game.py

from octorch import EmulatorState

# ── 1. ROM filename ──────────────────────────────────────────────────────────
rom_file = "MyGame.ch8"   # must exist in roms/

# ── 2. Score function ────────────────────────────────────────────────────────
def score_fn(state: EmulatorState):
    """Return the current score from the (batched) emulator state."""
    return state.V[..., 5]          # replace with your score register

# ── 3. Termination function ──────────────────────────────────────────────────
def terminated_fn(state: EmulatorState):
    """Return True where the game is definitively over."""
    return state.V[..., 12] == 0    # replace with your termination condition

# ── 4. Action set ────────────────────────────────────────────────────────────
action_set = [4, 6]                 # CHIP-8 key indices actually used by the game

# ── 5. Startup skip (optional) ───────────────────────────────────────────────
startup_instructions = 500          # instructions to run before the first reset

# ── 6. Timers (optional) ─────────────────────────────────────────────────────
disable_delay = False               # True forces both timers to 0 every step

# ── 7. Metadata ──────────────────────────────────────────────────────────────
metadata = {
    "title": "My Game",
    "description": "Brief description of the game.",
    "release": "2026",
    "authors": ["Your Name"],
    "roms": {
        "sha1hashofrom": {
            "file": "MyGame.ch8",
            "platforms": ["originalChip8"],
        }
    },
}
```

Place the ROM in `roms/MyGame.ch8` and test immediately:

```python
import torch
from octorch.environments import create_environment

env, metadata = create_environment("my_game")

state, obs, info = env.reset(0)
print("Reset OK, obs shape:", tuple(obs.shape))

for _ in range(100):
    action = int(torch.randint(0, env.num_actions, ()))
    state, obs, reward, terminated, truncated, info = env.step(state, action)
    if terminated or truncated:
        print("Episode ended. Score:", float(info["score"]))
        break
```

You can also build the environment without a module, directly from `OctorchEnv`, which is handy while experimenting:

```python
from octorch.env import OctorchEnv
from octorch.environments import get_rom_path

env = OctorchEnv(
    rom_path=get_rom_path("Missile [David Winter].ch8"),
    score_fn=lambda state: state.V[..., 7],
    terminated_fn=lambda state: state.V[..., 6] == 0,
    action_set=[8],
    startup_instructions=50,
    device="cpu",
)
state, obs, info = env.reset(0, batch_size=4)
state, obs, reward, terminated, truncated, info = env.step(state, torch.zeros(4, dtype=torch.long))
print(tuple(reward.shape), info["score"].tolist())
```

## Advanced Patterns

### Multi-component Score

Some games store their score across multiple registers (BCD digits, high/low bytes, etc.). Registers are `uint8` tensors, so convert before arithmetic that may exceed 255:

```python
import torch
from octorch import EmulatorState

def score_fn(state: EmulatorState):
    # Score stored as two decimal digits: V0 = tens, V1 = ones
    return state.V[..., 0].to(torch.float32) * 10.0 + state.V[..., 1].to(torch.float32)
```

### Custom Startup Sequence

If the game requires specific key presses during initialisation (e.g. "press 0 to start"), use `custom_startup`. It receives the unbatched start state; `run_n_instruction` runs it through the fast scalar interpreter:

```python
from octorch import EmulatorState, run_n_instruction
from octorch.ops import put

def custom_startup(state: EmulatorState) -> EmulatorState:
    # Press key 0 to dismiss the title screen
    state = state.replace(keypad=put(state.keypad, 0, True))
    state = run_n_instruction(state, 200)
    state = state.replace(keypad=put(state.keypad, 0, False))
    state = run_n_instruction(state, 50)
    return state
```

Set `custom_startup` in your module and omit `startup_instructions` (a module that defines `custom_startup` ignores `startup_instructions`).

### Protecting Against Score Overflow

CHIP-8 registers are 8-bit (0–255). Games that would overflow often wrap to 0. Clamp the score to avoid spurious negative rewards:

```python
import torch
from octorch import EmulatorState

def score_fn(state: EmulatorState):
    v0 = state.V[..., 0]
    return torch.where(v0 > 200, torch.zeros_like(v0), v0)
```

### Timers

Games that rely on the delay timer (Tetris, Blinky, ...) keep `disable_delay = False`, and the environment decrements both timers once per step. Octorch reproduces Octax's `uint8` wraparound (a timer at 0 becomes 255) by default; pass `timer_wraparound=False` to `create_environment` if you want hardware-like clamping for your game.

### Levelled Games

Cavern, Space Flight and Target Shooter ship multiple ROMs (one per level). `create_environment` handles the naming convention automatically:

```python
from octorch.environments import create_environment

# File naming: <env_id><level>.ch8  ->  env_id = "cavern", module = cavern.py
env, _ = create_environment("cavern3")         # loads cavern3.ch8
env, _ = create_environment("space_flight5")   # loads space_flight5.ch8
```

To support this in your own module, add its base name to the levelled-game list in `octorch/environments/__init__.py` (`create_environment`) and leave `rom_file` undefined; the id suffix selects the ROM.

## Testing Your Environment

Octorch's test suite does not cover custom environments automatically, but you can quickly write a sanity-check:

```python
# tests/test_my_game.py

import torch
from octorch.environments import create_environment

def test_my_game_resets():
    env, _ = create_environment("my_game", device="cpu")
    state, obs, info = env.reset(0)
    assert tuple(obs.shape) == (env.frame_skip, 64, 32)
    assert float(info["score"]) >= 0

def test_my_game_steps_batched():
    env, _ = create_environment("my_game", device="cpu")
    state, obs, info = env.reset(1, batch_size=4)
    for _ in range(50):
        action = torch.randint(0, env.num_actions, (4,))
        state, obs, reward, terminated, truncated, info = env.step(state, action)
    assert tuple(reward.shape) == (4,)

def test_episode_terminates():
    """Episode must terminate within max_steps."""
    env, _ = create_environment("my_game", device="cpu", max_num_steps_per_episodes=500)
    state, obs, _ = env.reset(42)
    done = False
    for _ in range(600):
        action = int(torch.randint(0, env.num_actions, ()))
        state, obs, _, terminated, truncated, _ = env.step(state, action)
        if terminated or truncated:
            done = True
            break
    assert done, "Episode never terminated"
```

## Submitting Your Environment

If you want to contribute your environment back to Octorch:

1. Add your module to `octorch/environments/` and its id to `ENV_IDS`
2. Add the ROM to `roms/` (verify it is public domain or appropriately licensed)
3. Add an entry to `docs/environments/games.md`
4. Write a test in `tests/`
5. Open a pull request with a short description of the game's mechanics
