"""CHIP-8 reinforcement learning environment (PyTorch port of ``octax.env``)."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

from octorch import prng
from octorch.constants import PROGRAM_START
from octorch.emulator import fetch_, execute_, run_n_instruction, load_rom_bytes
from octorch.ops import put, put_, bmask
from octorch.rendering import chip8_display_to_rgb, create_color_scheme
from octorch.state import EmulatorState, create_state
from octorch.struct import tree_map

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class OctorchEnvState(EmulatorState):
    """Extended emulator state with environment-specific tracking.

    Attributes:
        time: ``(...)`` int32 current timestep in the episode
        previous_score: ``(...)`` float32 score from the previous step (for reward calculation)
        current_score: ``(...)`` float32 current score in the episode
    """
    time: Tensor = None
    previous_score: Tensor = None
    current_score: Tensor = None


OctaxEnvState = OctorchEnvState  # alias for Octax API compatibility


def asdict_non_recursive(obj) -> dict:
    """Convert dataclass to dictionary without recursive conversion."""
    return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}


def run_instruction(state: EmulatorState) -> EmulatorState:
    """Fetch-decode-execute one instruction in place (Octax ``run_instruction`` equivalent)."""
    state, instruction = fetch_(state)
    return execute_(state, instruction)


class OctorchEnv:
    """PyTorch CHIP-8 environment for reinforcement learning.

    Provides an OpenAI Gym-style ``reset`` / ``step`` interface for CHIP-8 games.
    Unlike Octax, batching is explicit: ``reset`` takes a batch size (or a batch
    of PRNG states) and ``step`` advances every environment of the batch with one
    action per environment.  A batch size of ``None`` gives the unbatched
    single-environment behaviour of Octax.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        max_num_steps_per_episodes: int = 4500,
        instruction_frequency: int = 700,
        fps: int = 60,
        frame_skip: int = 4,
        action_set=None,
        score_fn: Callable[[EmulatorState], Union[float, Tensor]] = lambda _: 0.0,
        terminated_fn: Callable[[EmulatorState], Union[bool, Tensor]] = lambda _: False,
        startup_instructions: int = 0,
        custom_startup: Optional[Callable[[EmulatorState], EmulatorState]] = None,
        render_mode: Optional[str] = "rgb_array",
        disable_delay: bool = True,
        render_scale: int = 8,
        color_scheme: str = "classic",
        device: Union[str, torch.device, None] = None,
        modern_mode: bool = True,
        timer_wraparound: bool = True,
    ):
        """Initialize the CHIP-8 RL environment.

        Args:
            rom_path: Path to the CHIP-8 ROM file to load
            max_num_steps_per_episodes: Maximum steps before episode truncation
            instruction_frequency: CHIP-8 CPU frequency in Hz (typically 700)
            fps: Environment frame rate (typically 60)
            frame_skip: Number of frames to skip between observations
            action_set: List/array of valid CHIP-8 key indices (0-15). If None, uses all 16 keys
            score_fn: Function to extract score from (batched) emulator state
            terminated_fn: Function to detect episode termination from (batched) emulator state
            startup_instructions: Number of instructions to run during reset to skip ROM initialization
            custom_startup: Custom startup function to run after ROM loading
            render_mode: Rendering mode ("rgb_array" or None)
            disable_delay: Whether to disable delay and sound timers for faster execution
            render_scale: Upscaling factor for rendered frames (default: 8x)
            color_scheme: Color scheme for rendering ("octax", "classic", "amber", "white", "blue", "retro")
            device: torch device the environment lives on (default: CUDA if available)
            modern_mode: CHIP-8 quirk mode
            timer_wraparound: Octax (0.1.1) decrements its uint8 timers with
                ``maximum(timer - 1, 0)``, so a timer at 0 wraps to 255. ``True``
                (default) reproduces this bit-exactly; ``False`` clamps at 0 like
                real CHIP-8 hardware.
        """
        self.rom_path = rom_path
        self.max_num_steps_per_episodes = max_num_steps_per_episodes
        self.instruction_frequency = instruction_frequency
        self.fps = fps
        self.frame_skip = frame_skip
        self.terminated_fn = terminated_fn
        self.score_fn = score_fn
        self.startup_instructions = startup_instructions
        self.custom_startup = custom_startup
        self.disable_delay = disable_delay
        self.modern_mode = modern_mode
        self.timer_wraparound = timer_wraparound

        self.render_mode = render_mode
        self.render_scale = render_scale
        self.color_scheme = color_scheme

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if action_set is None:
            action_set = range(16)
        self.action_set = torch.as_tensor(list(action_set), dtype=torch.int64, device=self.device)

        with open(rom_path, "rb") as f:
            self.rom_data = f.read()

        self.cached_reset_state: OctorchEnvState = self._reset()
        self._compiled: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def instructions_per_step(self) -> int:
        """Number of CHIP-8 instructions to execute per frame (frequency // fps)."""
        return self.instruction_frequency // self.fps

    @property
    def num_actions(self) -> int:
        """Number of actions (length of action_set + 1 for no-op)."""
        return len(self.action_set) + 1

    @property
    def observation_shape(self) -> tuple:
        return (self.frame_skip, 64, 32)

    def from_minutes(self, minutes: float):
        """Set episode length based on desired gameplay duration in real-world minutes."""
        self.max_num_steps_per_episodes = int(minutes * 60 * self.fps) // self.frame_skip

    def pad_frame(self, display: Tensor) -> Tensor:
        """Pad frame(s) ``(..., 64, 32)`` into a ``(..., frame_skip, 64, 32)`` observation.

        Previous frames are zero-filled, the current frame is last.
        """
        lead = display.shape[:-2]
        zeros = torch.zeros(lead + (self.frame_skip - 1,) + display.shape[-2:], dtype=display.dtype, device=display.device)
        return torch.cat((zeros, display.unsqueeze(-3)), dim=-3)

    def _score(self, state: EmulatorState) -> Tensor:
        score = self.score_fn(state)
        score = torch.as_tensor(score, device=self.device)
        return torch.broadcast_to(score, state.pc.shape).to(torch.float32)

    def _terminated(self, state: EmulatorState) -> Tensor:
        term = torch.as_tensor(self.terminated_fn(state), device=self.device)
        return torch.broadcast_to(term, state.pc.shape).to(torch.bool)

    def _reset(self) -> OctorchEnvState:
        state = create_state(0, device=self.device, modern_mode=self.modern_mode)
        state = load_rom_bytes(state, self.rom_data)

        if self.custom_startup:
            state = self.custom_startup(state)
        elif self.startup_instructions > 0:
            state = run_n_instruction(state, self.startup_instructions)

        initial_score = self._score(state)
        return OctorchEnvState(
            **asdict_non_recursive(state),
            time=torch.zeros((), dtype=torch.int32, device=self.device),
            current_score=initial_score.clone(),
            previous_score=initial_score.clone(),
        )

    # ------------------------------------------------------------------ API
    def reset(
        self,
        rng: Union[int, Tensor, None] = 0,
        batch_size: Union[int, Sequence[int], None] = None,
    ):
        """Reset the environment to its initial state.

        Args:
            rng: integer seed, or a tensor of PRNG states (see :func:`octorch.make_rng`)
                whose shape defines the batch shape.
            batch_size: batch size / shape when ``rng`` is an integer seed.
                ``None`` returns an unbatched (single) environment state.

        Returns:
            ``(state, observation, info)`` where observation has shape
            ``(*batch, frame_skip, 64, 32)`` and ``info = {"score": ...}``.
        """
        if isinstance(rng, torch.Tensor) and rng.numel() > 1:
            batch_shape = tuple(rng.shape)
            rng = rng.to(device=self.device, dtype=torch.int64)
        else:
            if batch_size is None:
                batch_shape = ()
            elif isinstance(batch_size, int):
                batch_shape = (batch_size,)
            else:
                batch_shape = tuple(batch_size)
            rng = prng.make_rng(0 if rng is None else rng, batch_shape, self.device)

        base = self.cached_reset_state
        state = base.expand(batch_shape) if batch_shape else base.clone()
        state = state.replace(rng=rng)
        return state, self.pad_frame(state.display), {"score": state.current_score.clone()}

    def reset_state(self, rng: Tensor) -> OctorchEnvState:
        """Reset state only, batch shape given by ``rng``."""
        return self.reset(rng)[0]

    def step(self, state: OctorchEnvState, action: Union[int, Tensor]):
        """Execute one environment step for every environment in the batch.

        Presses the specified key, runs CHIP-8 instructions for ``frame_skip`` frames,
        updates timers, calculates rewards, and checks termination.

        Args:
            state: Current (possibly batched) environment state; it is not modified.
            action: int or int tensor of batch shape. Values ``0..len(action_set)-1``
                press the corresponding key, ``len(action_set)`` is the no-op.

        Returns:
            ``(next_state, observation, reward, terminated, truncated, info)``
        """
        state = state.clone()
        return self.step_(state, action)

    def step_(self, state: OctorchEnvState, action: Union[int, Tensor]):
        """In-place variant of :meth:`step`: ``state``'s tensors are updated and returned."""
        compiled = self._compiled.get(tuple(state.pc.shape))
        if compiled is not None:
            return compiled(state, action)
        return self._step_eager(state, action)

    def _step_eager(self, state: OctorchEnvState, action: Union[int, Tensor]):
        batch_shape = state.pc.shape
        action = torch.as_tensor(action, device=self.device, dtype=torch.int64)
        action = torch.broadcast_to(action, batch_shape)
        is_noop = action == (self.num_actions - 1)
        key = self.action_set[action.clamp(0, len(self.action_set) - 1)]

        # Press the key (no-op leaves the keypad untouched)
        keypad = state.keypad
        pressed = torch.where(is_noop, keypad.gather(-1, key.unsqueeze(-1)).squeeze(-1), torch.ones_like(is_noop))
        put_(keypad, key, pressed)

        ipf = self.instructions_per_step
        frames = []
        for i in range(ipf * self.frame_skip):
            run_instruction(state)
            if i % ipf == ipf - 1:
                frames.append(state.display.clone())
        observation = torch.stack(frames, dim=-3)

        if self.disable_delay:
            state.delay_timer.zero_()
            state.sound_timer.zero_()
        elif self.timer_wraparound:
            state.delay_timer.sub_(1)  # uint8: 0 wraps to 255, exactly like Octax
            state.sound_timer.sub_(1)
        else:
            state.delay_timer.copy_((state.delay_timer.to(torch.int32) - 1).clamp(min=0).to(torch.uint8))
            state.sound_timer.copy_((state.sound_timer.to(torch.int32) - 1).clamp(min=0).to(torch.uint8))

        # Release the key
        released = torch.where(is_noop, keypad.gather(-1, key.unsqueeze(-1)).squeeze(-1), torch.zeros_like(is_noop))
        put_(keypad, key, released)

        previous_score = state.previous_score
        current_score = self._score(state)
        reward = current_score - previous_score

        state.current_score.copy_(current_score)
        state.previous_score.copy_(current_score)
        state.time.add_(1)

        terminated = self._terminated(state)
        truncated = state.time >= self.max_num_steps_per_episodes
        return state, observation, reward, terminated, truncated, {"score": state.current_score.clone()}

    # ------------------------------------------------------------------ compilation
    def compile(
        self,
        batch_size: Union[int, Sequence[int]],
        granularity: str = "frame",
        cudagraph: Optional[bool] = None,
        mode: Optional[str] = None,
        warmup: int = 3,
    ) -> "CompiledStep":
        """Compile ``step`` for a fixed batch size with ``torch.compile`` (+ CUDA graph).

        After calling this, :meth:`step` / :meth:`step_` transparently use the
        compiled version whenever the state has the compiled batch shape.  The
        compiled step runs on static buffers; :meth:`static_state` returns them so
        callers that hold their own state (e.g. :class:`octorch.wrappers.OctorchVectorEnv`)
        can avoid any copies.

        Args:
            batch_size: batch size (or shape) to compile for.
            granularity: what ``torch.compile`` sees as one graph – ``"instruction"``
                (fast compile, most kernels), ``"frame"`` (``instructions_per_step``
                instructions; good trade-off, default) or ``"step"`` (whole step; best
                throughput, slowest compile).
            cudagraph: capture the whole step into a CUDA graph (default: on CUDA).
            mode: ``torch.compile`` mode (e.g. ``"max-autotune-no-cudagraphs"``).
            warmup: warm-up iterations before graph capture.
        """
        batch_shape = (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)
        if cudagraph is None:
            cudagraph = self.device.type == "cuda"
        compiled = CompiledStep(self, batch_shape, granularity, cudagraph, mode, warmup)
        self._compiled[batch_shape] = compiled
        return compiled

    def static_state(self, batch_size: Union[int, Sequence[int]]) -> Optional[OctorchEnvState]:
        """The static state buffers of the compiled step for ``batch_size`` (if compiled)."""
        batch_shape = (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)
        compiled = self._compiled.get(batch_shape)
        return compiled.state if compiled is not None else None

    def reset_where(self, state: OctorchEnvState, mask: Tensor, rng: Optional[Tensor] = None) -> OctorchEnvState:
        """Return a copy of ``state`` where environments with ``mask`` set are reset.

        ``rng`` optionally provides fresh PRNG states (batch shape); by default the
        existing PRNG state of each environment is re-seeded with ``prng.split``.
        """
        if rng is None:
            rng = prng.split(state.rng, 1)[0]
        fresh, _, _ = self.reset(rng)
        state = state.clone()
        state.masked_copy_(mask, fresh)
        return state

    def render(self, state: OctorchEnvState) -> Optional[np.ndarray]:
        """Render the current environment state (first environment if batched).

        Returns:
            RGB array of shape (height, width, 3) if render_mode="rgb_array", else None
        """
        if self.render_mode == "rgb_array":
            on_color, off_color = create_color_scheme(self.color_scheme)
            display = state.display
            while display.dim() > 2:
                display = display[0]
            return chip8_display_to_rgb(display, scale=self.render_scale, on_color=on_color, off_color=off_color)
        return None


class CompiledStep:
    """``torch.compile`` + CUDA-graph version of :meth:`OctorchEnv.step_` for one batch shape."""

    def __init__(self, env: OctorchEnv, batch_shape: tuple, granularity: str, cudagraph: bool,
                 mode: Optional[str], warmup: int):
        if granularity not in ("instruction", "frame", "step"):
            raise ValueError(f"granularity must be 'instruction', 'frame' or 'step', got {granularity!r}")
        self.env = env
        self.batch_shape = batch_shape
        self.granularity = granularity
        self.cudagraph = cudagraph
        self.state: OctorchEnvState = env.reset(0, batch_size=batch_shape)[0]
        self.action = torch.zeros(batch_shape, dtype=torch.int64, device=env.device)
        self.graph = None
        self._build(mode, warmup)

    # -- pieces ----------------------------------------------------------------
    def _press(self, state, action):
        is_noop = action == (self.env.num_actions - 1)
        key = self.env.action_set[action.clamp(0, len(self.env.action_set) - 1)]
        keypad = state.keypad
        pressed = torch.where(is_noop, keypad.gather(-1, key.unsqueeze(-1)).squeeze(-1), torch.ones_like(is_noop))
        put_(keypad, key, pressed)
        return is_noop, key

    def _release(self, state, is_noop, key):
        keypad = state.keypad
        released = torch.where(is_noop, keypad.gather(-1, key.unsqueeze(-1)).squeeze(-1), torch.zeros_like(is_noop))
        put_(keypad, key, released)

    def _frame(self, state):
        for _ in range(self.env.instructions_per_step):
            run_instruction(state)
        return state.display.clone()

    def _finish(self, state):
        env = self.env
        if env.disable_delay:
            state.delay_timer.zero_()
            state.sound_timer.zero_()
        elif env.timer_wraparound:
            state.delay_timer.sub_(1)
            state.sound_timer.sub_(1)
        else:
            state.delay_timer.copy_((state.delay_timer.to(torch.int32) - 1).clamp(min=0).to(torch.uint8))
            state.sound_timer.copy_((state.sound_timer.to(torch.int32) - 1).clamp(min=0).to(torch.uint8))

    def _body(self, state, action):
        """Full step on ``state`` in place; returns ``(obs, reward, terminated, truncated, score)``."""
        env = self.env
        is_noop, key = self._press(state, action)
        frames = [self._frame_fn(state) for _ in range(env.frame_skip)]
        observation = torch.stack(frames, dim=-3)
        self._finish(state)
        self._release(state, is_noop, key)
        current_score = env._score(state)
        reward = current_score - state.previous_score
        state.current_score.copy_(current_score)
        state.previous_score.copy_(current_score)
        state.time.add_(1)
        terminated = env._terminated(state)
        truncated = state.time >= env.max_num_steps_per_episodes
        return observation, reward, terminated, truncated, state.current_score.clone()

    def _build(self, mode, warmup):
        opts = dict(fullgraph=True, dynamic=False)
        if mode is not None:
            opts["mode"] = mode
        if self.granularity == "instruction":
            cins = torch.compile(run_instruction, **opts)

            def frame(state):
                for _ in range(self.env.instructions_per_step):
                    cins(state)
                return state.display.clone()
            self._frame_fn = frame
            body = self._body
        elif self.granularity == "frame":
            self._frame_fn = torch.compile(self._frame, **opts)
            body = self._body
        else:
            self._frame_fn = self._frame
            body = torch.compile(self._body, **opts)
        self._run = body

        if not self.cudagraph:
            for _ in range(warmup):
                self._run(self.state, self.action)
            return
        stream = torch.cuda.Stream(self.env.device)
        stream.wait_stream(torch.cuda.current_stream(self.env.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._run(self.state, self.action)
        torch.cuda.current_stream(self.env.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = self._run(self.state, self.action)

    # -- call ------------------------------------------------------------------
    def __call__(self, state: OctorchEnvState, action):
        """In-place step of ``state`` (copied through the static buffers if it is not them)."""
        action = torch.broadcast_to(torch.as_tensor(action, device=self.env.device, dtype=torch.int64), self.batch_shape)
        is_static = state.pc.data_ptr() == self.state.pc.data_ptr()
        if not is_static:
            self.state.copy_(state)
        self.action.copy_(action)
        if self.graph is not None:
            self.graph.replay()
            outputs = self.outputs
        else:
            outputs = self._run(self.state, self.action)
        if not is_static:
            state.copy_(self.state)
        obs, reward, terminated, truncated, score = (o.clone() for o in outputs)
        return state, obs, reward, terminated, truncated, {"score": score}


OctaxEnv = OctorchEnv  # alias for Octax API compatibility
