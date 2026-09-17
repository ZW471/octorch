"""Vectorised environment wrappers.

Octax ships a Gymnax wrapper; the PyTorch equivalent is a Gymnasium-style
*vector* environment operating on batched torch tensors, with automatic reset
of finished episodes.  Observations are transposed to ``(batch, frame_skip, 32, 64)``
(height before width) exactly like ``OctaxGymnaxWrapper``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from octorch import prng
from octorch.env import OctorchEnv, OctorchEnvState

Tensor = torch.Tensor


class OctorchVectorEnv:
    """Batched, auto-resetting environment with torch tensor I/O.

    Args:
        octorch_env: configured :class:`OctorchEnv`
        num_envs: number of parallel environments
        seed: base seed
        auto_reset: reset finished environments at the end of ``step`` (the
            observation returned for those environments is the first observation
            of the new episode, and ``info["final_observation"]`` holds the last
            one of the finished episode)
        obs_dtype: dtype of returned observations (``torch.float32`` or ``torch.bool``)
    """

    def __init__(
        self,
        octorch_env: OctorchEnv,
        num_envs: int,
        seed: int = 0,
        auto_reset: bool = True,
        obs_dtype: torch.dtype = torch.float32,
    ):
        self._env = octorch_env
        self.num_envs = num_envs
        self.seed = seed
        self.auto_reset = auto_reset
        self.obs_dtype = obs_dtype
        self.device = octorch_env.device
        self.state: Optional[OctorchEnvState] = None
        self._rng = prng.make_rng(seed, (num_envs,), self.device)

    # ------------------------------------------------------------------
    @property
    def env(self) -> OctorchEnv:
        return self._env

    @property
    def name(self) -> str:
        return f"Octorch_{self._env.rom_path.split('/')[-1].split('.')[0]}"

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def single_observation_shape(self) -> tuple:
        return (self._env.frame_skip, 32, 64)

    @property
    def single_action_space(self):
        from gymnasium.spaces import Discrete
        return Discrete(self.num_actions)

    @property
    def single_observation_space(self):
        from gymnasium.spaces import Box
        return Box(low=0.0, high=1.0, shape=self.single_observation_shape, dtype=np.float32)

    @property
    def max_steps_in_episode(self) -> int:
        return self._env.max_num_steps_per_episodes

    def _obs(self, obs: Tensor) -> Tensor:
        return obs.transpose(-1, -2).to(self.obs_dtype)

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        """Reset all environments. Returns ``(obs, info)``."""
        if seed is not None:
            self.seed = seed
            self._rng = prng.make_rng(seed, (self.num_envs,), self.device)
        self._rng = prng.split(self._rng, 1)[0]
        self.state, obs, info = self._env.reset(self._rng)
        static = self._env.static_state(self.num_envs)
        if static is not None:  # compiled step: work directly on its static buffers
            static.copy_(self.state)
            self.state = static
        return self._obs(obs), info

    def step(self, action: Union[Tensor, np.ndarray, int]):
        """Step all environments. Returns ``(obs, reward, terminated, truncated, info)``."""
        action = torch.as_tensor(action, device=self.device, dtype=torch.int64)
        self.state, obs, reward, terminated, truncated, info = self._env.step_(self.state, action)
        obs = self._obs(obs)
        if self.auto_reset:
            done = terminated | truncated
            info["final_observation"] = obs
            info["done"] = done
            self._rng = prng.split(self._rng, 1)[0]
            fresh, fresh_obs, _ = self._env.reset(self._rng)
            self.state.masked_copy_(done, fresh)
            obs = torch.where(done[:, None, None, None], self._obs(fresh_obs), obs)
        return obs, reward, terminated, truncated, info

    def render(self, index: int = 0):
        return self._env.render(self.state[index])


OctaxGymnaxWrapper = OctorchVectorEnv  # name kept for discoverability


def make_gymnasium_env(octorch_env: OctorchEnv, num_envs: int, seed: int = 0):
    """Wrap into a ``gymnasium.vector.VectorEnv`` subclass with numpy I/O (requires gymnasium)."""
    import gymnasium as gym

    class _GymVectorEnv(gym.vector.VectorEnv):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self):
            self.inner = OctorchVectorEnv(octorch_env, num_envs, seed)
            self.num_envs = num_envs
            self.single_observation_space = self.inner.single_observation_space
            self.single_action_space = self.inner.single_action_space
            self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

        def reset(self, *, seed=None, options=None):
            obs, info = self.inner.reset(seed)
            return obs.cpu().numpy(), {k: v.cpu().numpy() for k, v in info.items()}

        def step(self, actions):
            obs, r, te, tr, info = self.inner.step(np.asarray(actions))
            return (obs.cpu().numpy(), r.cpu().numpy(), te.cpu().numpy(), tr.cpu().numpy(),
                    {k: v.cpu().numpy() for k, v in info.items()})

        def render(self):
            return self.inner.render(0)

    return _GymVectorEnv()
