"""Shared building blocks for the Octorch reference agents."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import torch
from torch import nn

Tensor = torch.Tensor


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class ConvFeatures(nn.Module):
    """Convolutional torso used by both PPO and PQN agents.

    Mirrors the Octax agents (three convolutions ``(32, 64, 64)`` with kernels
    ``((8, 4), 4, 3)`` and strides ``((4, 2), 2, 1)``) applied to the
    ``(frame_skip, 32, 64)`` observation, frames as input channels.
    """

    def __init__(
        self,
        in_channels: int,
        obs_hw: Tuple[int, int] = (32, 64),
        features_list: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence = ((8, 4), 4, 3),
        strides_list: Sequence = ((4, 2), 2, 1),
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()
        layers = []
        c = in_channels
        for features, kernel_size, strides in zip(features_list, kernel_sizes, strides_list):
            k, s = _pair(kernel_size), _pair(strides)
            layers.append(nn.Conv2d(c, features, kernel_size=k, stride=s, padding=(k[0] // 2, k[1] // 2)))
            layers.append(activation())
            c = features
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, in_channels, *obs_hw)).flatten(1).shape[1]

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs.to(torch.float32)).flatten(1)


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation=nn.ReLU, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def get_activation(name: Union[str, Callable, None]) -> Callable[[], nn.Module]:
    if name is None:
        return nn.ReLU
    if callable(name):
        return name
    table = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU}
    return table[str(name).lower()]


def evaluate(act: Callable[[Tensor], Tensor], env, num_episodes: int, max_steps: int, seed: int = 0):
    """Roll out ``num_episodes`` episodes in parallel with a deterministic policy.

    ``act(obs) -> action`` receives batched observations ``(N, C, H, W)``.
    Returns ``(lengths, returns)`` tensors of shape ``(num_episodes,)``.
    """
    from octorch import prng
    base = env.env if hasattr(env, "env") else env
    rng = prng.make_rng(seed + 1_000_003, (num_episodes,), base.device)
    state, obs, _ = base.reset(rng)
    static = base.static_state(num_episodes)
    if static is not None:  # compiled step available for this batch size: use it without copies
        static.copy_(state)
        state = static
    obs = obs.transpose(-1, -2)
    done = torch.zeros(num_episodes, dtype=torch.bool, device=base.device)
    lengths = torch.zeros(num_episodes, dtype=torch.int64, device=base.device)
    returns = torch.zeros(num_episodes, dtype=torch.float32, device=base.device)
    for _ in range(max_steps):
        with torch.no_grad():
            action = act(obs)
        state, obs, reward, terminated, truncated, _ = base.step_(state, action)
        obs = obs.transpose(-1, -2)
        returns += reward * (~done)
        lengths += (~done).to(torch.int64)
        done |= terminated | truncated
        if bool(done.all()):
            break
    return lengths, returns
