"""Counter-free, vectorised pseudo random number generator.

Octax threads a ``jax.random.PRNGKey`` through the emulator state so that every
environment instance owns an independent, reproducible random stream.  PyTorch
has no vectorised, functional RNG, so Octorch stores a 32-bit xorshift state per
environment inside the emulator state (as an ``int64`` tensor holding an unsigned
32-bit value) and advances it with pure tensor arithmetic.  This keeps ``step``
free of host synchronisation and CUDA-graph friendly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

Tensor = torch.Tensor
_M32 = 0xFFFFFFFF
_GOLDEN = 0x9E3779B9


def mix32(x: Tensor) -> Tensor:
    """32-bit integer hash (lowbias32) applied element-wise on int64 tensors."""
    x = x & _M32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & _M32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & _M32
    x = x ^ (x >> 16)
    return x


def make_rng(
    seed: Union[int, Tensor] = 0,
    shape: Sequence[int] = (),
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Create RNG state(s) of the given batch ``shape`` from an integer seed.

    Every element receives an independent, non-zero xorshift32 state derived from
    ``seed`` and its flat index, so ``make_rng(seed, (n,))`` plays the role of
    ``jax.random.split(jax.random.PRNGKey(seed), n)``.
    """
    if isinstance(seed, torch.Tensor):
        if seed.numel() == 1:
            seed = int(seed.item())
        else:
            return seed.to(device=device, dtype=torch.int64)
    shape = tuple(shape)
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64, device=device)
    base = mix32(torch.tensor(int(seed) & _M32, dtype=torch.int64, device=device))
    state = mix32(base + ((idx * _GOLDEN) & _M32))
    state = torch.where(state == 0, torch.ones_like(state), state)
    return state.reshape(shape)


def split(rng: Tensor, num: int = 2) -> Tensor:
    """Derive ``num`` fresh RNG states from each element of ``rng`` (new leading dim)."""
    k = torch.arange(1, num + 1, dtype=torch.int64, device=rng.device)
    shape = (num,) + tuple(rng.shape)
    out = mix32(rng.unsqueeze(0) + (k.reshape((num,) + (1,) * rng.dim()) * _GOLDEN & _M32))
    out = torch.where(out == 0, torch.ones_like(out), out)
    return out.reshape(shape)


def fold_in(rng: Tensor, data: int) -> Tensor:
    out = mix32(rng + ((int(data) * _GOLDEN) & _M32))
    return torch.where(out == 0, torch.ones_like(out), out)


def next_state(rng: Tensor) -> Tensor:
    """Advance xorshift32 state(s) by one step."""
    s = rng & _M32
    s = (s ^ (s << 13)) & _M32
    s = s ^ (s >> 17)
    s = (s ^ (s << 5)) & _M32
    return s


def random_byte(rng: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(next_rng, value)`` with ``value`` uniform in ``[0, 255]``."""
    s = next_state(rng)
    return s, (s ^ (s >> 16)) & 0xFF


def randint(rng: Tensor, low: int, high: int) -> tuple[Tensor, Tensor]:
    """Return ``(next_rng, value)`` with ``value`` uniform in ``[low, high)``."""
    s = next_state(rng)
    return s, low + (mix32(s) % (high - low))


def uniform(rng: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(next_rng, value)`` with ``value`` uniform float in ``[0, 1)``."""
    s = next_state(rng)
    return s, mix32(s).to(torch.float32) * (1.0 / 4294967296.0)
