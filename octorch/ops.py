"""Small batched indexing helpers shared by the instruction handlers."""

import torch

Tensor = torch.Tensor


def take(t: Tensor, idx: Tensor) -> Tensor:
    """``t[..., idx]`` for a per-batch index ``idx`` of batch shape."""
    return t.gather(-1, idx.to(torch.int64).unsqueeze(-1)).squeeze(-1)


def take_many(t: Tensor, idx: Tensor) -> Tensor:
    """Gather several per-batch indices: ``idx`` has shape ``(..., k)``."""
    return t.gather(-1, idx.to(torch.int64))


def _prep(t: Tensor, idx, value):
    idx = torch.as_tensor(idx, device=t.device).to(torch.int64)
    idx = torch.broadcast_to(idx, t.shape[:-1])
    value = torch.as_tensor(value, device=t.device).to(t.dtype)
    value = torch.broadcast_to(value, idx.shape)
    return idx.unsqueeze(-1), value.unsqueeze(-1)


def put(t: Tensor, idx, value) -> Tensor:
    """Out-of-place ``t[..., idx] = value`` (``idx`` is an int or a batch-shaped tensor)."""
    idx, value = _prep(t, idx, value)
    return t.scatter(-1, idx, value)


def put_(t: Tensor, idx, value) -> Tensor:
    """In-place ``t[..., idx] = value``."""
    idx, value = _prep(t, idx, value)
    return t.scatter_(-1, idx, value)


def bmask(mask: Tensor, t: Tensor) -> Tensor:
    """Reshape a batch-shaped boolean ``mask`` so that it broadcasts against ``t``."""
    return mask.reshape(mask.shape + (1,) * (t.dim() - mask.dim()))


def where_(mask: Tensor, new: Tensor, t: Tensor) -> Tensor:
    """In-place ``t = where(mask, new, t)`` with ``mask`` of batch shape."""
    return t.copy_(torch.where(bmask(mask, t), torch.as_tensor(new, device=t.device).to(t.dtype), t))
