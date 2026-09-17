"""CHIP-8 stack operations (batched, functional)."""

import torch

from octorch.constants import ADDRESS_MASK, STACK_SIZE
from octorch.ops import take, put
from octorch.state import StackState

Tensor = torch.Tensor


def _normalize_index(idx: Tensor) -> tuple[Tensor, Tensor]:
    """Mimic ``jnp`` indexing semantics: negative indices wrap, reads clamp, writes drop OOB."""
    idx = torch.where(idx < 0, idx + STACK_SIZE, idx)
    valid = (idx >= 0) & (idx < STACK_SIZE)
    return idx.clamp(0, STACK_SIZE - 1), valid


def push(stack: StackState, address: Tensor) -> StackState:
    """Push address onto stack."""
    masked_address = (address.to(torch.int64) & ADDRESS_MASK).to(torch.int32)
    idx, valid = _normalize_index(stack.pointer.to(torch.int64))
    current = take(stack.data, idx)
    new_data = put(stack.data, idx, torch.where(valid, masked_address, current))
    return stack.replace(data=new_data, pointer=stack.pointer + 1)


def pop(stack: StackState) -> tuple[StackState, Tensor]:
    """Pop address from stack."""
    new_pointer = stack.pointer - 1
    idx, valid = _normalize_index(new_pointer.to(torch.int64))
    popped_address = take(stack.data, idx)
    new_data = put(stack.data, idx, torch.where(valid, torch.zeros_like(popped_address), popped_address))
    return stack.replace(data=new_data, pointer=new_pointer), popped_address
