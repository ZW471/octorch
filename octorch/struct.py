"""Minimal immutable "pytree" dataclasses backed by torch tensors.

This mirrors the role of ``flax.struct`` in Octax: states are frozen dataclasses
whose tensor fields can be mapped over, moved between devices, indexed along the
leading batch dimensions and functionally updated with ``replace``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator, Tuple

import torch

Tensor = torch.Tensor


def _is_node(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, Struct))


def tree_map(fn: Callable[[Tensor], Any], tree: Any) -> Any:
    """Apply ``fn`` to every tensor leaf of ``tree`` (Struct / tuple / list / dict)."""
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, Struct):
        return tree.replace(**{name: tree_map(fn, getattr(tree, name)) for name in tree.node_fields()})
    if isinstance(tree, (tuple, list)):
        return type(tree)(tree_map(fn, v) for v in tree)
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    return tree


def tree_leaves(tree: Any) -> Iterator[Tensor]:
    """Iterate over the tensor leaves of ``tree``."""
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, Struct):
        for name in tree.node_fields():
            yield from tree_leaves(getattr(tree, name))
    elif isinstance(tree, (tuple, list)):
        for v in tree:
            yield from tree_leaves(v)
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from tree_leaves(v)


def static_field(**kwargs):
    """Declare a non-tensor (static) field, like ``flax.struct.field(pytree_node=False)``."""
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def field(**kwargs):
    return dataclasses.field(**kwargs)


@dataclasses.dataclass(frozen=True)
class Struct:
    """Frozen dataclass with functional update helpers.

    Sub-classes must be decorated with ``@dataclasses.dataclass(frozen=True)``.
    Tensor / Struct fields are "nodes"; other fields (declared with
    :func:`static_field` or holding python scalars) are carried through unchanged.
    """

    @classmethod
    def node_fields(cls) -> Tuple[str, ...]:
        return tuple(
            f.name for f in dataclasses.fields(cls) if not f.metadata.get("static", False)
        )

    def replace(self, **changes) -> "Struct":
        """Return a copy with the given fields replaced (like ``flax.struct.replace``)."""
        return dataclasses.replace(self, **changes)

    def asdict(self) -> dict:
        """Non-recursive dict of fields (like Octax's ``asdict_non_recursive``)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    # ---- tensor helpers -------------------------------------------------
    def to(self, device=None, non_blocking: bool = False) -> "Struct":
        return tree_map(lambda t: t.to(device, non_blocking=non_blocking), self)

    def clone(self) -> "Struct":
        """Deep copy of all tensor leaves."""
        return tree_map(lambda t: t.clone(), self)

    def detach(self) -> "Struct":
        return tree_map(lambda t: t.detach(), self)

    def cpu(self) -> "Struct":
        return self.to("cpu")

    def cuda(self, device=None) -> "Struct":
        return self.to("cuda" if device is None else device)

    def leaves(self) -> Iterator[Tensor]:
        return tree_leaves(self)

    @property
    def device(self) -> torch.device:
        for leaf in self.leaves():
            return leaf.device
        return torch.device("cpu")

    def __getitem__(self, index) -> "Struct":
        """Index along the leading (batch) dimensions of every tensor leaf."""
        return tree_map(lambda t: t[index], self)

    def copy_(self, other: "Struct") -> "Struct":
        """In-place copy every tensor leaf of ``other`` into ``self``."""
        for name in self.node_fields():
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, Struct):
                mine.copy_(theirs)
            else:
                mine.copy_(theirs)
        return self

    def masked_copy_(self, mask: Tensor, other: "Struct") -> "Struct":
        """In-place: leaves of ``self`` take values from ``other`` where ``mask`` is True.

        ``mask`` has the batch shape; it is broadcast over trailing dims.
        """
        for name in self.node_fields():
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, Struct):
                mine.masked_copy_(mask, theirs)
            else:
                m = mask.reshape(mask.shape + (1,) * (mine.dim() - mask.dim()))
                mine.copy_(torch.where(m, theirs, mine))
        return self


def stack(structs, dim: int = 0):
    """Stack a sequence of structs along a new leading dimension."""
    first = structs[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(structs), dim)
    return first.replace(**{
        name: stack([getattr(s, name) for s in structs], dim) for name in first.node_fields()
    })
