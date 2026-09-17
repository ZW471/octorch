"""Tests for control flow instructions."""

import pytest
from octorch import execute
from octorch.ops import put


class TestJump:
    """Test jump instructions."""

    def test_execute_jump(self, fresh_state):
        """Test 1NNN - Jump to address."""
        state = execute(fresh_state, 0x1001)
        assert state.pc == 1


class TestSkipInstructions:
    """Test all skip instruction variants."""

    def test_skip_if_equal_immediate_true(self, fresh_state):
        """3XNN - Should skip when VX == NN."""
        state = fresh_state.replace(V=put(fresh_state.V, 5, 0x42))
        initial_pc = state.pc

        state = execute(state, 0x3542)  # Skip if V5 == 0x42
        assert state.pc == initial_pc + 2

    def test_skip_if_equal_immediate_false(self, fresh_state):
        """3XNN - Should not skip when VX != NN."""
        state = fresh_state.replace(V=put(fresh_state.V, 5, 0x41))
        initial_pc = state.pc

        state = execute(state, 0x3542)  # Skip if V5 == 0x42
        assert state.pc == initial_pc

    def test_skip_if_not_equal_immediate_true(self, fresh_state):
        """4XNN - Should skip when VX != NN."""
        state = fresh_state.replace(V=put(fresh_state.V, 3, 0x10))
        initial_pc = state.pc

        state = execute(state, 0x4320)  # Skip if V3 != 0x20
        assert state.pc == initial_pc + 2

    def test_skip_if_not_equal_immediate_false(self, fresh_state):
        """4XNN - Should not skip when VX == NN."""
        state = fresh_state.replace(V=put(fresh_state.V, 3, 0x20))
        initial_pc = state.pc

        state = execute(state, 0x4320)  # Skip if V3 != 0x20
        assert state.pc == initial_pc

    def test_skip_if_equal_register_true(self, fresh_state):
        """5XY0 - Should skip when VX == VY."""
        state = fresh_state
        state = state.replace(V=put(state.V, 1, 0x55))
        state = state.replace(V=put(state.V, 2, 0x55))
        initial_pc = state.pc

        state = execute(state, 0x5120)  # Skip if V1 == V2
        assert state.pc == initial_pc + 2

    def test_skip_if_equal_register_false(self, fresh_state):
        """5XY0 - Should not skip when VX != VY."""
        state = fresh_state
        state = state.replace(V=put(state.V, 1, 0x55))
        state = state.replace(V=put(state.V, 2, 0x44))
        initial_pc = state.pc

        state = execute(state, 0x5120)  # Skip if V1 == V2
        assert state.pc == initial_pc

    def test_skip_if_not_equal_register_true(self, fresh_state):
        """9XY0 - Should skip when VX != VY."""
        state = fresh_state
        state = state.replace(V=put(state.V, 7, 0xAA))
        state = state.replace(V=put(state.V, 8, 0xBB))
        initial_pc = state.pc

        state = execute(state, 0x9780)  # Skip if V7 != V8
        assert state.pc == initial_pc + 2

    def test_skip_if_not_equal_register_false(self, fresh_state):
        """9XY0 - Should not skip when VX == VY."""
        state = fresh_state
        state = state.replace(V=put(state.V, 7, 0xCC))
        state = state.replace(V=put(state.V, 8, 0xCC))
        initial_pc = state.pc

        state = execute(state, 0x9780)  # Skip if V7 != V8
        assert state.pc == initial_pc

    def test_skip_with_zero_values(self, fresh_state):
        """Test skip instructions with zero values."""
        state = fresh_state
        initial_pc = state.pc

        # V0 == 0, should skip
        state = execute(state, 0x3000)  # Skip if V0 == 0
        assert state.pc == initial_pc + 2

    def test_skip_boundary_values(self, fresh_state):
        """Test skip instructions with boundary values."""
        state = fresh_state.replace(V=put(fresh_state.V, 0, 0xFF))
        initial_pc = state.pc

        state = execute(state, 0x30FF)  # Skip if V0 == 255
        assert state.pc == initial_pc + 2


class TestJumpWithOffset:
    """Test jump with offset in both modes."""

    def test_jump_with_offset_legacy(self, legacy_state):
        """BNNN - Jump with V0 offset (legacy mode)."""
        state = execute(legacy_state, 0x6010)  # V0 = 0x10
        state = execute(state, 0xB250)  # Jump to 0x250 + V0
        assert state.pc == 0x260

    def test_jump_with_offset_modern(self, modern_state):
        """BXNN - Jump with VX offset (modern mode)."""
        state = execute(modern_state, 0x6210)  # V2 = 0x10
        state = execute(state, 0xB250)  # Jump to 0x50 + V2
        assert state.pc == 0x60

    def test_jump_mode_comparison(self):
        """Test that modern_mode changes jump behavior."""
        from octorch import create_state

        # Legacy mode: B250 = Jump to 0x250 + V0
        state_legacy = create_state().replace(modern_mode=False)
        state_legacy = execute(state_legacy, 0x6010)  # V0 = 0x10
        state_legacy = execute(state_legacy, 0x6230)  # V2 = 0x30
        state_legacy = execute(state_legacy, 0xB250)

        # Modern mode: B250 = Jump to 0x50 + V2
        state_modern = create_state().replace(modern_mode=True)
        state_modern = execute(state_modern, 0x6010)  # V0 = 0x10
        state_modern = execute(state_modern, 0x6230)  # V2 = 0x30
        state_modern = execute(state_modern, 0xB250)

        assert state_legacy.pc == 0x260  # 0x250 + 0x10 (used V0)
        assert state_modern.pc == 0x80  # 0x50 + 0x30 (used V2)