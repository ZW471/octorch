# Octorch Tutorial

This tutorial series covers all aspects of using Octorch, a PyTorch-based CHIP-8 emulator and reinforcement learning environment suite (a port of [Octax](https://github.com/riiswa/octax)).

## Tutorial Structure

### [01 - Quickstart Guide](01_quickstart.md)
Get up and running with Octorch in minutes. Covers installation, basic environment usage, batching and your first RL experiments.

### [02 - Core Emulator Concepts](02_core_emulator_concepts.md)
Understanding the CHIP-8 architecture, instruction execution, batched states and how the emulator works internally.

### [03 - Reinforcement Learning Environments](03_reinforcement_learning_environments.md)
Using Octorch for RL training, understanding reward structures, compiling the environment step and implementing training loops with PyTorch.

### [04 - Custom Game Environments](04_custom_game_environments.md)
Creating your own game environments, analyzing ROM files, and implementing score/termination functions.

## Prerequisites

- Python 3.10+
- PyTorch 2.14+ installed through `uv sync` (see the installation guide in the quickstart)
- Basic familiarity with reinforcement learning concepts
- Understanding of Python and PyTorch tensors

All code snippets are written to be pasted into `uv run python` from the repository root.
