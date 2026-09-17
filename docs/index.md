# Octorch documentation

Octorch is a PyTorch port of [Octax](https://github.com/riiswa/octax): CHIP-8 arcade games as
massively parallel reinforcement-learning environments.

- [Installation](installation.md)
- [Environments](environments/index.md) and the [game list](environments/games.md)
- [Validation table](validation.md)
- [FAQ](faq.md)
- [Changelog](changelog.md)
- Tutorials: [quick start](../tutorials/01_quickstart.md), [core emulator concepts](../tutorials/02_core_emulator_concepts.md),
  [RL environments](../tutorials/03_reinforcement_learning_environments.md), [custom games](../tutorials/04_custom_game_environments.md)

## Mental model

| Octax (JAX)                              | Octorch (PyTorch)                                                   |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `EmulatorState` flax struct              | `EmulatorState` frozen dataclass of tensors with a batch shape       |
| `jax.random.PRNGKey` in `state.rng`      | xorshift32 state (`int64`) in `state.rng`, `octorch.make_rng(seed, shape)` |
| `jax.vmap(env.step)`                     | `env.step(state, action)` on a batched state                        |
| `jax.jit`                                | `env.compile(batch_size)` (torch.compile + CUDA graph)              |
| `jax.lax.cond(done, env.reset, ...)`     | `env.reset_where(state, done)`                                      |
| `OctaxGymnaxWrapper`                     | `OctorchVectorEnv` (auto-reset, `(B, C, H, W)` float observations)  |
| `state.V[5]` in game definitions         | `state.V[..., 5]`                                                   |
| functional `execute(state, instr)`       | `execute` (functional) and `execute_` (in place)                    |
