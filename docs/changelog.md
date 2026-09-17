# Changelog

## [0.1.0] — 2026-09

Initial release: PyTorch port of Octax 0.1.1.

### Added
- Batched CHIP-8 emulator (`octorch.emulator`) evaluating all opcodes under masks; functional and in-place variants
- Scalar reference interpreter (`octorch.interpreter`) used as test oracle and for unbatched start-up
- Vectorised xorshift32 PRNG threaded through the emulator state (`octorch.prng`)
- `OctorchEnv` with batched `reset` / `step`, `reset_where`, `compile` (torch.compile + CUDA graph) and rendering
- All 22 games / 39 environment ids of Octax (incl. `cavern4a` / `cavern4b`)
- `OctorchVectorEnv` (auto-reset) and a gymnasium `VectorEnv` adapter
- PPO and PQN reference agents, `train.py`, `play.py`, `create_gifs.py`, examples
- Test-suite: ported Octax unit tests, engine consistency tests, bit-exact cross-validation with JAX Octax
