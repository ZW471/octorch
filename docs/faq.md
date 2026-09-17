# Frequently Asked Questions

## The environment is slow

Eager execution launches hundreds of tiny kernels per CHIP-8 instruction. Compile it:

```python
env, _ = create_environment("brix")
env.compile(batch_size=4096)          # once per batch size
```

Every `env.step` / `env.step_` on a state of that batch shape then replays a CUDA graph
(~1M steps/s on an A100). Use a large batch: throughput keeps improving up to tens of thousands of
environments.

## First `compile` is slow

`torch.compile` needs 10 s – 5 min depending on `granularity` (`"instruction"` < `"frame"` < `"step"`).
Inductor caches compiled kernels on disk, so later processes are much faster.

## How do I know which key index maps to which action?

`env.action_set` lists the CHIP-8 keys; action `i` presses `action_set[i]`, the last action is a no-op.

## The environment seems stuck in a menu

Games run `startup_instructions` (or `custom_startup`) at creation to skip title screens. When you load a
ROM manually use `octorch.run_n_instruction(state, 500)`.

## Why does the reward stay at 0?

Check the score register with `uv run python play.py roms/<rom>.ch8` (BCD operations are flagged
with 🎯), and make sure the start-up phase is long enough.

## Can I change the episode length?

`create_environment("brix", max_num_steps_per_episodes=10_000)` or `env.from_minutes(2.5)`.

## How is randomness handled?

Each environment carries an xorshift32 state in `state.rng`, advanced only by the `CXNN` instruction.
`env.reset(seed, batch_size=N)` derives `N` independent streams; `octorch.make_rng` / `octorch.split`
mirror `jax.random.PRNGKey` / `jax.random.split`.

## Is it really identical to Octax?

Yes, bit-for-bit: see `tests/test_octax_crossval.py`. The only intentional difference is that Octorch
does not use JAX's Threefry generator for `CXNN` (it has its own vectorised PRNG); the cross-validation
patches Octax to use the same generator.

## Does it run on CPU?

Yes. `create_environment(..., device="cpu")`; `env.compile(..., cudagraph=False)` still helps.
