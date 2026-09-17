# Environments

`octorch.environments.create_environment(env_id, **kwargs)` returns `(env, metadata)`. Valid ids are in
`octorch.environments.ENV_IDS`: the 19 single games plus `cavern1`–`cavern6` (and `cavern4a`, `cavern4b`),
`space_flight1`–`space_flight10` and `target_shooter1`–`target_shooter3`.

Keyword arguments are forwarded to `OctorchEnv`: `max_num_steps_per_episodes`, `instruction_frequency`,
`fps`, `frame_skip`, `device`, `render_mode`, `render_scale`, `color_scheme`, `modern_mode`, `timer_wraparound`.

Each game module defines `rom_file`, `score_fn(state)`, `terminated_fn(state)`, `action_set`,
`startup_instructions` or `custom_startup`, optionally `disable_delay`, and `metadata`. Functions receive a
*batched* state and must use ellipsis indexing (`state.V[..., 5]`) so that they work for any batch shape.

See [games.md](games.md) for the per-game register maps and descriptions.
