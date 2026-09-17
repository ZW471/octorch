# Validation against Octax

Every Octax-loadable environment replayed for 128 steps in JAX Octax and in Octorch with identical actions; ✅ = identical at every step. Produced by `benchmarks/octax_identity_table.py` from traces dumped with `benchmarks/octax_dump_traces.py`.

| env id | title | steps | observations | registers V | PC / I | PRNG | reward | terminated / truncated | score |
| --- | --- | ---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `airplane` | Airplane | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `blinky` | Blinky | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `brix` | Brix | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cavern1` | Cavern | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cavern2` | Cavern | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cavern3` | Cavern | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cavern5` | Cavern | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cavern6` | Cavern | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `deep` | Deep8 | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `filter` | Filter | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `flight_runner` | Flight Runner | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `missile` | Missile Command | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pong` | Pong | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `rocket` | Rocket | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `shooting_stars` | Shooting Stars | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight1` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight10` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight2` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight3` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight4` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight5` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight6` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight7` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight8` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `space_flight9` | Space Flight | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `spacejam` | Spacejam! | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `squash` | Squash | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `submarine` | Submarine | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tank` | Tank Battle | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `target_shooter1` | Target Shooter - LLM-Generated RL Environment | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `target_shooter2` | Target Shooter - LLM-Generated RL Environment | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `target_shooter3` | Target Shooter - LLM-Generated RL Environment | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tetris` | Tetris | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ufo` | UFO | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `vertical_brix` | Vertical Brix | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `wipe_off` | Wipe Off | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `worm` | SuperWorm V4 | 128 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
