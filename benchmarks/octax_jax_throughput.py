"""Throughput of the original JAX Octax: jit(vmap(env.step)) over B environments."""
import sys, time, json
sys.path.insert(0, "/tmp/octax")
import jax, jax.numpy as jnp
from octax.environments import create_environment
games = ["brix", "pong", "tetris", "blinky", "cavern1", "space_flight1"]
sizes = [256, 1024, 4096, 16384, 65536]
rows = []
for game in games:
    env, _ = create_environment(game)
    for B in sizes:
        rngs = jax.random.split(jax.random.PRNGKey(0), B)
        state, obs, info = jax.vmap(env.reset)(rngs)
        actions = jax.random.randint(jax.random.PRNGKey(1), (B,), 0, env.num_actions)
        step = jax.jit(jax.vmap(env.step))
        t = time.time(); out = step(state, actions); jax.block_until_ready(out); ct = time.time() - t
        for _ in range(3): out = step(out[0], actions)
        jax.block_until_ready(out)
        n = 30; t = time.time()
        for _ in range(n): out = step(out[0], actions)
        jax.block_until_ready(out); dt = (time.time() - t) / n
        rows.append(dict(game=game, batch=B, compile_s=ct, ms_per_step=dt * 1e3, steps_per_s=B / dt))
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("scaling_jax.json", "w"), indent=1)
