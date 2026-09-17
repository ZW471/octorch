import torch, time, json, sys
from octorch.environments import create_environment
games = ["brix", "pong", "tetris", "blinky", "cavern1", "space_flight1"]
sizes = [256, 1024, 4096, 16384, 65536]
rows = []
for game in games:
    env, _ = create_environment(game, device="cuda")
    for B in sizes:
        t = time.time(); env.compile(B, granularity="frame"); torch.cuda.synchronize(); ct = time.time() - t
        state = env.static_state(B); s0, _, _ = env.reset(0, batch_size=B); state.copy_(s0)
        a = torch.randint(0, env.num_actions, (B,), device="cuda")
        for _ in range(3): env.step_(state, a)
        torch.cuda.synchronize(); t = time.time(); n = 30
        for _ in range(n): env.step_(state, a)
        torch.cuda.synchronize(); dt = (time.time() - t) / n
        # eager for small sizes only
        if B <= 4096:
            s1 = s0.clone(); torch.cuda.synchronize(); t = time.time()
            for _ in range(3): env._step_eager(s1, a)
            torch.cuda.synchronize(); de = (time.time() - t) / 3
        else:
            de = None
        rows.append(dict(game=game, batch=B, compile_s=ct, ms_per_step=dt * 1e3, steps_per_s=B / dt, eager_steps_per_s=(B / de if de else None)))
        print(json.dumps(rows[-1]), flush=True)
        env._compiled.clear(); torch.cuda.empty_cache()
json.dump(rows, open("scaling.json", "w"), indent=1)
