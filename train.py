"""Train PPO or PQN on an Octorch game.

Example:
    uv run python train.py --env brix --agent PPO --num-envs 512 --total-timesteps 5000000
    uv run python train.py --config conf/config.yaml
"""

import argparse
import os
import pickle
import time

import torch

from octorch.agents import PPOOctorch, PQNOctorch
from octorch.environments import create_environment
from octorch.wrappers import OctorchVectorEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML file with the same keys as the CLI flags")
    p.add_argument("--env", type=str, default="brix")
    p.add_argument("--agent", type=str, default="PPO", choices=["PPO", "PQN"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--eval-freq", type=int, default=131072)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--compile-env", action="store_true", help="torch.compile + CUDA graph the environment step")
    p.add_argument("--compile-agent", action="store_true", help="torch.compile the network")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)
    return args


def main():
    args = parse_args()
    env_name = args.env
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + "__".join(
        f"{k}_{str(v).lower().replace(' ', '_')}" for k, v in vars(args).items()
        if isinstance(v, (str, int, float, bool, type(None))) and k not in ("config", "results_dir")
    )

    algo_cls = {"PPO": PPOOctorch, "PQN": PQNOctorch}[args.agent]
    all_returns, all_steps = [], None
    start = time.time()
    for seed in range(args.seed, args.seed + args.num_seeds):
        base_env, metadata = create_environment(env_name, device=device)
        if args.compile_env:
            base_env.compile(args.num_envs)
        env = OctorchVectorEnv(base_env, num_envs=args.num_envs, seed=seed)
        algo = algo_cls(
            env=env,
            total_timesteps=args.total_timesteps,
            num_steps=args.num_steps,
            num_epochs=args.num_epochs,
            num_minibatches=args.num_minibatches,
            learning_rate=args.learning_rate,
            eval_freq=args.eval_freq,
            seed=seed,
            compile=args.compile_agent,
        )
        print(f"=== {metadata['title']} | {args.agent} | seed {seed} | {device}", flush=True)
        evaluations = algo.train(log_fn=lambda m: print(
            f"Step {m['step']}, Mean episode length: {m['mean_length']:.1f}, "
            f"Mean return: {m['mean_return']:.2f}, SPS: {m['sps']:.0f}", flush=True))
        all_steps = [e[0] for e in evaluations]
        all_returns.append([e[2].mean().item() for e in evaluations])
    end = time.time()

    returns = torch.tensor(all_returns)  # (num_seeds, num_evals)
    os.makedirs(f"{args.results_dir}/{env_name}/", exist_ok=True)
    with open(f"{args.results_dir}/{env_name}/{job_id}.pkl", "wb") as f:
        pickle.dump({"timesteps": all_steps, "returns": returns, "config": vars(args), "time": end - start}, f)
    with open(f"{args.results_dir}/results.txt", "a") as f:
        f.write(f"{env_name}__{job_id}\t{returns[:, -1].mean():.3f}±{returns[:, -1].std() if len(all_returns) > 1 else 0:.3f}\n")
    print(f"Results saved in {args.results_dir}/{env_name}/{job_id}.pkl  (wall time {end - start:.1f}s)")


if __name__ == "__main__":
    main()
