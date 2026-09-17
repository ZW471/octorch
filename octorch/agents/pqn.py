"""Parallelised Q-Network (PQN) for Octorch environments (port of ``octax.agents.pqn``)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import nn

from octorch.agents.common import ConvFeatures, mlp, get_activation, evaluate
from octorch.wrappers import OctorchVectorEnv

Tensor = torch.Tensor


class PQNAgent(nn.Module):
    """Convolutional Q-network with LayerNorm + ReLU hidden layers (as in PQN)."""

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 4,
        obs_hw=(32, 64),
        features_list: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence = ((8, 4), 4, 3),
        strides_list: Sequence = ((4, 2), 2, 1),
        activation="relu",
        hidden_layer_sizes: Sequence[int] = (256,),
    ):
        super().__init__()
        act = get_activation(activation)
        self.features = ConvFeatures(in_channels, obs_hw, features_list, kernel_sizes, strides_list, act)
        self.q_network = mlp(self.features.out_dim, hidden_layer_sizes, action_dim, nn.ReLU, layer_norm=True)
        self.action_dim = action_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.q_network(self.features(obs))

    def act(self, obs: Tensor, epsilon: float) -> Tensor:
        q_values = self(obs)
        greedy = q_values.argmax(-1)
        random_action = torch.randint(0, self.action_dim, greedy.shape, device=obs.device)
        choose_random = torch.rand(greedy.shape, device=obs.device) < epsilon
        return torch.where(choose_random, random_action, greedy)

    def take(self, obs: Tensor, action: Tensor) -> Tensor:
        return self(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1)


@dataclass
class PQNOctorch:
    """PQN trainer (on-policy Q-learning with λ-returns and epsilon-greedy exploration)."""
    env: OctorchVectorEnv
    agent: Optional[PQNAgent] = None
    total_timesteps: int = 1_000_000
    num_steps: int = 32
    num_epochs: int = 1
    num_minibatches: int = 32
    learning_rate: float = 5e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    td_lambda: float = 0.9
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.1
    eval_freq: int = 131072
    num_eval_episodes: Optional[int] = None  # default: num_envs (uses the compiled step if any)
    eval_callback: Optional[Callable] = None
    seed: int = 0
    compile: bool = False
    agent_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        if self.num_eval_episodes is None:
            self.num_eval_episodes = self.num_envs
        if self.agent is None:
            self.agent = self.create_agent(self.env, **self.agent_kwargs)
        self.agent = self.agent.to(self.device)
        self.optimizer = torch.optim.RAdam(self.agent.parameters(), lr=self.learning_rate)
        self.global_step = 0
        self._forward = torch.compile(self.agent) if self.compile else self.agent

    @classmethod
    def create_agent(cls, env: OctorchVectorEnv, **agent_kwargs) -> PQNAgent:
        return PQNAgent(action_dim=env.num_actions, in_channels=env.single_observation_shape[0], **agent_kwargs)

    def epsilon_schedule(self, step: int) -> float:
        frac = min(1.0, step / max(1.0, self.exploration_fraction * self.total_timesteps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def make_act(self, epsilon: float = 0.005) -> Callable[[Tensor], Tensor]:
        agent = self.agent

        @torch.no_grad()
        def act(obs: Tensor) -> Tensor:
            return agent.act(obs, epsilon)
        return act

    def evaluate(self):
        return evaluate(self.make_act(0.005), self.env, self.num_eval_episodes, self.env.max_steps_in_episode, seed=self.seed)

    def collect_trajectories(self, last_obs: Tensor, epsilon: float):
        T, N = self.num_steps, self.num_envs
        obs_buf = torch.empty((T, N) + tuple(last_obs.shape[1:]), device=self.device, dtype=last_obs.dtype)
        act_buf = torch.empty((T, N), device=self.device, dtype=torch.int64)
        nextq_buf = torch.empty((T, N), device=self.device)
        rew_buf = torch.empty((T, N), device=self.device)
        done_buf = torch.empty((T, N), device=self.device, dtype=torch.bool)
        for t in range(T):
            with torch.no_grad():
                action = self.agent.act(last_obs, epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            with torch.no_grad():
                next_q = self._forward(next_obs).max(-1).values
            obs_buf[t], act_buf[t], nextq_buf[t] = last_obs, action, next_q
            rew_buf[t], done_buf[t] = reward, terminated | truncated
            last_obs = next_obs
            self.global_step += N
        return (obs_buf, act_buf, nextq_buf, rew_buf, done_buf), last_obs

    def calculate_targets(self, rewards: Tensor, dones: Tensor, next_q: Tensor):
        """λ-returns computed backwards, exactly like the rejax/Octax PQN."""
        T = rewards.shape[0]
        targets = torch.empty_like(rewards)
        not_done = 1.0 - dones.to(torch.float32)
        lambda_return = rewards[-1] + self.gamma * not_done[-1] * next_q[-1]
        targets[-1] = lambda_return
        for t in reversed(range(T - 1)):
            bootstrap = next_q[t] + self.td_lambda * (lambda_return - next_q[t])
            lambda_return = rewards[t] + not_done[t] * self.gamma * bootstrap
            targets[t] = lambda_return
        return targets

    def update(self, obs, actions, targets):
        q_values = self._forward(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        loss = 0.5 * ((q_values - targets) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": loss.item()}

    def train_iteration(self, last_obs: Tensor):
        epsilon = self.epsilon_schedule(self.global_step)
        (obs, actions, next_q, rewards, dones), last_obs = self.collect_trajectories(last_obs, epsilon)
        targets = self.calculate_targets(rewards, dones, next_q)
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // self.num_minibatches
        flat = [x.reshape(batch_size, *x.shape[2:]) for x in (obs, actions, targets)]
        metrics = {}
        for _ in range(self.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                metrics = self.update(*(x[idx] for x in flat))
        metrics["epsilon"] = epsilon
        return last_obs, metrics

    def train(self, log_fn: Optional[Callable[[dict], None]] = print):
        torch.manual_seed(self.seed)
        obs, _ = self.env.reset(self.seed)
        evaluations = []
        next_eval = 0
        start = time.time()
        while self.global_step < self.total_timesteps:
            if self.global_step >= next_eval:
                lengths, returns = self.evaluate()
                evaluations.append((self.global_step, lengths.cpu(), returns.cpu()))
                if self.eval_callback is not None:
                    self.eval_callback(self, self.global_step, lengths, returns)
                elif log_fn is not None:
                    log_fn({"step": self.global_step, "mean_length": lengths.float().mean().item(),
                            "mean_return": returns.mean().item(), "sps": self.global_step / max(time.time() - start, 1e-9)})
                next_eval += self.eval_freq
            obs, metrics = self.train_iteration(obs)
        lengths, returns = self.evaluate()
        evaluations.append((self.global_step, lengths.cpu(), returns.cpu()))
        if log_fn is not None:
            log_fn({"step": self.global_step, "mean_length": lengths.float().mean().item(),
                    "mean_return": returns.mean().item(), "sps": self.global_step / max(time.time() - start, 1e-9)})
        return evaluations
