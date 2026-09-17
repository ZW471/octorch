"""Proximal Policy Optimisation for Octorch environments (port of ``octax.agents.ppo``)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from octorch.agents.common import ConvFeatures, mlp, get_activation, evaluate
from octorch.wrappers import OctorchVectorEnv

Tensor = torch.Tensor


class PPOAgent(nn.Module):
    """Convolutional actor-critic (shared torso, separate policy and value heads)."""

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
        self.actor = mlp(self.features.out_dim, hidden_layer_sizes, action_dim, act)
        self.critic = mlp(self.features.out_dim, hidden_layer_sizes, 1, act)
        self.action_dim = action_dim

    def forward(self, obs: Tensor, action: Optional[Tensor] = None):
        """Returns ``(action, log_prob, entropy, value)``; samples an action if none is given."""
        features = self.features(obs)
        dist = Categorical(logits=self.actor(features))
        value = self.critic(features).squeeze(-1)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def call_critic(self, obs: Tensor) -> Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    def call_actor(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        logits = self.actor(self.features(obs))
        if deterministic:
            return logits.argmax(-1)
        return Categorical(logits=logits).sample()


@dataclass
class PPOOctorch:
    """PPO trainer with a rejax-like configuration surface.

    Args:
        env: :class:`OctorchVectorEnv` (auto-resetting)
        agent: :class:`PPOAgent` (created with :meth:`create_agent` if ``None``)
        total_timesteps, num_envs (taken from env), num_steps, num_epochs,
        num_minibatches, learning_rate, max_grad_norm, gamma, gae_lambda,
        clip_eps, vf_coef, ent_coef, eval_freq, eval_callback, seed
    """
    env: OctorchVectorEnv
    agent: Optional[PPOAgent] = None
    total_timesteps: int = 1_000_000
    num_steps: int = 32
    num_epochs: int = 8
    num_minibatches: int = 32
    learning_rate: float = 5e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
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
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.global_step = 0
        self._forward = torch.compile(self.agent) if self.compile else self.agent

    @classmethod
    def create_agent(cls, env: OctorchVectorEnv, **agent_kwargs) -> PPOAgent:
        return PPOAgent(action_dim=env.num_actions, in_channels=env.single_observation_shape[0], **agent_kwargs)

    # ------------------------------------------------------------------
    def make_act(self, deterministic: bool = False) -> Callable[[Tensor], Tensor]:
        agent = self.agent

        @torch.no_grad()
        def act(obs: Tensor) -> Tensor:
            return agent.call_actor(obs, deterministic=deterministic)
        return act

    def evaluate(self):
        act = self.make_act(deterministic=False)
        return evaluate(act, self.env, self.num_eval_episodes, self.env.max_steps_in_episode, seed=self.seed)

    def collect_trajectories(self, last_obs: Tensor):
        T, N = self.num_steps, self.num_envs
        obs_buf = torch.empty((T, N) + tuple(last_obs.shape[1:]), device=self.device, dtype=last_obs.dtype)
        act_buf = torch.empty((T, N), device=self.device, dtype=torch.int64)
        logp_buf = torch.empty((T, N), device=self.device)
        rew_buf = torch.empty((T, N), device=self.device)
        val_buf = torch.empty((T, N), device=self.device)
        done_buf = torch.empty((T, N), device=self.device, dtype=torch.bool)
        for t in range(T):
            with torch.no_grad():
                action, log_prob, _, value = self._forward(last_obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs_buf[t], act_buf[t], logp_buf[t] = last_obs, action, log_prob
            rew_buf[t], val_buf[t], done_buf[t] = reward, value, terminated | truncated
            last_obs = next_obs
            self.global_step += N
        return (obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf), last_obs

    def calculate_gae(self, rewards: Tensor, values: Tensor, dones: Tensor, last_val: Tensor):
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        adv = torch.zeros_like(last_val)
        next_value = last_val
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t].to(torch.float32)
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            adv = delta + self.gamma * self.gae_lambda * not_done * adv
            advantages[t] = adv
            next_value = values[t]
        return advantages, advantages + values

    def update(self, obs, actions, old_logp, old_values, advantages, targets):
        _, log_prob, entropy, value = self._forward(obs, actions)
        entropy = entropy.mean()
        ratio = torch.exp(log_prob - old_logp)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        pi_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        actor_loss = pi_loss - self.ent_coef * entropy

        value_pred_clipped = old_values + (value - old_values).clamp(-self.clip_eps, self.clip_eps)
        value_losses = (value - targets) ** 2
        value_losses_clipped = (value_pred_clipped - targets) ** 2
        value_loss = 0.5 * torch.maximum(value_losses, value_losses_clipped).mean()
        loss = actor_loss + self.vf_coef * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": loss.item(), "pi_loss": pi_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}

    def train_iteration(self, last_obs: Tensor):
        (obs, actions, logp, rewards, values, dones), last_obs = self.collect_trajectories(last_obs)
        with torch.no_grad():
            last_val = self.agent.call_critic(last_obs)
            last_val = torch.where(dones[-1], torch.zeros_like(last_val), last_val)
        advantages, targets = self.calculate_gae(rewards, values, dones, last_val)

        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // self.num_minibatches
        flat = [x.reshape(batch_size, *x.shape[2:]) for x in (obs, actions, logp, values, advantages, targets)]
        metrics = {}
        for _ in range(self.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                metrics = self.update(*(x[idx] for x in flat))
        return last_obs, metrics

    def train(self, log_fn: Optional[Callable[[dict], None]] = print):
        """Run training. Returns a list of ``(global_step, lengths, returns)`` evaluations."""
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
