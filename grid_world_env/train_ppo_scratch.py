"""Hand-written PPO for the two-phase GridWorld.

Components:
  - ActorCritic: shared MLP with separate actor and critic heads
  - RolloutBuffer: numpy on-policy buffer with GAE computation
  - Training loop: collect rollout → GAE → K epochs of PPO clip updates

Uses FullyObservable wrapper (4D obs: [dx, dy, phase, t/T]) so the agent
knows its current phase and timestep — identical to train_dqn_scratch.py.

Model saved as .pt with prefix 'ppo_scratch_' to distinguish from SB3 models.

Usage:
  python grid_world_env/train_ppo_scratch.py --timesteps 500000
  python grid_world_env/train_ppo_scratch.py --use-potential-shaping \\
      --reward-0-step 0 --reward-0-terminal 1 \\
      --reward-1-step -2 --reward-1-terminal 0
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium

import grid_world_env
from grid_world_env.wrappers import FullyObservable


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared MLP backbone with separate actor (policy) and critic (value) heads.

    Architecture: Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU →
      actor: Linear(64, n_actions)
      critic: Linear(64, 1)
    """

    def __init__(self, obs_dim=4, n_actions=4, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Returns (action_logits, value) — shapes (batch, n_actions) and (batch, 1)."""
        h = self.backbone(x)
        return self.actor(h), self.critic(h)

    def get_action_and_value(self, obs, action=None):
        """Sample action (or evaluate given action) and return statistics.

        Returns:
            action:   (batch,)  int64 — sampled or provided
            log_prob: (batch,)  float — log π(a|s)
            entropy:  (batch,)  float — H(π(·|s))
            value:    (batch,)  float — V(s) (squeezed from critic head)
        """
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """On-policy rollout buffer with GAE computation.

    Convention for `dones`:
      dones[t] = 1 if the transition at step t ended the episode.
    This means nextnonterminal = 1 - dones[t] when bootstrapping
    from step t to step t+1.
    """

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pos = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def reset(self):
        self.pos = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        """Backward pass to fill self.advantages and self.returns via GAE.

        Args:
            last_value: V(s_{T}) — critic estimate for the state after the rollout
            last_done:  bool — True if the episode ended at the last rollout step
            gamma:      discount factor
            gae_lambda: GAE smoothing parameter (1 = MC, 0 = TD)
        """
        last_gae_lam = 0.0
        T = len(self)
        for step in reversed(range(T)):
            if step == T - 1:
                nextnonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                nextnonterminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * nextnonterminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self):
        """Return all stored data as torch tensors (on CPU)."""
        T = len(self)
        return {
            "obs":        torch.tensor(self.obs[:T],      dtype=torch.float32),
            "actions":    torch.tensor(self.actions[:T],  dtype=torch.long),
            "log_probs":  torch.tensor(self.log_probs[:T],dtype=torch.float32),
            "advantages": torch.tensor(self.advantages[:T],dtype=torch.float32),
            "returns":    torch.tensor(self.returns[:T],  dtype=torch.float32),
        }

    def __len__(self):
        return self.pos


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(render_mode=None,
             reward_0_step=0, reward_0_terminal=1,
             reward_1_step=-2, reward_1_terminal=0,
             max_steps=100,
             reward_0_approach=None, reward_0_retreat=0.1,
             use_potential_shaping=False, potential_gamma=0.99):
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=max_steps,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_1_step,
        reward_1_terminal=reward_1_terminal,
        reward_0_approach=reward_0_approach,
        reward_0_retreat=reward_0_retreat,
        use_potential_shaping=use_potential_shaping,
        potential_gamma=potential_gamma,
    )
    env = FullyObservable(env, max_steps=max_steps)
    return env


# ---------------------------------------------------------------------------
# Run-name helpers
# ---------------------------------------------------------------------------

def make_run_name(reward_0_step, reward_0_terminal, reward_1_step, reward_1_terminal):
    def fmt(v):
        return str(int(v)) if v == int(v) else str(v)
    return (
        f"ppo_scratch_r0s{fmt(reward_0_step)}_r0t{fmt(reward_0_terminal)}"
        f"_r1s{fmt(reward_1_step)}_r1t{fmt(reward_1_terminal)}"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env,
    total_timesteps=500_000,
    lr=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    seed=0,
    device="cpu",
    init_policy=None,
):
    """PPO training loop.

    Args:
        init_policy: optional ActorCritic to fine-tune from instead of
                     initialising a fresh network.  Useful for carrying a
                     policy across RLHF rounds without resetting weights.

    Returns:
        Trained ActorCritic model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if init_policy is not None:
        import copy
        model = copy.deepcopy(init_policy).to(device)
    else:
        model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(capacity=n_steps, obs_dim=obs_dim)

    obs, _ = env.reset(seed=seed)
    last_step_done = False

    global_step = 0
    update = 0
    episode_rewards = []
    current_ep_reward = 0.0

    while global_step < total_timesteps:
        # ---- Collect rollout ----
        buffer.reset()
        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            step_done = terminated or truncated
            current_ep_reward += reward

            buffer.add(
                obs,
                action.item(),
                reward,
                step_done,
                value.item(),
                log_prob.item(),
            )
            obs = next_obs
            last_step_done = step_done
            global_step += 1

            if step_done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()
                last_step_done = False

        # Bootstrap from the state after the rollout
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, last_value = model.get_action_and_value(obs_t)

        buffer.compute_returns_and_advantages(
            last_value.item(), last_step_done, gamma, gae_lambda
        )

        # ---- PPO update ----
        data = buffer.get()
        b_obs       = data["obs"].to(device)
        b_actions   = data["actions"].to(device)
        b_log_probs = data["log_probs"].to(device)
        b_advantages = data["advantages"].to(device)
        b_returns   = data["returns"].to(device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for _ in range(n_epochs):
            idx = torch.randperm(n_steps, device=device)
            for start in range(0, n_steps, batch_size):
                mb_idx = idx[start : start + batch_size]

                _, new_log_prob, entropy, new_value = model.get_action_and_value(
                    b_obs[mb_idx], action=b_actions[mb_idx]
                )

                ratio = torch.exp(new_log_prob - b_log_probs[mb_idx])
                mb_adv = b_advantages[mb_idx]

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
                ).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                e_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * e_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        update += 1
        if update % 10 == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0]
            mean_r = np.mean(recent)
            print(
                f"  update={update:>4}  step={global_step:>7}"
                f"  episodes={len(episode_rewards)}"
                f"  mean_reward(last20)={mean_r:.2f}"
            )

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO from scratch on GridWorld")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Rollout length before each update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-path", type=str, default=None)
    # Reward modes
    parser.add_argument("--reward-0-step", type=float, default=0.0)
    parser.add_argument("--reward-0-terminal", type=float, default=1.0)
    parser.add_argument("--reward-1-step", type=float, default=-2.0)
    parser.add_argument("--reward-1-terminal", type=float, default=0.0)
    parser.add_argument("--reward-0-approach", type=float, default=None,
                        help="Scenario A: reward for reducing L1 distance in phase 0")
    parser.add_argument("--reward-0-retreat", type=float, default=0.1,
                        help="Scenario A: penalty for increasing L1 distance in phase 0")
    parser.add_argument("--use-potential-shaping", action="store_true",
                        help="Enable potential-based reward shaping F=γΦ(s')-Φ(s)")
    parser.add_argument("--potential-gamma", type=float, default=0.99,
                        help="Discount factor used in potential shaping")
    args = parser.parse_args()

    run_name = make_run_name(
        reward_0_step=args.reward_0_step,
        reward_0_terminal=args.reward_0_terminal,
        reward_1_step=args.reward_1_step,
        reward_1_terminal=args.reward_1_terminal,
    )
    if args.save_path is None:
        args.save_path = f"models/{run_name}.pt"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    env = make_env(
        reward_0_step=args.reward_0_step,
        reward_0_terminal=args.reward_0_terminal,
        reward_1_step=args.reward_1_step,
        reward_1_terminal=args.reward_1_terminal,
        max_steps=args.max_steps,
        reward_0_approach=args.reward_0_approach,
        reward_0_retreat=args.reward_0_retreat,
        use_potential_shaping=args.use_potential_shaping,
        potential_gamma=args.potential_gamma,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training PPO (scratch) — run: {run_name}")
    print(f"  timesteps={args.timesteps}, device={device}")
    print(f"  obs: {env.observation_space.shape}, actions: {env.action_space.n}")
    print(f"  n_steps={args.n_steps}, batch_size={args.batch_size}, n_epochs={args.n_epochs}")
    if args.use_potential_shaping:
        print(f"  [potential shaping] potential_gamma={args.potential_gamma}")
    elif args.reward_0_approach is not None:
        print(f"  [Scenario A] approach={args.reward_0_approach}, retreat={args.reward_0_retreat}")

    model = train(
        env=env,
        total_timesteps=args.timesteps,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=device,
    )

    torch.save({"actor_critic": model.state_dict(), "run_name": run_name}, args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
