"""Hand-written DQN for the two-phase GridWorld.

Components:
  - QNetwork: MLP with two hidden layers of 64 units
  - ReplayBuffer: numpy circular buffer
  - Training loop with epsilon-greedy exploration, Bellman targets, target network

Uses FullyObservable wrapper (4D obs: [dx, dy, phase, t/T]) instead of
RelativePosition, so the agent knows its current phase and timestep.

Model saved as .pt with prefix 'dqn_scratch_' to distinguish from SB3 models.

Usage:
  python grid_world_env/train_dqn_scratch.py --timesteps 100000
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
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim=4, n_actions=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(render_mode=None, reward_0_step=1, reward_0_terminal=100,
             reward_1_step=-1, reward_1_terminal=100, max_steps=100,
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
        f"dqn_scratch_r0s{fmt(reward_0_step)}_r0t{fmt(reward_0_terminal)}"
        f"_r1s{fmt(reward_1_step)}_r1t{fmt(reward_1_terminal)}"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env,
    total_timesteps=100_000,
    lr=1e-4,
    batch_size=32,
    gamma=0.99,
    buffer_size=50_000,
    learning_starts=1_000,
    train_freq=4,
    target_update_interval=500,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_fraction=0.1,
    seed=0,
    device="cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim=obs_dim, n_actions=n_actions).to(device)
    target_net = QNetwork(obs_dim=obs_dim, n_actions=n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size, obs_dim=obs_dim)

    epsilon_decay_steps = int(total_timesteps * epsilon_decay_fraction)

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_count = 0
    total_episodes_reward = []

    for step in range(total_timesteps):
        # Epsilon-linear decay
        frac = min(1.0, step / max(epsilon_decay_steps, 1))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        # Select action
        if step < learning_starts or random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = q_net(obs_t)
                action = int(q_vals.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        buffer.push(obs, next_obs, action, reward, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()
            total_episodes_reward.append(episode_reward)
            episode_reward = 0.0
            episode_count += 1

        # Training update
        if step >= learning_starts and step % train_freq == 0 and len(buffer) >= batch_size:
            obs_b, next_obs_b, act_b, rew_b, done_b = buffer.sample(batch_size)

            obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
            next_obs_t = torch.tensor(next_obs_b, dtype=torch.float32, device=device)
            act_t = torch.tensor(act_b, dtype=torch.long, device=device)
            rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device)
            done_t = torch.tensor(done_b, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_q = target_net(next_obs_t).max(dim=1).values
                y = rew_t + gamma * next_q * (1.0 - done_t)

            current_q = q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(current_q, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync target network
        if step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (step + 1) % 10_000 == 0:
            recent = total_episodes_reward[-20:] if total_episodes_reward else [0]
            mean_r = np.mean(recent)
            print(f"  step={step+1:>7}  eps={epsilon:.3f}  episodes={episode_count}"
                  f"  mean_reward(last20)={mean_r:.1f}")

    return q_net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DQN from scratch on GridWorld")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--reward-0-step", type=float, default=1.0)
    parser.add_argument("--reward-0-terminal", type=float, default=100.0)
    parser.add_argument("--reward-1-step", type=float, default=-1.0)
    parser.add_argument("--reward-1-terminal", type=float, default=100.0)
    parser.add_argument("--reward-0-approach", type=float, default=None,
                        help="Scenario A: reward for reducing L1 distance in phase 0")
    parser.add_argument("--reward-0-retreat", type=float, default=0.1,
                        help="Scenario A: penalty for increasing L1 distance in phase 0")
    parser.add_argument("--use-potential-shaping", action="store_true",
                        help="Enable potential-based reward shaping F=γΦ(s')-Φ(s)")
    parser.add_argument("--potential-gamma", type=float, default=0.99,
                        help="Discount factor used in potential shaping (default same as gamma)")
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
    print(f"Training DQN (scratch) — run: {run_name}")
    print(f"  timesteps={args.timesteps}, device={device}")
    print(f"  obs: {env.observation_space.shape}, actions: {env.action_space.n}")

    q_net = train(
        env=env,
        total_timesteps=args.timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        epsilon_start=1.0,
        epsilon_end=args.exploration_final_eps,
        epsilon_decay_fraction=args.exploration_fraction,
        seed=args.seed,
        device=device,
    )

    torch.save({"q_network": q_net.state_dict(), "run_name": run_name}, args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
