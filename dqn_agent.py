"""
dqn_agent.py

Step 7: DQN agent for the Auto.defi pipeline.

Architecture / schedule, per the proposal:
- QNetwork: 5-layer MLP (4 hidden layers x 256 units + 1 output layer)
- Prioritized replay: TD-error-based priority, x3 multiplier for
  transitions where the Risk Signal R (info["risk_signal"]) was > 0.7 at
  the time of that transition -- makes the agent replay near-liquidation
  moments more often than routine, low-risk ones.
- Epsilon-greedy exploration, linearly decayed from EPSILON_START to
  EPSILON_END over EPSILON_DECAY_STEPS.
- Target network: NOT active from the start. Below EPSILON_START the agent
  bootstraps directly off the online network; once epsilon drops below
  TARGET_NET_EPSILON_THRESHOLD (0.3), a separate target network activates
  and is hard-updated every TARGET_UPDATE_EVERY steps. This delayed
  activation is unusual (standard DQN uses a target net from step one for
  stability) but implemented as specified in the proposal.
- GAMMA = 0.5 (short discount horizon, per proposal).

Hyperparameters not specified in the proposal (learning rate, buffer size,
batch size, epsilon decay length, target update frequency, PER alpha/beta)
are reasonable defaults below -- flagged as tunable, not locked-in.

Usage (standalone smoke test against LendingEnv):
    python dqn_agent.py
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

# ---- Hyperparameters ----
# Specified by the proposal:
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 4  # + 1 output layer = 5 layers total
GAMMA = 0.5
PRIORITY_HIGH_RISK_MULTIPLIER = 3.0
PRIORITY_HIGH_RISK_THRESHOLD = 0.7
TARGET_NET_EPSILON_THRESHOLD = 0.3  # target net activates once epsilon drops below this

# Not specified by the proposal -- reasonable defaults, tune later:
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20_000
TARGET_UPDATE_EVERY = 500  # steps, once target net is active
PRIORITY_EPSILON = 1e-3    # floor so no transition ever has zero priority
PRIORITY_ALPHA = 0.6       # how much prioritization vs. uniform sampling (0=uniform)


class QNetwork(nn.Module):
    """5-layer MLP: 4 hidden layers of 256 units + 1 output layer."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = HIDDEN_DIM, n_hidden: int = N_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PrioritizedReplayBuffer:
    """Experience replay with TD-error-based priorities, boosted x3 for
    transitions where the Risk Signal R at that step was > 0.7 -- these are
    the near-liquidation moments, so the agent should learn from them more
    often than an average low-risk transition."""

    def __init__(self, capacity: int = BUFFER_CAPACITY, alpha: float = PRIORITY_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.risk_signals = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done, risk_signal: float):
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        priority = max_priority
        if risk_signal > PRIORITY_HIGH_RISK_THRESHOLD:
            priority *= PRIORITY_HIGH_RISK_MULTIPLIER

        transition = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.risk_signals.append(risk_signal)
        else:
            self.buffer[self.pos] = transition
            self.risk_signals[self.pos] = risk_signal
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs)
        transitions = [self.buffer[i] for i in indices]
        sampled_risk_signals = [self.risk_signals[i] for i in indices]

        # importance-sampling weights correct for the sampling bias PER introduces
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
            sampled_risk_signals,
        )

    def update_priorities(self, indices, td_errors, risk_signals):
        """Refresh priorities after a learning step using fresh TD-errors,
        re-applying the high-risk multiplier."""
        for idx, td_err, r in zip(indices, td_errors, risk_signals):
            priority = abs(float(td_err)) + PRIORITY_EPSILON
            if r > PRIORITY_HIGH_RISK_THRESHOLD:
                priority *= PRIORITY_HIGH_RISK_MULTIPLIER
            self.priorities[idx] = priority


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, seed: int | None = None, device: str = "cpu"):
        self.n_actions = n_actions
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.online_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.replay_buffer = PrioritizedReplayBuffer()

        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.target_net_active = False  # flips on once epsilon < TARGET_NET_EPSILON_THRESHOLD

    # ---- Epsilon schedule ----
    def update_epsilon(self):
        frac = min(1.0, self.steps_done / EPSILON_DECAY_STEPS)
        self.epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)

        if not self.target_net_active and self.epsilon < TARGET_NET_EPSILON_THRESHOLD:
            self.target_net_active = True
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"[DQNAgent] epsilon={self.epsilon:.3f} < {TARGET_NET_EPSILON_THRESHOLD} -> target network activated")

    # ---- Action selection ----
    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    # ---- Store a transition ----
    def store_transition(self, obs, action, reward, next_obs, done, risk_signal: float):
        self.replay_buffer.add(obs, action, reward, next_obs, done, risk_signal)

    # ---- Learning step ----
    def learn(self, batch_size: int = BATCH_SIZE, beta: float = 0.4) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones, indices, weights, risk_signals = \
            self.replay_buffer.sample(batch_size, beta)

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        weights_t = torch.as_tensor(weights, device=self.device)

        q_values = self.online_net(obs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            # before the target net activates, bootstrap off the online net
            # itself (this delayed schedule is intentional -- see module docstring)
            bootstrap_net = self.target_net if self.target_net_active else self.online_net
            next_q_values = bootstrap_net(next_obs_t).max(dim=1)[0]
            targets = rewards_t + GAMMA * next_q_values * (1.0 - dones_t)

        td_errors = (q_values - targets).detach().cpu().numpy()
        loss = (weights_t * (q_values - targets).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors, risk_signals)

        if self.target_net_active and self.steps_done % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.steps_done += 1
        self.update_epsilon()

        return float(loss.item())


def main():
    """Smoke test: run the agent against LendingEnv for a few hundred steps
    to confirm nothing crashes and epsilon/target-net/loss behave sensibly.
    This is NOT the real training loop (that's Step 8)."""
    from lending_env import LendingEnv  # noqa: E402

    env = LendingEnv(seed=42)
    obs, info = env.reset()
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n, seed=42)

    losses = []
    for step_i in range(500):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, terminated or truncated, info["risk_signal"])

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

        if step_i % 100 == 0:
            avg_loss = np.mean(losses[-50:]) if losses else float("nan")
            print(f"step {step_i}: epsilon={agent.epsilon:.3f} target_active={agent.target_net_active} "
                  f"buffer={len(agent.replay_buffer)} avg_loss={avg_loss:.4f}")

    print(f"\nDone. Final epsilon={agent.epsilon:.3f}, target_net_active={agent.target_net_active}, "
          f"buffer size={len(agent.replay_buffer)}")


if __name__ == "__main__":
    main()