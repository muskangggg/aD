"""
dqn_agent.py  (v4 -- smooth learning: LR 1e-4 + soft target updates)

Step 7: DQN agent for the Auto.defi pipeline.

Architecture / schedule (Auto.gov Table I + proposal Sec 5.3):
- QNetwork: 5 layers in total -- input, 3 hidden layers x 256 units, output.
- Prioritized replay: TD-error-based priority, x3 multiplier for
  transitions where the Risk Signal R was > 0.7 at that step -- makes the
  agent replay near-liquidation moments more often than routine ones.
- Epsilon-greedy exploration, linearly decayed from EPSILON_START to
  EPSILON_END over EPSILON_DECAY_STEPS.
- Target network activated once epsilon < 0.3 (as in Auto.gov).

What changed in v4 (30 Oct -- the REWARD is unchanged):
  Check 0 (diagnose.py) showed the LR 1e-3 agent SWINGING between safe and
  unsafe from one checkpoint to the next (liq_rate 0.04 -> 0.54 -> 0.08 ->
  0.72), while its value guesses stayed flat. Its action scores were close
  together, so each big update flipped the greedy choice ("policy churn",
  Schaul et al., 2022). Two standard ways to make updates smaller/smoother:
  1. LR 1e-3 -> 1e-4 (tested 30 Oct: 0/300 liquidations on VAL for all 3
     seeds at ep1000, swings between checkpoints much smaller).
  2. SOFT TARGET UPDATES (Polyak averaging, Lillicrap et al., 2016): instead
     of copying the online net into the target net in ONE jump every 500
     steps, the target net moves a tiny step (TARGET_TAU = 0.5%) towards the
     online net after EVERY learning step. The learning target then changes
     smoothly instead of jumping every 500 steps. With tau 0.005 the target
     lags about 200 steps behind on average (hard copy every 500: about 250),
     so mainly the JUMPS are removed, not the lag.
  Set USE_SOFT_TARGET = False to get the v3 hard copy back.

What changed in v3 (fix plan Step 2):
  1. DOUBLE DQN (van Hasselt, Guez & Silver, 2016): the online network
     PICKS the best next action, the target network JUDGES its value.
  2. HUBER LOSS (Mnih et al., 2015) -- tried, turned OFF (made the agent
     forget liquidations, run v3h).
  3. GRADIENT CLIPPING (max norm GRAD_CLIP_NORM): caps each learning step.

What changed in v2:
- GAMMA = 0.95 (was 0.5). User-level liquidation risk builds over several
  days, so the agent needs a longer horizon to act early.
- ACTION MASKING (Huang & Ontanon, 2020): the agent only picks among
  actions that can actually do something (info["action_mask"]), both when
  acting and in its learning targets.

Hyperparameters not specified in the proposal (learning rate, buffer size,
batch size, epsilon decay length, target update, PER alpha/beta,
gradient clip) are reasonable defaults below -- tunable.

Usage (standalone smoke test against LendingEnv):
    python dqn_agent.py
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

# ---- Hyperparameters ----
# From Auto.gov / the proposal:
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 3  # + input and output = 5 layers total (Auto.gov Table I)
PRIORITY_HIGH_RISK_MULTIPLIER = 3.0
PRIORITY_HIGH_RISK_THRESHOLD = 0.7
TARGET_NET_EPSILON_THRESHOLD = 0.3  # target net activates once epsilon drops below this

# Changed from Auto.gov's 0.5 -- see module docstring:
GAMMA = 0.95

# v3 -- steadier learning (see module docstring):
USE_DOUBLE_DQN = True
USE_HUBER_LOSS = False   # turned off 29 Oct: Huber made the agent forget liquidations (v3h)
GRAD_CLIP_NORM = 10.0

# v4 -- smooth learning (see module docstring):
USE_SOFT_TARGET = False  # tested 30 Oct on seed 1: no clear gain, so the final setup keeps the hard copy
TARGET_TAU = 0.005       # soft update: target = (1 - tau) * target + tau * online, every step

# Not specified by the proposal -- reasonable defaults, tune later:
LR = 1e-4                # was 1e-3 (changed 30 Oct: 1e-3 made the policy swing)
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20_000
TARGET_UPDATE_EVERY = 500  # steps; only used when USE_SOFT_TARGET = False
PRIORITY_EPSILON = 1e-3    # floor so no transition ever has zero priority
PRIORITY_ALPHA = 0.6       # how much prioritization vs. uniform sampling (0=uniform)


class QNetwork(nn.Module):
    """MLP: N_HIDDEN_LAYERS hidden layers of HIDDEN_DIM units + 1 output layer."""

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


def masked_argmax(q_values: np.ndarray, mask: np.ndarray | None) -> int:
    """Best action among the allowed ones (all allowed if mask is None)."""
    q = np.array(q_values, dtype=np.float64)
    if mask is not None:
        q[~np.asarray(mask, dtype=bool)] = -np.inf
    return int(np.argmax(q))


class PrioritizedReplayBuffer:
    """Experience replay with TD-error-based priorities, boosted x3 for
    transitions where the Risk Signal R at that step was > 0.7. Also stores
    which actions were allowed in the NEXT state (for masked targets)."""

    def __init__(self, n_actions: int, capacity: int = BUFFER_CAPACITY, alpha: float = PRIORITY_ALPHA):
        self.n_actions = n_actions
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.risk_signals = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done, risk_signal: float, next_mask=None):
        if next_mask is None:
            next_mask = np.ones(self.n_actions, dtype=bool)
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        priority = max_priority
        if risk_signal > PRIORITY_HIGH_RISK_THRESHOLD:
            priority *= PRIORITY_HIGH_RISK_MULTIPLIER

        transition = (obs, action, reward, next_obs, done, np.asarray(next_mask, dtype=bool))
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

        obs, actions, rewards, next_obs, dones, next_masks = zip(*transitions)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
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
        self.replay_buffer = PrioritizedReplayBuffer(n_actions)

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
    def select_action(self, obs: np.ndarray, mask: np.ndarray | None = None) -> int:
        allowed = np.arange(self.n_actions) if mask is None else np.flatnonzero(mask)
        if random.random() < self.epsilon:
            return int(random.choice(list(allowed)))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(obs_t).squeeze(0).cpu().numpy()
        return masked_argmax(q_values, mask)

    # ---- Store a transition ----
    def store_transition(self, obs, action, reward, next_obs, done, risk_signal: float, next_mask=None):
        self.replay_buffer.add(obs, action, reward, next_obs, done, risk_signal, next_mask)

    # ---- Learning target ----
    def _next_state_values(self, next_obs_t, next_masks_t):
        """Value of the best ALLOWED next action ("hold" is always allowed).
        Double DQN: the online net picks the action, the target net judges it.
        Before the target net is active, both jobs use the online net."""
        judge_net = self.target_net if self.target_net_active else self.online_net
        judge_q = judge_net(next_obs_t)

        if USE_DOUBLE_DQN:
            picker_q = self.online_net(next_obs_t).masked_fill(~next_masks_t, float("-inf"))
            best_actions = picker_q.argmax(dim=1, keepdim=True)
            next_values = judge_q.gather(1, best_actions).squeeze(1)
        else:  # v2 behaviour: the same net picks and judges
            next_values = judge_q.masked_fill(~next_masks_t, float("-inf")).max(dim=1)[0]

        return torch.where(torch.isfinite(next_values), next_values, torch.zeros_like(next_values))

    # ---- Target network update ----
    @torch.no_grad()
    def _soft_update_target(self, tau: float = TARGET_TAU):
        """Polyak averaging: move every target weight a small step towards the online weight."""
        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.mul_(1.0 - tau).add_(o_param, alpha=tau)

    def _update_target(self):
        if not self.target_net_active:
            return
        if USE_SOFT_TARGET:
            self._soft_update_target()
        elif self.steps_done % TARGET_UPDATE_EVERY == 0:   # v3: hard copy in one jump
            self.target_net.load_state_dict(self.online_net.state_dict())

    # ---- Learning step ----
    def learn(self, batch_size: int = BATCH_SIZE, beta: float = 0.4) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones, next_masks, indices, weights, risk_signals = \
            self.replay_buffer.sample(batch_size, beta)

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, device=self.device)
        weights_t = torch.as_tensor(weights, device=self.device)

        q_values = self.online_net(obs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self._next_state_values(next_obs_t, next_masks_t)
            targets = rewards_t + GAMMA * next_q_values * (1.0 - dones_t)

        td_errors = (q_values - targets).detach().cpu().numpy()
        if USE_HUBER_LOSS:
            per_sample = F.smooth_l1_loss(q_values, targets, reduction="none")
        else:
            per_sample = (q_values - targets).pow(2)
        loss = (weights_t * per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors, risk_signals)

        self._update_target()

        self.steps_done += 1
        self.update_epsilon()

        return float(loss.item())


def weight_distance(net_a: nn.Module, net_b: nn.Module) -> float:
    """Total absolute difference between two networks' weights (0 = identical)."""
    with torch.no_grad():
        return float(sum((a - b).abs().sum() for a, b in zip(net_a.parameters(), net_b.parameters())))


def main():
    """Smoke test: run the agent (with action masking) against LendingEnv
    for a few thousand steps to confirm nothing crashes and that the target
    network follows the online network smoothly."""
    from lending_env import LendingEnv  # noqa: E402

    print(f"DQN v4: LR={LR:g}, double_dqn={USE_DOUBLE_DQN}, huber={USE_HUBER_LOSS}, "
          f"grad_clip={GRAD_CLIP_NORM}, soft_target={USE_SOFT_TARGET} (tau={TARGET_TAU})")
    env = LendingEnv(seed=42)
    obs, info = env.reset()
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n, seed=42)

    # Check A: one soft update moves the target exactly tau of the way to the online net.
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.add_(0.01)
    before = weight_distance(agent.online_net, agent.target_net)
    agent._soft_update_target()
    after = weight_distance(agent.online_net, agent.target_net)
    ok = abs(after - (1.0 - TARGET_TAU) * before) < 1e-3 * before
    print(f"check A (soft update): distance {before:.3f} -> {after:.3f} "
          f"(expected {(1.0 - TARGET_TAU) * before:.3f}) -> {'OK' if ok else 'FAIL'}")
    agent.target_net.load_state_dict(agent.online_net.state_dict())

    # Check B: run past target-net activation (epsilon < 0.3 at about step 14,800).
    losses, invalid, dists = [], 0, []
    for step_i in range(16_000):
        action = agent.select_action(obs, info["action_mask"])
        next_obs, reward, terminated, truncated, info = env.step(action)
        invalid += int(info["invalid_action"])
        agent.store_transition(obs, action, reward, next_obs, terminated,
                               info["risk_signal"], info["action_mask"])

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

        if agent.target_net_active:
            dists.append(weight_distance(agent.online_net, agent.target_net))
        if step_i % 2000 == 0:
            avg_loss = np.mean(losses[-50:]) if losses else float("nan")
            print(f"step {step_i}: epsilon={agent.epsilon:.3f} target_active={agent.target_net_active} "
                  f"buffer={len(agent.replay_buffer)} avg_loss={avg_loss:.4f}")

    print(f"\ninvalid actions chosen: {invalid} (should be 0 with masking)")
    if dists:
        jumps = int(np.sum(np.diff(dists) < -0.5 * np.array(dists[:-1])))
        print(f"check B (target follows smoothly): online-target distance after activation "
              f"min {min(dists):.2f} / max {max(dists):.2f}; sudden drops to ~0: {jumps} "
              f"(soft update -> 0; hard copy -> about 1 per {TARGET_UPDATE_EVERY} steps)")
    print("Done.")


if __name__ == "__main__":
    main()