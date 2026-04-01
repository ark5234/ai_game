"""Tabular Q-learning agent for FightEnv.

State representation:
  Tuple of (agent_hp_bin, agent_mp_bin, opp_hp_bin, opp_mp_bin, last_opp_move)
  Each component is already discretised by FightEnv.

Action space: 3 discrete actions (0=Attack, 1=Regen, 2=Special).

Hyperparameters (defaults):
  alpha  (learning rate)    : 0.1
  gamma  (discount factor)  : 0.95
  epsilon (exploration rate) : 1.0 → 0.05 via exponential decay
  epsilon_decay              : 0.995
"""

import json
import os
import random
from typing import List, Optional

import numpy as np

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "qtable.json")


class QLearningAgent:
    """Tabular epsilon-greedy Q-learning agent."""

    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict = {}
        self.episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def choose_action(self, obs) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        key = self._state_key(obs)
        return int(np.argmax(self._get_q(key)))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, obs, action: int, reward: float, next_obs, done: bool):
        """TD(0) Q-table update."""
        key = self._state_key(obs)
        next_key = self._state_key(next_obs)
        q_vals = list(self._get_q(key))
        next_q_max = 0.0 if done else float(np.max(self._get_q(next_key)))
        target = reward + self.gamma * next_q_max
        q_vals[action] += self.alpha * (target - q_vals[action])
        self.q_table[key] = q_vals

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        save_path = path or QTABLE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            # Keep last 1 000 rewards to avoid unbounded file growth
            "episode_rewards": self.episode_rewards[-1000:],
        }
        with open(save_path, "w") as fh:
            json.dump(data, fh)

    def load(self, path: Optional[str] = None) -> bool:
        """Load saved Q-table. Returns True if successful."""
        load_path = path or QTABLE_PATH
        if not os.path.exists(load_path):
            return False
        with open(load_path) as fh:
            data = json.load(fh)
        self.q_table = data.get("q_table", {})
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.episode_rewards = data.get("episode_rewards", [])
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(obs) -> str:
        return str(tuple(int(x) for x in obs))

    def _get_q(self, key: str) -> list:
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.n_actions
        return self.q_table[key]
