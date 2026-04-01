"""Gymnasium-compatible environment for the AI Fighting Game.

MDP definition
--------------
Agent:       The RL policy (controls the AI fighter).
Environment: Turn-based fight against a random opponent.

State space (observation):
  MultiDiscrete([5, 5, 5, 5, 4])
  Index 0 — agent_hp_bin    : HP ∈ [0,100] → 5 bins
  Index 1 — agent_mp_bin    : MP ∈ [0,50]  → 5 bins
  Index 2 — opp_hp_bin      : opponent HP binned
  Index 3 — opp_mp_bin      : opponent MP binned
  Index 4 — last_opp_move   : 0=none, 1=attack, 2=special, 3=regen

Action space: Discrete(3)
  0 = Attack  (MP cost 10, damage 10-20)
  1 = Regen   (no cost, +5 MP)
  2 = Special (MP cost 20, damage 25-35)

Reward shaping:
  +damage_dealt / MAX_DAMAGE * 0.5   per step
  -damage_taken / MAX_DAMAGE * 0.5   per step
  +1.0  if agent wins (opponent HP ≤ 0)
  -1.0  if agent loses (own HP ≤ 0)
  Episode truncated after MAX_ROUNDS rounds.
"""

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_HP = 100
MAX_MP = 50
MAX_DAMAGE = 35
HP_BINS = 5
MP_BINS = 5
MAX_ROUNDS = 50


def _bin(val: float, max_val: float, n_bins: int) -> int:
    return min(int(val / max_val * n_bins), n_bins - 1)


class FightEnv(gym.Env):
    """Gymnasium environment for the AI fighting game."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete(
            [HP_BINS, MP_BINS, HP_BINS, MP_BINS, 4]
        )
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        assert not self._done, "Call reset() before step() after episode end."

        agent_action = int(action)

        # Opponent acts randomly
        opp_action = self.np_random.integers(0, 3) if hasattr(self, "np_random") else random.randint(0, 2)
        opp_action = int(opp_action)

        agent_dmg = self._apply_action(
            action=agent_action,
            actor_mp=self._agent_mp,
            set_mp=lambda v: setattr(self, "_agent_mp", v),
        )
        self._opp_hp = max(0, self._opp_hp - agent_dmg)

        opp_dmg = self._apply_action(
            action=opp_action,
            actor_mp=self._opp_mp,
            set_mp=lambda v: setattr(self, "_opp_mp", v),
        )
        self._agent_hp = max(0, self._agent_hp - opp_dmg)

        self._last_opp_move = opp_action + 1  # store 1-indexed (1/2/3)
        self._round += 1

        # Reward
        reward = (agent_dmg - opp_dmg) / MAX_DAMAGE * 0.5
        terminated = False
        if self._opp_hp <= 0:
            reward += 1.0
            terminated = True
        elif self._agent_hp <= 0:
            reward -= 1.0
            terminated = True
        truncated = self._round >= MAX_ROUNDS
        self._done = terminated or truncated

        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        return (
            f"Round {self._round}: "
            f"Agent HP={self._agent_hp} MP={self._agent_mp} | "
            f"Opp HP={self._opp_hp} MP={self._opp_mp}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._agent_hp = MAX_HP
        self._agent_mp = MAX_MP
        self._opp_hp = MAX_HP
        self._opp_mp = MAX_MP
        self._round = 0
        self._last_opp_move = 0
        self._done = False

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                _bin(self._agent_hp, MAX_HP, HP_BINS),
                _bin(self._agent_mp, MAX_MP, MP_BINS),
                _bin(self._opp_hp, MAX_HP, HP_BINS),
                _bin(self._opp_mp, MAX_MP, MP_BINS),
                self._last_opp_move,
            ],
            dtype=np.int64,
        )

    def _apply_action(self, action: int, actor_mp: int, set_mp) -> int:
        """Apply an action and return the damage dealt."""
        if action == 0 and actor_mp >= 10:   # Attack
            set_mp(actor_mp - 10)
            return random.randint(10, 20)
        elif action == 2 and actor_mp >= 20:  # Special
            set_mp(actor_mp - 20)
            return random.randint(25, 35)
        else:                                  # Regen (or fallback)
            set_mp(min(actor_mp + 5, MAX_MP))
            return 0
