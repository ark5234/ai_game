"""Core headless battle engine.

Can be driven step-by-step by a GUI or run in a loop by the CLI.
Both the ML-ensemble AI and the RL agent are supported via `use_rl`.
"""

from typing import List, Optional

from .fighter import Fighter, MOVE_NAMES
from .ai_opponent import AdaptiveAIOpponent
from .rl_agent import QLearningAgent
from .damage_tracker import MatchTracker

MAX_HP = 100
MAX_MP = 50


class BattleEngine:
    """Manages a single match between a human fighter and an AI/RL opponent."""

    def __init__(
        self,
        player: Fighter,
        ai: AdaptiveAIOpponent,
        rl_agent: Optional[QLearningAgent] = None,
        use_rl: bool = False,
        tracker: Optional[MatchTracker] = None,
    ):
        self.player = player
        self.ai = ai
        self.rl_agent = rl_agent
        self.use_rl = use_rl and rl_agent is not None
        self.tracker = tracker or MatchTracker()

        self.round_num: int = 0
        self.game_over: bool = False
        self.winner: Optional[str] = None
        self.last_log_entries: List[str] = []

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def execute_player_move(self, player_move: int) -> dict:
        """
        Execute one full round.

        player_move: 1=Attack, 2=Special, 3=Regen

        Returns a dict with:
          round_num, player_move, ai_move,
          player_dmg, ai_dmg, ai_confidence,
          game_over, winner
        """
        if self.game_over:
            return {}

        self.round_num += 1
        self.last_log_entries = []

        prev_player_hp = self.player.health
        prev_ai_hp = self.ai.health

        # --- Player acts ---
        player_dmg = self.player.execute_move(player_move)
        self.ai.take_damage(player_dmg)
        move_label = MOVE_NAMES.get(player_move, "?")
        if player_dmg > 0:
            self.last_log_entries.append(
                f"Player used {move_label} for {player_dmg} dmg!"
            )
        else:
            self.last_log_entries.append(f"Player used {move_label}.")

        # --- AI acts ---
        if self.use_rl:
            ai_move = self._rl_ai_move()
            ai_confidence = 0.0
        else:
            ai_move, ai_confidence = self.ai.predict_move(player_move)

        ai_dmg = self.ai.execute_move(ai_move)
        self.player.take_damage(ai_dmg)
        ai_label = MOVE_NAMES.get(ai_move, "?")
        if ai_dmg > 0:
            self.last_log_entries.append(
                f"AI used {ai_label} for {ai_dmg} dmg! (conf={ai_confidence:.1f}%)"
            )
        else:
            self.last_log_entries.append(
                f"AI used {ai_label}. (conf={ai_confidence:.1f}%)"
            )

        # Log per-model confidences (ML mode only)
        if not self.use_rl and hasattr(self.ai, "last_confidences"):
            c = self.ai.last_confidences
            self.last_log_entries.append(
                f"  [RF:{c.get('rf', 0):.1f}% "
                f"NN:{c.get('nn', 0):.1f}% "
                f"NB:{c.get('nb', 0):.1f}%]"
            )

        # --- Update ML model ---
        if not self.use_rl:
            self.ai.update_and_train(player_move, ai_move)

        # --- Record round ---
        self.tracker.record_round(
            round_num=self.round_num,
            player_move=player_move,
            ai_move=ai_move,
            player_damage=player_dmg,
            ai_damage=ai_dmg,
            player_hp_after=self.player.health,
            ai_hp_after=self.ai.health,
            player_mp_after=self.player.mp,
            ai_mp_after=self.ai.mp,
            player_hp_delta=self.player.health - prev_player_hp,
            ai_hp_delta=self.ai.health - prev_ai_hp,
            ai_confidence=ai_confidence,
        )

        # --- Win check ---
        if not self.player.is_alive():
            self.game_over = True
            self.winner = self.ai.name
            self.last_log_entries.append(f"{self.player.name} is defeated!")
        elif not self.ai.is_alive():
            self.game_over = True
            self.winner = self.player.name
            self.last_log_entries.append(f"{self.ai.name} is defeated!")

        return {
            "round_num": self.round_num,
            "player_move": player_move,
            "ai_move": ai_move,
            "player_dmg": player_dmg,
            "ai_dmg": ai_dmg,
            "ai_confidence": ai_confidence,
            "game_over": self.game_over,
            "winner": self.winner,
        }

    # ------------------------------------------------------------------
    # RL move helper
    # ------------------------------------------------------------------

    def _rl_ai_move(self) -> int:
        """Translate RL observation → action → 1-indexed move."""
        import numpy as np

        obs = np.array(
            [
                min(int(self.ai.health / MAX_HP * 5), 4),
                min(int(self.ai.mp / MAX_MP * 5), 4),
                min(int(self.player.health / MAX_HP * 5), 4),
                min(int(self.player.mp / MAX_MP * 5), 4),
                0,
            ],
            dtype=np.int64,
        )
        action = self.rl_agent.choose_action(obs)
        # Map RL actions: 0→Attack(1), 1→Regen(3), 2→Special(2)
        return {0: 1, 1: 3, 2: 2}[action]
