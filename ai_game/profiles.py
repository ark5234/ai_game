"""Player profile system — persistent per-player stats stored as JSON."""

import json
import os
import time
from typing import Optional

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "profiles")


class PlayerProfile:
    """Stores and persists per-player performance statistics."""

    def __init__(self, name: str):
        self.name = name
        self.games_played: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.total_damage_dealt: int = 0
        self.total_damage_taken: int = 0
        self.total_moves: int = 0
        self.move_usage_counts: dict = {1: 0, 2: 0, 3: 0}
        self.last_played: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "total_moves": self.total_moves,
            "move_usage_counts": self.move_usage_counts,
            "last_played": self.last_played,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        p = cls(data["name"])
        p.games_played = data.get("games_played", 0)
        p.wins = data.get("wins", 0)
        p.losses = data.get("losses", 0)
        p.total_damage_dealt = data.get("total_damage_dealt", 0)
        p.total_damage_taken = data.get("total_damage_taken", 0)
        p.total_moves = data.get("total_moves", 0)
        raw_usage = data.get("move_usage_counts", {})
        p.move_usage_counts = {int(k): int(v) for k, v in raw_usage.items()}
        for k in (1, 2, 3):
            p.move_usage_counts.setdefault(k, 0)
        p.last_played = data.get("last_played")
        return p

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        os.makedirs(PROFILES_DIR, exist_ok=True)
        path = os.path.join(PROFILES_DIR, f"{self.name}.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, name: str) -> Optional["PlayerProfile"]:
        path = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_or_create(cls, name: str) -> "PlayerProfile":
        profile = cls.load(name)
        return profile if profile is not None else cls(name)

    @staticmethod
    def list_profiles() -> list:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        return sorted(
            fn[:-5] for fn in os.listdir(PROFILES_DIR) if fn.endswith(".json")
        )

    # ------------------------------------------------------------------
    # Stats update
    # ------------------------------------------------------------------

    def record_match(
        self,
        won: bool,
        damage_dealt: int,
        damage_taken: int,
        moves: int,
        move_usage: dict,
    ):
        """Called at the end of each match to update stats."""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.total_damage_dealt += damage_dealt
        self.total_damage_taken += damage_taken
        self.total_moves += moves
        for move, count in move_usage.items():
            key = int(move)
            self.move_usage_counts[key] = self.move_usage_counts.get(key, 0) + count
        self.last_played = time.strftime("%Y-%m-%dT%H:%M:%S")
