"""Per-round damage tracking with CSV and JSONL persistence."""

import csv
import json
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import List

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


@dataclass
class RoundRecord:
    round_num: int
    player_move: int
    ai_move: int
    player_damage: int
    ai_damage: int
    player_hp_after: int
    ai_hp_after: int
    player_mp_after: int
    ai_mp_after: int
    player_hp_delta: int
    ai_hp_delta: int
    ai_confidence: float
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


class MatchTracker:
    """
    Tracks per-round data for a single match.

    On each round it:
    - Appends to a global `logs/game_logs.csv`
    - Accumulates records in memory

    At the end of a match call `save()` to write per-match CSV and JSONL.
    """

    def __init__(self, match_id: str = None):
        if match_id is None:
            match_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.match_id = match_id
        self.rounds: List[RoundRecord] = []
        os.makedirs(LOGS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_round(
        self,
        round_num: int,
        player_move: int,
        ai_move: int,
        player_damage: int,
        ai_damage: int,
        player_hp_after: int,
        ai_hp_after: int,
        player_mp_after: int,
        ai_mp_after: int,
        player_hp_delta: int,
        ai_hp_delta: int,
        ai_confidence: float,
    ):
        rec = RoundRecord(
            round_num=round_num,
            player_move=player_move,
            ai_move=ai_move,
            player_damage=player_damage,
            ai_damage=ai_damage,
            player_hp_after=player_hp_after,
            ai_hp_after=ai_hp_after,
            player_mp_after=player_mp_after,
            ai_mp_after=ai_mp_after,
            player_hp_delta=player_hp_delta,
            ai_hp_delta=ai_hp_delta,
            ai_confidence=ai_confidence,
        )
        self.rounds.append(rec)
        self._append_global_log(rec)

    def _append_global_log(self, rec: RoundRecord):
        global_csv = os.path.join(LOGS_DIR, "game_logs.csv")
        row = {**asdict(rec), "match_id": self.match_id}
        fieldnames = list(asdict(rec).keys()) + ["match_id"]
        write_header = not os.path.exists(global_csv)
        with open(global_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Write per-match CSV and JSONL files to logs/."""
        if not self.rounds:
            return
        match_csv = os.path.join(LOGS_DIR, f"match_{self.match_id}.csv")
        match_jsonl = os.path.join(LOGS_DIR, f"match_{self.match_id}.jsonl")

        fieldnames = list(asdict(self.rounds[0]).keys())
        with open(match_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rounds:
                writer.writerow(asdict(r))

        with open(match_jsonl, "w") as fh:
            for r in self.rounds:
                fh.write(json.dumps(asdict(r)) + "\n")
