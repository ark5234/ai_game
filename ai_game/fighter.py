"""Fighter base class used by both the human player and AI opponents."""

import random

# Move identifiers
MOVE_ATTACK = 1
MOVE_SPECIAL = 2
MOVE_REGEN = 3

MOVE_NAMES = {MOVE_ATTACK: "Attack", MOVE_SPECIAL: "Special", MOVE_REGEN: "Regen"}
MOVE_MP_COST = {MOVE_ATTACK: 10, MOVE_SPECIAL: 20, MOVE_REGEN: 0}


class Fighter:
    """Represents either the human player or an AI combatant."""

    def __init__(self, name: str, health: int = 100, mp: int = 50):
        self.name = name
        self.health = health
        self.max_health = health
        self.mp = mp
        self.max_mp = mp

        # Per-session stats
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.total_moves = 0
        self.move_usage: dict = {MOVE_ATTACK: 0, MOVE_SPECIAL: 0, MOVE_REGEN: 0}
        self.wins = 0
        self.losses = 0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def attack(self) -> int:
        """Normal attack. Returns damage dealt (0 if not enough MP)."""
        if self.mp >= MOVE_MP_COST[MOVE_ATTACK]:
            damage = random.randint(10, 20)
            self.mp -= MOVE_MP_COST[MOVE_ATTACK]
            self._register_damage(damage, MOVE_ATTACK)
            return damage
        return 0

    def special_move(self) -> int:
        """Special attack. Returns damage dealt (0 if not enough MP)."""
        if self.mp >= MOVE_MP_COST[MOVE_SPECIAL]:
            damage = random.randint(25, 35)
            self.mp -= MOVE_MP_COST[MOVE_SPECIAL]
            self._register_damage(damage, MOVE_SPECIAL)
            return damage
        return 0

    def regenerate_mp(self) -> int:
        """Regenerate 5 MP. Always succeeds. Returns 0 (no damage)."""
        self.mp = min(self.mp + 5, self.max_mp)
        self.move_usage[MOVE_REGEN] += 1
        self.total_moves += 1
        return 0

    def execute_move(self, move: int) -> int:
        """Execute move by integer ID (1/2/3). Returns damage dealt."""
        if move == MOVE_ATTACK:
            return self.attack()
        if move == MOVE_SPECIAL:
            return self.special_move()
        if move == MOVE_REGEN:
            return self.regenerate_mp()
        return 0

    def take_damage(self, damage: int):
        self.health = max(0, self.health - damage)
        self.total_damage_taken += damage

    def is_alive(self) -> bool:
        return self.health > 0

    def reset(self):
        """Reset to starting state for a new match."""
        self.health = self.max_health
        self.mp = self.max_mp
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.total_moves = 0
        self.move_usage = {MOVE_ATTACK: 0, MOVE_SPECIAL: 0, MOVE_REGEN: 0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_damage(self, damage: int, move: int):
        self.total_damage_dealt += damage
        self.move_usage[move] += 1
        self.total_moves += 1
