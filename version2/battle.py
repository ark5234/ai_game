# battle.py
import random
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class Battle:
    def __init__(self, player, ai):
        self.player = player
        self.ai = ai
        self.round_details = []
        self.move_log = []
        self.previous_moves = []
        self.ai_model = self._train_ai_model()

    def _train_ai_model(self):
        # Placeholder for the AI model (decision tree, random forest, etc.)
        # For now, a simple random model that improves over time
        return None

    def player_turn(self, move):
        if move == "Attack":
            damage = random.randint(50, 150)
            self.ai.take_damage(damage)
            self.player.use_mp(20)
            self.round_details.append(f"Player used Attack. AI took {damage} damage.")
        elif move == "Defense":
            self.round_details.append(f"Player used Defense. No damage dealt.")
        elif move == "Special":
            damage = random.randint(200, 400)
            self.ai.take_damage(damage)
            self.player.use_mp(40)
            self.round_details.append(f"Player used Special. AI took {damage} damage.")
        
        self.previous_moves.append('Attack' if move == 'Attack' else 'Special')
        self._update_ai_decision()

    def ai_turn(self):
        # Predict AI's move based on past player behavior
        if random.random() < 0.7:  # Just an example of adaptive learning
            move = random.choice(["Attack", "Defend", "Special"])
        else:
            move = random.choice(self.previous_moves)  # AI adapts based on history

        damage = random.randint(50, 150) if move == "Attack" else 0
        self.player.take_damage(damage)
        self.ai.use_mp(20)
        self.round_details.append(f"AI used {move}. Player took {damage} damage.")
        self.move_log.append(move)

    def _update_ai_decision(self):
        # Placeholder for updating AI decision-making after each round
        pass

    def get_round_details(self):
        return "\n".join(self.round_details)
