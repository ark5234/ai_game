# game_ui.py
import tkinter as tk
from tkinter import Toplevel
from code.version2.player_ai import Player, AI
from code.version2.battle import Battle
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import random

class GameUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Battle Game")
        self.master.geometry("800x600")

        self.player = Player("Player")
        self.ai = AI("Adaptive AI")
        self.battle = Battle(self.player, self.ai)

        self.start_button = tk.Button(self.master, text="Start Battle", command=self.start_battle)
        self.start_button.pack()

        self.show_calculations_button = tk.Button(self.master, text="Show Calculations", command=self.show_calculations)
        self.show_calculations_button.pack()

        self.restart_button = tk.Button(self.master, text="Restart Game", command=self.restart_game)
        self.restart_button.pack()

        self.f1_label = tk.Label(self.master, text="F1 Score: 0.0")
        self.f1_label.pack()

        self.hp_label = tk.Label(self.master, text="Player HP: 1000 | AI HP: 1000")
        self.hp_label.pack()

        self.mp_label = tk.Label(self.master, text="Player MP: 100 | AI MP: 100")
        self.mp_label.pack()

    def start_battle(self):
        move = random.choice(["Attack", "Defense", "Special"])  # Example move
        self.battle.player_turn(move)
        self.battle.ai_turn()
        self.update_ui()

    def show_calculations(self):
        calculations_window = Toplevel(self.master)
        calculations_window.title("Round Calculations")
        
        calculations_text = self.battle.get_round_details()
        
        label = tk.Label(calculations_window, text=calculations_text, justify="left")
        label.pack()

    def update_ui(self):
        # Update UI based on the battle status
        self.hp_label.config(text=f"Player HP: {self.player.health} | AI HP: {self.ai.health}")
        self.mp_label.config(text=f"Player MP: {self.player.mp} | AI MP: {self.ai.mp}")
        self.f1_label.config(text="F1 Score: 0.0")  # Example placeholder F1 score
        self.master.update()

    def restart_game(self):
        self.player.health = 1000
        self.player.mp = 100
        self.ai.health = 1000
        self.ai.mp = 100
        self.update_ui()
