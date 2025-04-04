# player_ai.py
import random

class Player:
    def __init__(self, name, health=1000, mp=100):
        self.name = name
        self.health = health
        self.mp = mp
        self.description = "The brave warrior fighting to reclaim lost lands."

    def take_damage(self, damage):
        self.health -= damage

    def heal(self, amount):
        self.health += amount

    def use_mp(self, cost):
        self.mp -= cost

class AI:
    def __init__(self, name, health=1000, mp=100):
        self.name = name
        self.health = health
        self.mp = mp
        self.description = "The adaptive AI opponent using advanced strategies."

    def take_damage(self, damage):
        self.health -= damage

    def heal(self, amount):
        self.health += amount

    def use_mp(self, cost):
        self.mp -= cost
