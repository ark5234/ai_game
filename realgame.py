import pygame
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sys
# Initialize Pygame
pygame.init()

# Screen Settings
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Fighting Game with Adaptive AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Fonts
font = pygame.font.Font(None, 30)
large_font = pygame.font.Font(None, 40)

# Clock
clock = pygame.time.Clock()


# Fighter Class (Player and AI)
class Fighter:
    def __init__(self, name, health=100, mp=50):
        self.name = name
        self.health = health
        self.mp = mp
        self.max_mp = mp
        self.total_damage = 0  # Tracks total damage dealt
        self.move_usage = {1: 0, 2: 0, 3: 0}  # Track move usage
        self.wins = 0
        self.losses = 0

    def attack(self):
        """Basic attack with MP cost."""
        if self.mp >= 10:
            damage = random.randint(10, 20)
            self.mp -= 10
            self.total_damage += damage
            self.move_usage[1] += 1  # Track move usage
            return damage
        return 0

    def special_move(self):
        """Special move with higher damage and MP cost."""
        if self.mp >= 20:
            damage = random.randint(25, 35)
            self.mp -= 20
            self.total_damage += damage
            self.move_usage[2] += 1
            return damage
        return 0

    def regenerate_mp(self):
        """Regenerate MP gradually every round."""
        self.mp = min(self.mp + 5, self.max_mp)
        self.move_usage[3] += 1  # Track move usage

    def track_result(self, won):
        """Track win/loss results."""
        if won:
            self.wins += 1
        else:
            self.losses += 1


    def take_damage(self, damage):
        """Reduce health based on damage taken."""
        self.health = max(0, self.health - damage)


# AI Opponent with Adaptive Learning
class AdaptiveAIOpponent(Fighter):
    def __init__(self, name):
        super().__init__(name, health=100, mp=50)
        self.train_data = []
        self.target_data = []
        
        # Ensemble models
        self.rf_model = RandomForestClassifier(n_estimators=10)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
        self.nb_model = GaussianNB()
        
        self.conf_matrix = None
        self.f1_score = 0.0

    def predict_move(self, player_action):
     """Predict AI's next move along with confidence score."""
     if len(self.train_data) > 1:
        X_test = np.array([[player_action]])
        
        # Get prediction probabilities
        rf_probs = self.rf_model.predict_proba(X_test)[0]
        nn_probs = self.nn_model.predict_proba(X_test)[0]
        nb_probs = self.nb_model.predict_proba(X_test)[0]
        # Average the confidence scores
        final_probs = (rf_probs + nn_probs + nb_probs) / 3
        best_move = np.argmax(final_probs) + 1  # Get the move with highest probability
        # Check if final_probs contains NaN values
        if np.isnan(final_probs).any():
            confidence = 33.3  # Default confidence when AI lacks training
        else:
            confidence = round(max(final_probs) * 100, 2)  # Convert to percentage
        return best_move, confidence
     else:
        return random.choice([1, 2, 3]), 33.3  # Default confidence


    def train(self):
      """Train the AI models using collected data."""
      if len(self.train_data) > 1:
        X = np.array(self.train_data)
        y = np.array(self.target_data)

        # Train models
        self.rf_model.fit(X, y)
        self.nn_model.fit(X, y)
        self.nb_model.fit(X, y)

        print("AI Models trained successfully!")  # Debugging)

    def update_metrics(self, player_move, ai_move):
     """Update AI performance metrics (Confusion Matrix & F1 Score)."""
     self.train_data.append([player_move])  # Store player's move
     self.target_data.append(ai_move)  # Store AI's move

     if len(self.train_data) > 1:  # Ensure enough data for training
        self.train()  # Train models

        y_true = np.array(self.target_data)
        y_pred = self.rf_model.predict(self.train_data) if len(self.train_data) > 1 else np.zeros_like(y_true)

        if len(np.unique(y_true)) > 1:  # Ensure multiple classes exist
            self.conf_matrix = confusion_matrix(y_true, y_pred)
            self.f1_score = f1_score(y_true, y_pred, average="weighted", zero_division=1)
        else:
            self.conf_matrix = None  # Not enough variation in moves
            self.f1_score = 0.0  # Default to 0.0 if not enough data

# Draw UI Elements
def draw_health_bar(x, y, health):
    pygame.draw.rect(screen, RED, (x, y, 200, 20))
    pygame.draw.rect(screen, GREEN, (x, y, 2 * health, 20))
    display_text(f"{health}/100", x + 80, y, font, WHITE)

def draw_mp_bar(x, y, mp):
    pygame.draw.rect(screen, GRAY, (x, y, 200, 10))
    pygame.draw.rect(screen, BLUE, (x, y, 2 * mp, 10))
    display_text(f"{mp}/50", x + 70, y, font, WHITE)

def display_text(text, x, y, font, color=WHITE):
    screen.blit(font.render(text, True, color), (x, y))

def calculate_probabilities(player, opponent):
    total_health = player.health + opponent.health
    if total_health == 0:
        return 0, 0
    player_prob = (player.health / total_health) * 100
    ai_prob = (opponent.health / total_health) * 100
    return round(player_prob, 1), round(ai_prob, 1)


def game_over_screen(winner, player, ai):
    """Displays game-over screen with AI statistics."""
    screen.fill(BLACK)
    display_text(f"{winner} Wins!", SCREEN_WIDTH // 2 - 100, 200, large_font, YELLOW)
    display_text("Final Statistics", SCREEN_WIDTH // 2 - 120, 260, large_font, WHITE)
    display_text(f"Total Damage Dealt: {player.total_damage}", SCREEN_WIDTH // 2 - 100, 300, font, GREEN)
    display_text(f"Most Used Move: {max(player.move_usage, key=player.move_usage.get)}", SCREEN_WIDTH // 2 - 100, 330, font, GREEN)

    # Display AI Metrics
    if ai.conf_matrix is not None:
        display_text(f"AI F1 Score: {round(ai.f1_score, 2)}", SCREEN_WIDTH // 2 - 100, 360, font, BLUE)
        display_text(f"AI Confusion Matrix:", SCREEN_WIDTH // 2 - 100, 390, font, BLUE)
        for i, row in enumerate(ai.conf_matrix):
            display_text(f"{row}", SCREEN_WIDTH // 2 - 100, 420 + i * 30, font, BLUE)
    else:
        display_text("AI Metrics Not Available", SCREEN_WIDTH // 2 - 100, 360, font, RED)

    pygame.display.update()

    # **Wait for user input before closing**
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_r:  # Restart
                    battle()  # Restart the game
                    return  # Exit function to start new game


# Game Loop
def battle():
    player = Fighter("Player")
    ai = AdaptiveAIOpponent("AI")
    logs = []
    running = True
    game_over = False
    winner = ""

    while running:
        screen.fill(BLACK)

        # Draw health bars
        draw_health_bar(50, 50, player.health)
        draw_health_bar(550, 50, ai.health)
        draw_mp_bar(50, 80, player.mp)
        draw_mp_bar(550, 80, ai.mp)
        

        # Display Instructions
        display_text("Press 1: Attack | 2: Special Move | 3: Regenerate MP", 50, 120, font)
        display_text("Move Details:", 50, 240, font, YELLOW)
        display_text("1: Attack (MP: 10, Damage: 10-20)", 50, 260, font, WHITE)
        display_text("2: Special Move (MP: 20, Damage: 25-35)", 50, 280, font, WHITE)
        display_text("3: Regenerate MP (+5 MP)", 50, 300, font, WHITE)


        if game_over:
            game_over_screen(winner, player, ai)


        # Handle Player Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1 and player.mp >= 10:
                    damage = player.attack()
                    ai.take_damage(damage)
                    logs.append(f"Player attacked for {damage} damage!")

                elif event.key == pygame.K_2 and player.mp >= 20:
                    damage = player.special_move()
                    ai.take_damage(damage)
                    logs.append(f"Player used special move for {damage} damage!")

                elif event.key == pygame.K_3:
                    player.regenerate_mp()
                    logs.append("Player regenerated MP!")

                # AI Move
                ai_move, ai_confidence = ai.predict_move(random.choice([1, 2, 3]))
                logs.append(f"AI chose move {ai_move} with {ai_confidence}% confidence!")
                if ai_move == 1:
                    damage = ai.attack()
                    player.take_damage(damage)
                    logs.append(f"AI attacked for {damage} damage!")

                elif ai_move == 2:
                    damage = ai.special_move()
                    player.take_damage(damage)
                    logs.append(f"AI used special move for {damage} damage!")

                elif ai_move == 3:
                    ai.regenerate_mp()
                    logs.append(f"AI regenerated MP! Current MP: {ai.mp}")
                # AI prioritizes attacking more aggressively
                if ai.mp >= 20:
                    damage = ai.special_move()
                    player.take_damage(damage)
                    logs.append(f"AI used special move for {damage} damage!")
                elif ai.mp >= 10:
                    damage = ai.attack()
                    player.take_damage(damage)
                    logs.append(f"AI attacked for {damage} damage!")
                else:
                    ai.regenerate_mp()
                    logs.append("AI regenerated MP!")

                    
                ai.update_metrics(player_move=max(player.move_usage, key=player.move_usage.get), ai_move=ai_move)


                # **Check for Game Over**
                if player.health <= 0:
                    logs.append("Player is defeated!")
                    player.track_result(won=False)
                    ai.track_result(won=True)
                    winner = "AI"
                    game_over = True  # Don't exit loop—show game over screen

                elif ai.health <= 0:
                    logs.append("AI is defeated!")
                    player.track_result(won=True)
                    ai.track_result(won=False)
                    winner = "Player"
                    game_over = True  # Show game over screen

        # Display Logs
        pygame.draw.rect(screen, GRAY, (50, 600, 800, 180))
        display_text("Live Log", 60, 610, font, BLACK)
        for i, log in enumerate(logs[-5:]):  # Show last 5 moves
            display_text(log, 60, 640 + i * 20, font, BLACK)

        # Display Player Stats
        display_text(f"Total Damage Dealt: {player.total_damage}", 50, 160, font)
        display_text(f"Win/Loss: {player.wins}/{player.losses}", 50, 180, font)
        display_text(f"Most Used Move: {max(player.move_usage, key=player.move_usage.get)}", 50, 200, font)

        pygame.display.update()
        clock.tick(30)

# Run Game
battle()
