import pygame
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# Initialize PyGame
pygame.init()

# Screen Dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Fighting Game with Adaptive AI and Metrics")

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

# Load dataset containing possible player and AI move combinations
dataset_path = "player_ai_moves_dataset.csv"
df = pd.read_csv(dataset_path)

# Classes
class Fighter:
    def __init__(self, name, health, stamina, mp):
        self.name = name
        self.health = health
        self.stamina = stamina
        self.mp = mp

    def attack(self):
        return random.randint(10, 20), 10  # Damage, MP Cost

    def defend(self):
        return random.randint(5, 15), 5  # Block, MP Cost

    def special_move(self):
        return random.randint(15, 30), 20  # Damage, MP Cost

class AdaptiveAIOpponent(Fighter):
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.mp = 100
        self.history = []
        self.train_data = []
        self.target_data = []
        self.model = RandomForestClassifier()  # Or any AI model
        self.conf_matrix = None
        self.f1 = 0.0

    def reset_metrics(self):
        """Reset all metrics for a new game."""
        self.history = []
        self.train_data = []
        self.target_data = []
        self.conf_matrix = None
        self.f1 = 0.0
  # To store ai_action as target

    def predict_move(self, player_action):
        # Use trained decision tree to predict the AI move
        if len(self.train_data) > 1:
            # Predict the next move based on player_action
            prediction = self.model.predict([[player_action]])[0]
            return prediction
        else:
            # Default behavior if no model is trained yet (simple heuristic)
            if player_action == 1:  # Player attacks
                return 2  # Defend
            elif player_action == 2:  # Player defends
                return 3  # Special move
            else:  # Player special move
                return 1  # Attack

    def update_metrics(self):
        if len(self.history) > 0:
            player_actions, ai_actions = zip(*self.history)
            self.conf_matrix = confusion_matrix(player_actions, ai_actions, labels=[1, 2, 3])
            self.f1 = f1_score(player_actions, ai_actions, average="weighted", zero_division=1)
        else:
        # No actions yet, reset metrics
            self.conf_matrix = None
            self.f1 = 0.0


    def train(self):
        if len(self.train_data) > 1:  # Train model after some data is collected
            self.model.fit(self.train_data, self.target_data)

# Functions
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

def draw_dashboard(ai, player, player_description, ai_description, logs, probabilities):
    # Metrics Panel
    pygame.draw.rect(screen, BLUE, (900, 0, 300, 800))
    display_text("Metrics", 910, 20, large_font, YELLOW)

    if ai.conf_matrix is not None:
        display_text("Confusion Matrix:", 910, 70, font)
        for i, row in enumerate(ai.conf_matrix):
            display_text(f"{row}", 910, 100 + i * 30, font)

    display_text(f"F1 Score: {ai.f1:.2f}", 910, 200, font)
    display_text(f"Win Probability:", 910, 250, font)
    display_text(f"Player: {probabilities[0]}%", 910, 280, font, GREEN)
    display_text(f"AI: {probabilities[1]}%", 910, 310, font, RED)

    display_text(f"Player: {player_description}", 50, 500, font, YELLOW)
    display_text(f"AI: {ai_description}", 50, 540, font, YELLOW)

    # Live Log Panel
    pygame.draw.rect(screen, GRAY, (50, 600, 800, 180))
    display_text("Live Log", 60, 610, font, BLACK)
    for i, log in enumerate(logs[-4:]):  # Show last 4 moves
        display_text(log, 60, 640 + i * 20, font, BLACK)

def create_restart_button():
    pygame.draw.rect(screen, GRAY, (1000, 730, 120, 50))
    display_text("Restart", 1020, 745, font, BLACK)

def battle(player, opponent):
    running = True
    round_number = 1
    logs = []

    while running:
        screen.fill(BLACK)

        # Display Health, MP, and Round
        draw_health_bar(50, 50, player.health)
        draw_health_bar(550, 50, opponent.health)
        draw_mp_bar(50, 80, player.mp)
        draw_mp_bar(550, 80, opponent.mp)
        display_text(f"Round {round_number}", 350, 20, large_font, YELLOW)

        # Display Player and Opponent Names
        display_text(f"{player.name}", 50, 20, font)
        display_text(f"{opponent.name}", 550, 20, font)

        # Player Options
        display_text("Press 1: Attack (10 MP) | 2: Defend (5 MP) | 3: Special Move (20 MP)", 50, 100, font)

        player_description = "No move yet"
        ai_description = "No move yet"

        probabilities = calculate_probabilities(player, opponent)

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                player_action = None
                ai_action = None
                if event.key == pygame.K_1:
                    if player.mp >= 10:
                        damage, cost = player.attack()
                        opponent.health -= damage
                        player.mp -= cost
                        player_action = 1
                        player_description = f"Player attacks and deals {damage} damage"
                    else:
                        player_description = "Not enough MP to attack!"

                elif event.key == pygame.K_2:
                    if player.mp >= 5:
                        block, cost = player.defend()
                        player.mp -= cost
                        player_action = 2
                        player_description = f"Player defends, reducing damage by {block}"
                    else:
                        player_description = "Not enough MP to defend!"

                elif event.key == pygame.K_3:
                    if player.mp >= 20:
                        special, cost = player.special_move()
                        opponent.health -= special
                        player.mp -= cost
                        player_action = 3
                        player_description = f"Player uses special move and deals {special} damage"
                    else:
                        player_description = "Not enough MP to use special move!"

                # AI's Turn
                if player_action:
                    ai_action = opponent.predict_move(player_action)
                    if ai_action == 1 and opponent.mp >= 10:
                        damage, cost = opponent.attack()
                        player.health -= damage
                        opponent.mp -= cost
                        ai_description = f"AI attacks and deals {damage} damage"
                    elif ai_action == 2 and opponent.mp >= 5:
                        block, cost = opponent.defend()
                        opponent.mp -= cost
                        ai_description = f"AI defends, reducing damage by {block}"
                    elif ai_action == 3 and opponent.mp >= 20:
                        special, cost = opponent.special_move()
                        player.health -= special
                        opponent.mp -= cost
                        ai_description = f"AI uses special move and deals {special} damage"
                    else:
                        ai_description = "AI has insufficient MP for any action!"

                    # Log actions for metrics
                    opponent.history.append((player_action, ai_action))
                    opponent.train_data.append([player_action])
                    opponent.target_data.append(ai_action)

                    # Train AI model after a few rounds
                    opponent.train()

                # Add moves to logs
                logs.append(f"Round {round_number}: {player_description} | {ai_description}")

        # Update AI Metrics after every round
        opponent.update_metrics()

        # Draw Dashboard and Restart Button
        draw_dashboard(opponent, player, player_description, ai_description, logs, probabilities)
        create_restart_button()

        # Check for Restart Button Click
        if pygame.mouse.get_pressed()[0]:  # Restart Button Click
            mouse_pos = pygame.mouse.get_pos()
            if 1000 <= mouse_pos[0] <= 1120 and 730 <= mouse_pos[1] <= 780:
                player = Fighter("Hero", 100, 100, 50)  # Reset player stats
                ai = AdaptiveAIOpponent("Adaptive AI")  # Reset AI
                ai.reset_metrics()  # Clear all AI metrics for the new game
                return battle(player, ai)


        # Check for End Conditions
        if player.health <= 0 or opponent.health <= 0:
            winner = player.name if player.health > 0 else opponent.name
            display_text(f"{winner} wins!", 350, 300, large_font, YELLOW)
            pygame.display.flip()
            pygame.time.delay(3000)
            running = False

        # Check if both players are out of MP
        if player.mp <= 0 and opponent.mp <= 0:
            display_text("Both players are out of MP! Stalemate!", 350, 300, large_font, YELLOW)
            pygame.display.flip()
            pygame.time.delay(3000)
            running = False

        round_number += 1
        pygame.display.flip()
        clock.tick(30)

    # After the game ends, display the confusion matrix and F1 score
    screen.fill(BLACK)
    winner = player.name if player.health > 0 else opponent.name
    display_text(f"{winner} wins!" if player.health > 0 or opponent.health > 0 else "It's a draw!", 350, 200, large_font, YELLOW)

    if opponent.conf_matrix is not None:
        display_text("Confusion Matrix:", 350, 250, font)
        for i, row in enumerate(opponent.conf_matrix):
            display_text(f"{row}", 350, 280 + i * 30, font)
    display_text(f"F1 Score: {opponent.f1:.2f}", 350, 400, font)
    pygame.display.flip()
    pygame.time.delay(5000)


# Main Loop
if __name__ == "__main__":
    player = Fighter("Hero", 100, 100, 50)
    ai = AdaptiveAIOpponent("Adaptive AI")
    battle(player, ai)
