# AI-Powered Fighting Game with Adaptive Learning

An intelligent Python-based fighting game where an AI opponent **learns from past battles**, adapts its strategy over time, and challenges the player using real-time predictions. The system integrates **machine learning**, **persistent gameplay logging**, and **in-depth analytics** to create an evolving, competitive experience.

---

## Objective

To design a turn-based fighting game that features an **AI agent capable of improving over time** using historical battle data. The game focuses on:

- Real-time move prediction using ensemble models
- Continuous learning from past battles (CSV logging)
- Detailed post-game analytics and performance visualization

---

## Gameplay Overview

- Each round: Player selects a move (Attack, Defend, Special)
- The AI:
  - Predicts player's likely move using trained ML models
  - Responds with the best possible counter-move
- Actions consume MP and inflict damage based on predefined rules
- Battle continues for multiple rounds until health depletes

---

## Tech Stack

| Area              | Technology Used                            |
|-------------------|---------------------------------------------|
| Language          | Python 3.x                                  |
| ML Models         | Random Forest, MLPClassifier, Naive Bayes   |
| Data Storage      | CSV (Pandas)                                |
| Visualization     | Matplotlib, Seaborn                         |
| Evaluation        | Confusion Matrix, Classification Report     |
| UI                | Console-based with future GUI expansion     |

---
```
## 🧠 AI System Architecture


 ┌────────────────────────────┐
 │       Player Move          │
 └────────────┬───────────────┘
              ▼
 ┌────────────────────────────┐
 │  CSV Logger (game_logs.csv)│
 └────────────┬───────────────┘
              ▼
 ┌────────────────────────────┐
 │  Ensemble ML Model Trainer │ <── Historical Logs
 └────────────┬───────────────┘
              ▼
 ┌────────────────────────────┐
 │   AI Move Prediction Logic │ ──► Decision Based on Confidence Score
 └────────────┬───────────────┘
              ▼
 ┌────────────────────────────┐
 │    Game Battle Engine      │
 └────────────────────────────┘


ai_game/
├── code/
│   ├── game.py              # Core battle loop and player interaction
│   ├── ai_model.py          # AI training and ensemble prediction
│   ├── logger.py            # Logging each round to CSV
│   ├── matrix.py            # Post-game analysis and metrics
├── logs/
│   └── game_logs.csv        # Persistent game log for training
├── README.md
├── requirements.txt         # Python package requirements

```
## Features:
# Core Gameplay

-Turn-based system with 3 move types:

        1 = Attack

        2 = Defend

        3 = Special

-Health and MP management

-Damage is calculated dynamically

Analytics & Statistics:

-Confusion Matrix between:

-True PlayerMove vs AI Prediction (ML)

-True PlayerMove vs Actual AIMove (In-Game)

-F1 Score, Precision, Recall, and Accuracy

-Heatmaps of move performance

-Most used moves, damage dealt, and win ratio

## Machine Learning Highlights:

-Uses ensemble voting (Random Forest + MLP + Naive Bayes)

-Predicts the next likely PlayerMove and counters accordingly

-Learns continuously from past games via a persistent CSV

-Improves as more rounds are logged

## Installation
```
Clone this repo:
git clone https://github.com/your-username/ai-fighting-game.git
cd ai-fighting-game

Install dependencies:
pip install -r requirements.txt

Run the game:
python code/game.py
```

## Run analytics:

Sample game_logs.csv Format
```
Round	PlayerMove	AIMove	PlayerDamage	AIDamage	PlayerMPUsed	AIMPUsed	Result	ConfidenceScore
1	        1	        3	          20	       10	        5	            3	      Win	      0.85
Result = Win/Loss/Draw (can be used for future model scoring)

ConfidenceScore shows how confident the AI was in its prediction

Example Outputs (from matrix.py)
Confusion Matrix (ML Model Prediction)
True → / Predicted ↓	Attack	Defend	Special
Attack	12	3	1
Defend	2	15	2
Special	0	4	13
Classification Report:

              precision    recall  f1-score   support
     1           0.80       0.75      0.77        16
     2           0.79       0.83      0.81        18
     3           0.85       0.80      0.82        17

```
## Future Improvements

    Add player profiles for persistent performance tracking

    Track and visualize damage dealt per round

    Implement ensemble learning with confidence-weighted votes

    Upgrade to GUI with Pygame or Tkinter

    Add reinforcement learning for real-time adaptation


## License
This project is licensed under the MIT License. Feel free to use it for learning, teaching, or building your own intelligent game agents.

## Acknowledgements
Special thanks to all open-source tools and ML libraries used in this project.

Author
```
Vikrant
AI Developer | Data Science Enthusiast | Game Architect
Contact: vikarantkawadkar2099@gmail.com
```

“Games are the most elevated form of investigation.” — Albert Einstein


---
