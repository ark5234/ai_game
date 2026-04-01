# AI Fighting Game — v2.0

An intelligent Python-based turn-based fighting game where an AI opponent **learns from past battles**, adapts its strategy using an ensemble ML model with **confidence-weighted voting**, and now features a full **Pygame GUI**, **persistent player profiles**, **per-round damage tracking with plots**, and an optional **reinforcement learning (Q-learning) agent**.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run GUI (Pygame)
```bash
python -m ai_game.gui
```

### Run CLI / Headless
```bash
python -m ai_game.cli --profile YourName
python -m ai_game.cli --profile YourName --rl   # use RL agent
```

### Train / Evaluate RL Agent
```bash
python -m ai_game.train_rl --episodes 2000
python -m ai_game.train_rl --eval --eval-episodes 200
```

---

## Repository Structure

```
ai_game/                  # Main Python package
├── __init__.py
├── __main__.py           # `python -m ai_game` — usage help
├── gui.py                # Pygame GUI entry point
├── cli.py                # CLI / headless entry point
├── train_rl.py           # RL training & evaluation
├── fighter.py            # Fighter base class
├── ai_opponent.py        # Adaptive ML AI (ensemble + confidence-weighted voting)
├── rl_env.py             # Gymnasium-compatible environment (FightEnv)
├── rl_agent.py           # Tabular Q-learning agent
├── battle_engine.py      # Headless battle logic (shared by GUI and CLI)
├── profiles.py           # Per-player profile persistence (JSON)
├── damage_tracker.py     # Per-round tracking (CSV + JSONL)
└── visualize.py          # Matplotlib plots (outputs/)
profiles/                 # JSON player profiles (auto-created)
logs/                     # Match logs: game_logs.csv, match_*.csv, match_*.jsonl
outputs/                  # Plot images (PNG)
requirements.txt
```

---

## Feature Details

### 1. Player Profiles (Persistent Performance Tracking)
- Profiles are stored as `profiles/<name>.json`.
- Tracked stats: games played, wins/losses, total damage dealt/taken, total moves, move usage counts, last played timestamp.
- Create or select a profile in the GUI main menu, or pass `--profile NAME` in CLI mode.
- Updated automatically at the end of every match.

### 2. Per-Round Damage Tracking & Visualisation
- Every round is logged to `logs/game_logs.csv` (global) and `logs/match_<id>.csv` / `logs/match_<id>.jsonl` (per match).
- Fields: round_num, player_move, ai_move, player_damage, ai_damage, HP/MP after & delta, ai_confidence, timestamp.
- After each match, two Matplotlib plots are saved to `outputs/`:
  - `damage_per_round_<id>.png` — grouped bar chart
  - `cumulative_damage_<id>.png` — cumulative line chart
- In the GUI, use the **"Generate Plots"** button on the end screen.
- From code:
  ```python
  from ai_game.visualize import plot_damage_per_round, plot_cumulative_damage
  plot_damage_per_round("<match_id>")
  ```

### 3. Ensemble Learning with Confidence-Weighted Voting
- Models: RandomForestClassifier, MLPClassifier, GaussianNB.
- Each model's vote is weighted by its **maximum predicted probability** (`predict_proba`).
- Final move = `argmax(Σ weight_i × proba_i)`.
- Per-model and ensemble confidences are logged each round and shown in the GUI/CLI.

### 4. Pygame GUI
- **Main Menu**: create/select profile, toggle ML vs RL AI, start game, view stats.
- **In-Game**: HP/MP bars for both fighters, ensemble confidence display, scrolling battle log, on-screen move buttons (also mapped to keys 1/2/3).
- **End Screen**: winner announcement, damage summary, "Generate Plots" button.
- **Stats Screen**: full per-profile statistics.

### 5. Reinforcement Learning Agent (Q-Learning)
#### MDP Definition

| Component       | Details |
|-----------------|---------|
| **Agent**       | Tabular Q-learning policy |
| **Environment** | `FightEnv` (Gymnasium-compatible) |
| **State space** | `MultiDiscrete([5, 5, 5, 5, 4])` — agent HP bin, agent MP bin, opp HP bin, opp MP bin, last opponent move |
| **Action space**| `Discrete(3)` — 0=Attack, 1=Regen, 2=Special |
| **Reward**      | `(dmg_dealt − dmg_taken) / 35 × 0.5` per step; `+1.0` win, `−1.0` loss |
| **Termination** | HP reaches 0 or 50 rounds elapsed |

#### Algorithm & Hyperparameters

| Parameter | Default |
|-----------|---------|
| Learning rate α | 0.1 |
| Discount factor γ | 0.95 |
| Initial ε | 1.0 |
| Minimum ε | 0.05 |
| ε decay | 0.995 per episode |

#### Persistence
- Q-table saved to `outputs/qtable.json` after training.
- Training reward curve saved to `outputs/rl_training_rewards.png`.
- Use `--rl` flag in GUI or CLI to play against the trained RL agent.

---

## Tech Stack

| Area              | Technology                                  |
|-------------------|---------------------------------------------|
| Language          | Python 3.10+                                |
| ML Models         | RandomForest, MLPClassifier, GaussianNB     |
| RL Algorithm      | Tabular Q-learning                          |
| RL Environment    | Gymnasium                                   |
| Data Storage      | CSV, JSONL, JSON (Pandas)                   |
| Visualisation     | Matplotlib, Seaborn                         |
| GUI               | Pygame                                      |

---

## Evaluation

After training for 1 000 episodes the RL agent is evaluated in greedy mode:

```
python -m ai_game.train_rl --episodes 1000 --eval-episodes 200
```

Example output:
```
Win rate  : 120/200 (60.0%)
Mean reward: 0.312 ± 0.581
```
