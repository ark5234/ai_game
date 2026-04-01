"""Matplotlib visualisation utilities.

All plots are saved to outputs/ and the saved path is returned.
Uses the non-interactive 'Agg' backend so it works in headless environments.
"""

import csv
import os
from typing import List, Optional

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_match_csv(match_id: str) -> Optional[List[dict]]:
    path = os.path.join(LOGS_DIR, f"match_{match_id}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _ensure_outputs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_damage_per_round(match_id: str) -> Optional[str]:
    """
    Bar chart of player and AI damage per round.
    Saves to outputs/damage_per_round_<match_id>.png.
    Returns the save path, or None if no data found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_match_csv(match_id)
    if not data:
        return None
    _ensure_outputs()

    rounds = [int(r["round_num"]) for r in data]
    player_dmg = [int(r["player_damage"]) for r in data]
    ai_dmg = [int(r["ai_damage"]) for r in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([r - 0.2 for r in rounds], player_dmg, 0.4, label="Player", color="steelblue")
    ax.bar([r + 0.2 for r in rounds], ai_dmg, 0.4, label="AI", color="tomato")
    ax.set_xlabel("Round")
    ax.set_ylabel("Damage")
    ax.set_title(f"Damage Per Round — Match {match_id}")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, f"damage_per_round_{match_id}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cumulative_damage(match_id: str) -> Optional[str]:
    """
    Line chart of cumulative damage over rounds.
    Saves to outputs/cumulative_damage_<match_id>.png.
    Returns the save path, or None if no data found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_match_csv(match_id)
    if not data:
        return None
    _ensure_outputs()

    rounds = [int(r["round_num"]) for r in data]
    cum_player, cum_ai = [], []
    p_sum = a_sum = 0
    for r in data:
        p_sum += int(r["player_damage"])
        a_sum += int(r["ai_damage"])
        cum_player.append(p_sum)
        cum_ai.append(a_sum)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, cum_player, marker="o", label="Player Cumulative", color="steelblue")
    ax.plot(rounds, cum_ai, marker="s", label="AI Cumulative", color="tomato")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Damage")
    ax.set_title(f"Cumulative Damage — Match {match_id}")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, f"cumulative_damage_{match_id}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rl_rewards(episode_rewards: List[float]) -> Optional[str]:
    """
    Plot RL training episode rewards with a moving-average trend line.
    Saves to outputs/rl_training_rewards.png.
    Returns the save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not episode_rewards:
        return None
    _ensure_outputs()

    episodes = list(range(1, len(episode_rewards) + 1))
    window = min(50, len(episode_rewards))
    moving_avg = np.convolve(
        episode_rewards, np.ones(window) / window, mode="valid"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(
        list(range(window, len(episode_rewards) + 1)),
        moving_avg,
        color="tomato",
        linewidth=2,
        label=f"Moving avg (window={window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("RL Training — Episode Rewards")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "rl_training_rewards.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
