"""Entry point: python -m ai_game  (defaults to CLI help)."""
import sys


def main():
    print("AI Fighting Game v2.0")
    print("  python -m ai_game.gui        — Pygame GUI")
    print("  python -m ai_game.cli        — CLI / headless mode")
    print("  python -m ai_game.train_rl   — RL training / evaluation")


if __name__ == "__main__":
    main()
