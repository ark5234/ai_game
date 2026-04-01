"""CLI / headless entry point.

Usage:
    python -m ai_game.cli [--profile NAME] [--rl]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="AI Fighting Game — CLI mode")
    parser.add_argument("--profile", default="Player", help="Player profile name")
    parser.add_argument(
        "--rl", action="store_true", help="Use RL agent instead of ML ensemble"
    )
    args = parser.parse_args()

    from .fighter import Fighter
    from .ai_opponent import AdaptiveAIOpponent
    from .rl_agent import QLearningAgent
    from .profiles import PlayerProfile
    from .damage_tracker import MatchTracker
    from .battle_engine import BattleEngine
    from .visualize import plot_damage_per_round, plot_cumulative_damage

    print("=== AI Fighting Game (CLI Mode) ===")
    profile = PlayerProfile.load_or_create(args.profile)
    print(
        f"Profile: {profile.name}  |  "
        f"Games: {profile.games_played}  |  "
        f"Wins: {profile.wins}  |  "
        f"Losses: {profile.losses}"
    )
    if profile.last_played:
        print(f"Last played: {profile.last_played}")

    player = Fighter(profile.name)
    ai = AdaptiveAIOpponent("AI")
    ai.load_history()

    rl_agent = None
    if args.rl:
        rl_agent = QLearningAgent()
        if rl_agent.load():
            print("RL agent loaded from disk.")
        else:
            print("No saved RL agent; using untrained agent.")

    tracker = MatchTracker()
    engine = BattleEngine(player, ai, rl_agent=rl_agent, use_rl=args.rl, tracker=tracker)

    ai_type = "RL" if args.rl else "ML-Ensemble"
    print(f"\nAI type: {ai_type}")
    print("Moves: 1=Attack (MP:10, dmg:10-20) | 2=Special (MP:20, dmg:25-35) | 3=Regen (+5 MP)")
    print("Type 'q' to quit.\n")

    while not engine.game_over:
        print(
            f"\nRound {engine.round_num + 1}  "
            f"— Player HP:{player.health:3d} MP:{player.mp:2d}  "
            f"| AI HP:{ai.health:3d} MP:{ai.mp:2d}"
        )
        choice = input("  Your move (1/2/3): ").strip()
        if choice.lower() == "q":
            print("Quit.")
            sys.exit(0)
        if choice not in ("1", "2", "3"):
            print("  Invalid — enter 1, 2, or 3.")
            continue

        move = int(choice)
        if move == 1 and player.mp < 10:
            print("  Not enough MP for Attack. Use Regen (3) first.")
            continue
        if move == 2 and player.mp < 20:
            print("  Not enough MP for Special. Use Regen (3) first.")
            continue

        engine.execute_player_move(move)
        for entry in engine.last_log_entries:
            print(" ", entry)

    # ------------------------------------------------------------------
    print(f"\n{'=' * 40}")
    print(f"  GAME OVER — {engine.winner} wins!")
    print(f"  Player : dealt {player.total_damage_dealt} dmg, took {player.total_damage_taken} dmg")
    print(f"  AI     : dealt {ai.total_damage_dealt} dmg, took {ai.total_damage_taken} dmg")
    print(f"{'=' * 40}\n")

    # Save match logs and profile
    tracker.save()
    profile.record_match(
        won=(engine.winner == profile.name),
        damage_dealt=player.total_damage_dealt,
        damage_taken=player.total_damage_taken,
        moves=player.total_moves,
        move_usage=player.move_usage,
    )
    profile.save()
    print(f"Profile saved: {profile.name} ({profile.wins}W / {profile.losses}L)")

    # Generate and save plots
    p1 = plot_damage_per_round(tracker.match_id)
    p2 = plot_cumulative_damage(tracker.match_id)
    for p in (p1, p2):
        if p:
            print(f"Plot saved: {p}")


if __name__ == "__main__":
    main()
