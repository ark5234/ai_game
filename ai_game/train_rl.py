"""RL training and evaluation entry point.

Usage:
    python -m ai_game.train_rl                     # train 1 000 episodes
    python -m ai_game.train_rl --episodes 5000
    python -m ai_game.train_rl --eval              # evaluate saved agent
    python -m ai_game.train_rl --eval-episodes 200
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="RL training for AI Fighting Game")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate only (requires saved agent)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100, help="Episodes for evaluation"
    )
    args = parser.parse_args()

    from .rl_env import FightEnv
    from .rl_agent import QLearningAgent
    from .visualize import plot_rl_rewards

    agent = QLearningAgent()
    env = FightEnv()

    if args.eval:
        if not agent.load():
            print("No saved Q-table found. Run training first.")
            return
        print(f"Evaluating saved agent over {args.eval_episodes} episodes...")
        _evaluate(env, agent, args.eval_episodes)
        return

    # Load existing checkpoint if available (continue training)
    loaded = agent.load()
    if loaded:
        print(f"Resuming from saved agent (ε={agent.epsilon:.3f}, "
              f"{len(agent.q_table)} states explored).")
    else:
        print("Starting fresh Q-learning agent.")

    print(f"Training for {args.episodes} episodes "
          f"(α={agent.alpha}, γ={agent.gamma}, ε→{agent.epsilon_min})...")
    _train(env, agent, args.episodes)

    agent.save()
    print(f"\nQ-table saved. Total states explored: {len(agent.q_table)}")
    print(f"Last 10 episode rewards: "
          f"{[round(r, 3) for r in agent.episode_rewards[-10:]]}")

    plot_path = plot_rl_rewards(agent.episode_rewards)
    if plot_path:
        print(f"Training reward plot saved: {plot_path}")

    print("\nRunning post-training evaluation...")
    _evaluate(env, agent, min(args.eval_episodes, 200))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train(env, agent, n_episodes: int):
    import numpy as np

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0

        while True:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break

        agent.episode_rewards.append(total_reward)
        agent.decay_epsilon()

        if ep % 100 == 0:
            mean_r = float(np.mean(agent.episode_rewards[-100:]))
            print(
                f"  Episode {ep:5d}/{n_episodes}  "
                f"mean(last 100)={mean_r:+.3f}  ε={agent.epsilon:.3f}"
            )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _evaluate(env, agent, n_episodes: int):
    import numpy as np

    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy policy during evaluation

    wins = 0
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        while True:
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            if terminated or truncated:
                if reward >= 1.0:
                    wins += 1
                break
        rewards.append(total_r)

    agent.epsilon = old_eps

    win_rate = 100 * wins / n_episodes
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f"  Win rate  : {wins}/{n_episodes} ({win_rate:.1f}%)")
    print(f"  Mean reward: {mean_r:.3f} ± {std_r:.3f}")


if __name__ == "__main__":
    main()
