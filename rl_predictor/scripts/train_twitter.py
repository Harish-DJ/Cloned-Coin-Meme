"""
train_twitter.py
────────────────
Trains the DQN agent on real Twitter social signal data from
mock_twitter_200 (1).json instead of the synthetic simulator.

Run:
    python train_twitter.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add both the web and core directories to the path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "..", "core"))
sys.path.append(os.path.join(base_dir, "..", "loaders"))

from environment import TrendEnvironment
from agent import TrendPredictorAgent

def train_on_twitter(n_episodes=80):
    print("=" * 55)
    print("  HypeSense -- Training on Real Twitter Signal Data")
    print("=" * 55)

    env = TrendEnvironment(use_twitter=True)
    agent = TrendPredictorAgent(n_features=10, n_actions=3)

    rewards_history   = []
    accuracy_history  = []
    epsilon_history   = []

    for ep in range(n_episodes):
        state    = env.reset()
        total_reward = 0
        correct  = 0
        total    = 0
        done     = False

        while not done:
            action, conf = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action, confidence=conf)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            actual = int(info["actual_trend"])
            # actual_trend in info is NEXT step's label; compare against current action
            # Use the pre-step label (stored before step) for accuracy
            state = next_state
            total_reward += reward
            total += 1

        # End-of-episode accuracy: replay the episode in inference mode
        state = env.reset()
        done  = False
        ep_correct = 0
        ep_total   = 0
        while not done:
            action, conf = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action, confidence=conf)
            # The actual trend we are predicting is the label at the CURRENT step
            # info["actual_trend"] is the next step's label, so we track reward sign
            ep_correct += 1 if reward > 0 else 0
            ep_total   += 1
            state = next_state

        acc = ep_correct / max(ep_total, 1) * 100
        rewards_history.append(total_reward)
        accuracy_history.append(acc)
        epsilon_history.append(agent.epsilon)

        if (ep + 1) % 10 == 0 or ep == 0:
            avg_rew = np.mean(rewards_history[-10:])
            avg_acc = np.mean(accuracy_history[-10:])
            print(f"  Episode {ep+1:3d}/{n_episodes} | "
                  f"Avg Reward: {avg_rew:+6.2f} | "
                  f"Accuracy: {avg_acc:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "trend_agent.pth")
    agent.save(model_path)
    print(f"\n  Training complete. Agent saved to {model_path}")

    # Plot
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(rewards_history, color='#00ff88')
        axes[0].set_title("Total Reward per Episode")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')

        axes[1].plot(accuracy_history, color='#00e5ff')
        axes[1].set_title("Accuracy per Episode (%)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Accuracy %")
        axes[1].set_facecolor('#0a0a0f')

        axes[2].plot(epsilon_history, color='#ff00ff')
        axes[2].set_title("Epsilon Decay")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Epsilon")
        axes[2].set_facecolor('#0a0a0f')

        for ax in axes:
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results.png"), dpi=100, facecolor='#0a0a0f')
        plt.close('all')
        print("  Training plots saved to training_results.png")
    except Exception as e:
        print(f"  Plot save skipped: {e}")

    final_acc = np.mean(accuracy_history[-10:])
    print(f"\n  Final 10-episode average accuracy: {final_acc:.1f}%")
    return agent


if __name__ == "__main__":
    train_on_twitter(n_episodes=80)
