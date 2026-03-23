"""
train.py
────────
Training loop for the Trend Predictor RL Agent.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend — safe for Flask
import matplotlib.pyplot as plt
from environment import TrendEnvironment
from agent import TrendPredictorAgent

def train_agent(n_episodes=50, steps_per_episode=200):
    # Cap environment size to prevent memory issues
    total_steps = min(n_episodes * steps_per_episode, 5000)
    env = TrendEnvironment(n_steps=total_steps)
    agent = TrendPredictorAgent(n_features=10, n_actions=3)
    
    rewards_history = []
    epsilon_history = []
    
    print(f"Starting training for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps_done = 0
        done = False
        
        while not done and steps_done < steps_per_episode:
            action, conf = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action, confidence=conf)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
            steps_done += 1

        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        if (ep + 1) % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:])
            print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    # Save the trained agent
    agent.save("trend_agent.pth")
    print("Training complete. Agent saved.")
    
    # Plot results (saved to file, never shown)
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        
        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.close('all')
        print("Training plots saved to training_results.png")
    except Exception as e:
        print(f"Plot save skipped: {e}")
    
    return agent

if __name__ == "__main__":
    train_agent(n_episodes=20)

