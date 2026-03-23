"""
agent.py
────────
Deep Q-Network (DQN) agent for Trend Prediction.
Features:
  • MLP architecture (Input 10 -> Dense 64 -> Dense 32 -> Output 3)
  • Experience Replay Buffer
  • Target Network for stable training
  • ε-greedy exploration
  • Action Explainability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

class QNetwork(nn.Module):
    def __init__(self, n_features, n_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

class TrendPredictorAgent:
    def __init__(self, n_features=10, n_actions=3):
        self.n_features = n_features
        self.n_actions = n_actions
        
        self.policy_net = QNetwork(n_features, n_actions)
        self.target_net = QNetwork(n_features, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Returns (action, confidence).
        """
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            return action, 0.0 # Low confidence during exploration
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            # Use softmax of Q-values as a proxy for confidence
            probs = torch.softmax(q_values, dim=1)
            action = torch.argmax(q_values).item()
            confidence = probs[0, action].item()
            return action, confidence

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(done_batch)
        
        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * GAMMA * max_next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def explain_action(self, state: np.ndarray, action: int, confidence: float) -> Dict:
        """
        Explain why the agent chose this action based on dominant features.
        """
        actions = ["Downward", "Neutral", "Upward"]
        chosen = actions[action]
        
        # Simple rule-based explanation for hackathon demo
        feat_names = [
            "Sentiment", "Mentions", "Growth", "Engagement", "Spike", 
            "Platforms", "Momentum", "Influence", "Hype", "Volatility"
        ]
        
        explanation = f"Predicting {chosen} trend with {confidence:.1%} confidence."
        
        # Find strongest signal in current state
        strongest_idx = np.argmax(np.abs(state))
        strongest_feat = feat_names[strongest_idx]
        
        if state[strongest_idx] > 0.5:
             explanation += f" Driven primarily by strong {strongest_feat} signals."
        elif state[strongest_idx] < -0.5:
             explanation += f" Triggered by negative {strongest_feat} sentiment."
             
        return {
            "prediction": chosen,
            "confidence": confidence,
            "explanation": explanation,
            "strongest_signal": strongest_feat
        }

    def save(self, path="trend_agent.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="trend_agent.pth"):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = EPSILON_END

if __name__ == "__main__":
    agent = TrendPredictorAgent()
    state = np.random.rand(10)
    action, conf = agent.select_action(state, training=False)
    explanation = agent.explain_action(state, action, conf)
    
    print("=== Agent Test ===")
    print(f"Action: {action} ({explanation['prediction']})")
    print(f"Conf  : {conf:.1%}")
    print(f"Reason: {explanation['explanation']}")
