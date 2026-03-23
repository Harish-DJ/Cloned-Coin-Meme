"""
reward.py
─────────
The "Secret Sauce" Reward Function for Trend Prediction.
Designed to incentivize:
  1. Directional Accuracy (Down, Neutral, Up)
  2. Early Trend Detection (Catching moves before the peak)
  3. High-Confidence Correctness
  4. Penalizing Random Guessing/Wrong Directions
"""

import numpy as np
from typing import Dict, Tuple

def compute_reward(
    prediction: int,           # 0=Down, 1=Neutral, 2=Up
    actual_trend: int,         # 0=Down, 1=Neutral, 2=Up
    confidence: float,         # 0.0 to 1.0 (from model softmax)
    is_early: bool = False,    # True if caught trend before peak
) -> Tuple[float, Dict]:
    """
    Returns (total_reward, breakdown_dict).
    
    Logic:
    - Basic: +1 for correct direction, -1 for wrong.
    - Bonus: +0.5 for early prediction.
    - Bonus: +0.5 for high-confidence correct (conf > 0.8).
    - Neutral: 0 for neutral results or non-critical errors.
    """
    reward = 0.0
    correct_direction = (prediction == actual_trend)
    wrong_prediction = not correct_direction and (prediction != 1 and actual_trend != 1) # Down vs Up or vice versa
    
    # If neutral was expected and we predicted neutral, it's correct but maybe less valuable than catching a move
    if prediction == 1 and actual_trend == 1:
        reward = 0.5 # Small reward for staying neutral when market is neutral
    elif correct_direction:
        reward += 1.0
        
        if is_early:
            reward += 0.5
            
        if confidence > 0.8:
            reward += 0.5
    elif wrong_prediction:
        reward -= 1.0
    else:
        # One is neutral and the other is a direction (Missed opportunity or false alarm)
        reward = -0.2
        
    breakdown = {
        "correct_direction": float(correct_direction),
        "is_early": float(is_early),
        "high_confidence_bonus": float(correct_direction and confidence > 0.8),
        "wrong_prediction_penalty": float(wrong_prediction),
        "total": round(float(reward), 4)
    }
    
    return reward, breakdown

def explain_reward(breakdown: Dict) -> str:
    """Human-readable explanation of the reward."""
    if breakdown["wrong_prediction_penalty"]:
        return "❌ Wrong direction predicted! Major penalty."
    
    reasons = []
    if breakdown["correct_direction"]:
        reasons.append("✅ Correct trend prediction")
        if breakdown["is_early"]:
            reasons.append("🔥 Early detection bonus!")
        if breakdown["high_confidence_bonus"]:
            reasons.append("💪 High confidence accuracy bonus")
    else:
        reasons.append("⚠️ Missed trend or false signal")
        
    return " | ".join(reasons) if reasons else "Neutral step"

if __name__ == "__main__":
    # Test cases
    print("Testing Reward Function:")
    
    # Case 1: Early correct with high confidence
    r1, b1 = compute_reward(2, 2, 0.9, is_early=True)
    print(f"Early + High Conf Up: {r1} | {explain_reward(b1)}")
    
    # Case 2: Wrong direction
    r2, b2 = compute_reward(0, 2, 0.7, is_early=False)
    print(f"Predicted Down, Actual Up: {r2} | {explain_reward(b2)}")
    
    # Case 3: Correct but late
    r3, b3 = compute_reward(2, 2, 0.6, is_early=False)
    print(f"Correct Up (Late): {r3} | {explain_reward(b3)}")
