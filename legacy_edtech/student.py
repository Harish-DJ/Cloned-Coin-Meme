"""
student.py
──────────
Student profiles with:
  • Learner type archetypes (fast / slow / topic-weak / balanced)
  • Per-topic mastery tracking with forgetting curve (retention decay)
  • Rolling accuracy history
  • Performance snapshot for explainability

The SimulatedStudent is used during RL training.
The StudentProfile is the real-world object — it is serializable
and passed in/out of the RL environment for live sessions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from curriculum import N_TOPICS, TOPICS, DIFFICULTIES, N_DIFF


# ── Learner archetypes ─────────────────────────────────────────────────────────

class LearnerType(Enum):
    BALANCED    = "balanced"      # average everything
    FAST        = "fast"          # learns quickly, high initial mastery
    SLOW        = "slow"          # learns slowly, needs repetition
    TOPIC_WEAK  = "topic_weak"    # strong overall but weak in specific topics
    INCONSISTENT = "inconsistent" # high variance performance


LEARNER_CONFIGS: Dict[LearnerType, dict] = {
    LearnerType.BALANCED: dict(
        initial_mastery_range=(0.05, 0.20),
        growth_scale=1.0,
        noise_std=0.05,
        decay_rate=0.002,
    ),
    LearnerType.FAST: dict(
        initial_mastery_range=(0.15, 0.35),
        growth_scale=1.8,
        noise_std=0.04,
        decay_rate=0.001,
    ),
    LearnerType.SLOW: dict(
        initial_mastery_range=(0.02, 0.12),
        growth_scale=0.5,
        noise_std=0.06,
        decay_rate=0.004,
    ),
    LearnerType.TOPIC_WEAK: dict(
        initial_mastery_range=(0.10, 0.30),
        growth_scale=1.0,
        noise_std=0.05,
        decay_rate=0.002,
    ),
    LearnerType.INCONSISTENT: dict(
        initial_mastery_range=(0.05, 0.25),
        growth_scale=1.0,
        noise_std=0.20,      # very high noise → inconsistent answers
        decay_rate=0.003,
    ),
}

# Topics that TOPIC_WEAK students are particularly weak in
WEAK_TOPICS_FOR_TYPE = [2, 4, 6]   # Linear Equations, Quadratic, Trig


# ── Student Profile (serializable, used in live sessions) ─────────────────────

@dataclass
class StudentProfile:
    name:         str
    learner_type: LearnerType
    mastery:      List[float]          # [0..1] per topic
    history:      List[dict] = field(default_factory=list)   # step-by-step log
    step:         int        = 0

    # Rolling window for recent performance
    _recent_scores: List[float] = field(default_factory=list, repr=False)

    def record(self, topic_idx: int, diff_idx: int, score: float):
        self.history.append({
            "step":       self.step,
            "topic_idx":  topic_idx,
            "topic":      TOPICS[topic_idx].name,
            "difficulty": DIFFICULTIES[diff_idx],
            "score":      score,
            "mastery_snapshot": list(self.mastery),
        })
        self._recent_scores.append(score)
        if len(self._recent_scores) > 10:
            self._recent_scores.pop(0)
        self.step += 1

    def recent_accuracy(self, window: int = 5) -> float:
        if not self._recent_scores:
            return 0.0
        return float(np.mean(self._recent_scores[-window:]))

    def accuracy_bucket(self, window: int = 5) -> int:
        acc = self.recent_accuracy(window)
        if acc < 0.40: return 0   # low
        if acc < 0.70: return 1   # mid
        return 2                  # high

    def topic_trend(self, topic_idx: int, window: int = 5) -> str:
        """Return 'improving', 'declining', or 'stable' for a topic."""
        scores = [h["score"] for h in self.history if h["topic_idx"] == topic_idx]
        if len(scores) < 3:
            return "insufficient data"
        recent = np.mean(scores[-window:])
        older  = np.mean(scores[:-window] or scores)
        delta  = recent - older
        if delta >  0.10: return "improving"
        if delta < -0.10: return "declining"
        return "stable"

    def weakest_topics(self, n: int = 3) -> List[int]:
        return sorted(range(N_TOPICS), key=lambda i: self.mastery[i])[:n]

    def strongest_topics(self, n: int = 3) -> List[int]:
        return sorted(range(N_TOPICS), key=lambda i: -self.mastery[i])[:n]

    def performance_summary(self) -> dict:
        return {
            "overall_accuracy":  round(self.recent_accuracy(), 3),
            "mastery":           {TOPICS[i].name: round(self.mastery[i], 3) for i in range(N_TOPICS)},
            "weakest":           [TOPICS[i].name for i in self.weakest_topics(3)],
            "strongest":         [TOPICS[i].name for i in self.strongest_topics(3)],
            "total_questions":   self.step,
        }

    def apply_retention_decay(self, decay_rate: float = 0.002):
        """Simulate forgetting — mastery decays slightly each episode."""
        for i in range(N_TOPICS):
            self.mastery[i] = max(0.0, self.mastery[i] - decay_rate)


# ── Simulated Student (used during RL training) ────────────────────────────────

class SimulatedStudent:
    """
    Generates synthetic performance data for RL training.
    Includes learning effects, forgetting, noise, and learner-type behaviour.
    """

    def __init__(self, learner_type: LearnerType = LearnerType.BALANCED, seed: int = 42):
        self.learner_type = learner_type
        self.cfg          = LEARNER_CONFIGS[learner_type]
        self.rng          = np.random.default_rng(seed)

        lo, hi         = self.cfg["initial_mastery_range"]
        self.mastery   = list(self.rng.uniform(lo, hi, N_TOPICS))

        # Topic-weak students have deliberately low mastery in specific topics
        if learner_type == LearnerType.TOPIC_WEAK:
            for idx in WEAK_TOPICS_FOR_TYPE:
                self.mastery[idx] = self.rng.uniform(0.01, 0.08)

        self.profile = StudentProfile(
            name         = f"Simulated_{learner_type.value}",
            learner_type = learner_type,
            mastery      = self.mastery,
        )

    def answer(self, topic_idx: int, diff_idx: int) -> float:
        """
        Simulate answering a question. Returns score in [0, 1].
        Updates mastery with learning + decay effects.
        """
        diff      = DIFFICULTIES[diff_idx]
        mastery   = self.mastery[topic_idx]

        # Base probability: mastery vs difficulty
        base_prob = np.clip(mastery / diff, 0.0, 1.0)

        # Add learner noise
        noise     = self.rng.normal(0, self.cfg["noise_std"])
        prob      = np.clip(base_prob + noise, 0.05, 0.95)
        score     = float(self.rng.random() < prob)

        # Learning effect — grows faster when difficulty ≈ mastery * 3
        zone_match    = 1.0 - abs(mastery * 3 - diff) / 3.0   # [0..1]
        growth        = 0.03 * self.cfg["growth_scale"] * max(0.1, zone_match)
        self.mastery[topic_idx] = min(1.0, mastery + growth)

        # Decay un-practiced topics slightly
        for i in range(N_TOPICS):
            if i != topic_idx:
                self.mastery[i] = max(0.0, self.mastery[i] - self.cfg["decay_rate"])

        self.profile.mastery = list(self.mastery)
        self.profile.record(topic_idx, diff_idx, score)
        return score

    def reset_episode(self):
        """Clear rolling history between episodes (mastery persists)."""
        self.profile._recent_scores.clear()


# ── Factory ────────────────────────────────────────────────────────────────────

def make_student(learner_type: str = "balanced", seed: int = 42) -> SimulatedStudent:
    lt = LearnerType(learner_type.lower())
    return SimulatedStudent(learner_type=lt, seed=seed)


def make_real_student(name: str, learner_type: str = "balanced",
                      initial_mastery: Optional[List[float]] = None) -> StudentProfile:
    """Create a StudentProfile for a real (non-simulated) user."""
    lt = LearnerType(learner_type.lower())
    if initial_mastery is None:
        initial_mastery = [0.0] * N_TOPICS
    return StudentProfile(name=name, learner_type=lt, mastery=list(initial_mastery))


if __name__ == "__main__":
    import json

    for lt in LearnerType:
        s = SimulatedStudent(learner_type=lt, seed=7)
        # Simulate 10 questions
        for _ in range(10):
            s.answer(topic_idx=1, diff_idx=1)
        print(f"\n{lt.value.upper()}")
        print(json.dumps(s.profile.performance_summary(), indent=2))
