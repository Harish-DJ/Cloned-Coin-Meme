"""
curriculum.py
─────────────
Defines the subject curriculum: topics, difficulty levels,
prerequisite graph, and mastery thresholds.

Design choices:
  • Topics are nodes in a DAG — agent cannot assign a topic unless all
    prerequisites are sufficiently mastered (>= PREREQ_THRESHOLD).
  • Each topic has 3 difficulty levels: easy / medium / hard.
  • `get_available_actions` enforces curriculum intelligence so the agent
    can never make educationally nonsensical jumps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

# ── Mastery threshold to unlock a topic's dependents ──────────────────────────
PREREQ_THRESHOLD = 0.50   # student must have ≥50% mastery before moving forward


@dataclass
class Topic:
    idx:           int
    name:          str
    prerequisites: List[int]          # indices of required topics
    description:   str = ""
    tags:          List[str] = field(default_factory=list)


# ── Curriculum definition ──────────────────────────────────────────────────────
#    Structured as a prerequisite DAG (Directed Acyclic Graph)
#
#    Number Sense
#         │
#    Algebra Basics
#         │
#    Linear Equations ──────┐
#         │                 │
#    Quadratic Equations  Systems of Equations
#         │
#    Polynomials
#         │
#    Trigonometry
#
TOPICS: List[Topic] = [
    Topic(0, "Number Sense",          prerequisites=[],    description="Integers, fractions, decimals, order of operations",   tags=["foundational"]),
    Topic(1, "Algebra Basics",        prerequisites=[0],   description="Variables, expressions, simplification",               tags=["algebra"]),
    Topic(2, "Linear Equations",      prerequisites=[1],   description="Solving one and two-variable linear equations",        tags=["algebra"]),
    Topic(3, "Systems of Equations",  prerequisites=[2],   description="Substitution and elimination methods",                 tags=["algebra"]),
    Topic(4, "Quadratic Equations",   prerequisites=[2],   description="Factoring, completing the square, quadratic formula",  tags=["algebra"]),
    Topic(5, "Polynomials",           prerequisites=[4],   description="Polynomial arithmetic, division, factoring",           tags=["algebra"]),
    Topic(6, "Trigonometry",          prerequisites=[4],   description="Sin, cos, tan, unit circle, identities",               tags=["geometry"]),
    Topic(7, "Statistics Basics",     prerequisites=[0],   description="Mean, median, mode, standard deviation",               tags=["statistics"]),
    Topic(8, "Probability",           prerequisites=[7],   description="Events, conditional probability, Bayes theorem",       tags=["statistics"]),
    Topic(9, "Calculus Intro",        prerequisites=[5, 6],description="Limits, derivatives, basic integration",               tags=["calculus"]),
]

N_TOPICS = len(TOPICS)

DIFFICULTIES: List[int] = [1, 2, 3]     # 1=easy, 2=medium, 3=hard
N_DIFF = len(DIFFICULTIES)

DIFF_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}


# ── Prerequisite helpers ───────────────────────────────────────────────────────

def prerequisites_met(topic_idx: int, mastery: List[float]) -> bool:
    """Return True if all prerequisites for topic_idx are sufficiently mastered."""
    for pre in TOPICS[topic_idx].prerequisites:
        if mastery[pre] < PREREQ_THRESHOLD:
            return False
    return True


def get_available_topics(mastery: List[float]) -> List[int]:
    """Return topic indices the student is currently eligible to study."""
    return [i for i in range(N_TOPICS) if prerequisites_met(i, mastery)]


def get_available_actions(mastery: List[float]) -> List[Tuple[int, int]]:
    """
    Return (topic_idx, diff_idx) pairs that are educationally valid
    given current mastery. Also filters out hard difficulty unless
    mastery in that topic is already > 0.30.
    """
    actions = []
    for topic_idx in get_available_topics(mastery):
        for diff_idx, diff in enumerate(DIFFICULTIES):
            # Don't offer hard difficulty on a topic the student barely knows
            if diff == 3 and mastery[topic_idx] < 0.30:
                continue
            actions.append((topic_idx, diff_idx))
    return actions if actions else [(0, 0)]   # fallback: always have at least one


def action_to_idx(topic_idx: int, diff_idx: int) -> int:
    return topic_idx * N_DIFF + diff_idx


def idx_to_action(action_idx: int) -> Tuple[int, int]:
    return divmod(action_idx, N_DIFF)


N_ACTIONS = N_TOPICS * N_DIFF


def topic_name(topic_idx: int) -> str:
    return TOPICS[topic_idx].name


def diff_label(diff_idx: int) -> str:
    return DIFF_LABELS[DIFFICULTIES[diff_idx]]


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_curriculum():
    print("\n── Curriculum ──────────────────────────────────────────────")
    for t in TOPICS:
        prereqs = ", ".join(TOPICS[p].name for p in t.prerequisites) or "None"
        print(f"  [{t.idx}] {t.name:<25} prerequisites: {prereqs}")
    print()


if __name__ == "__main__":
    print_curriculum()
    dummy_mastery = [0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0]
    available = get_available_topics(dummy_mastery)
    print("Available topics with partial mastery:", [TOPICS[i].name for i in available])
