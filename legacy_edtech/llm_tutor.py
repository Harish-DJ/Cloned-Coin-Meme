"""
llm_tutor.py
────────────
LLM-powered teaching layer using the Anthropic API.

Responsibilities:
  1. generate_explanation()  — explain a topic at the right difficulty
  2. generate_question()     — create a quiz question for (topic, difficulty)
  3. evaluate_answer()       — score a free-text student answer
  4. generate_feedback()     — personalised feedback after answering
  5. generate_study_plan()   — full study plan from performance summary

The RL agent decides WHAT to teach.
The LLM decides HOW to teach it.

Setup:
    export ANTHROPIC_API_KEY=your_key_here
    pip install anthropic
"""

import os
import json
from typing import Optional
from static_content import get_static_explanation, get_static_question

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

from curriculum import TOPICS, DIFFICULTIES, diff_label, topic_name

MODEL = "claude-opus-4-5"

# ── Difficulty → plain English ─────────────────────────────────────────────────
DIFF_DESCRIPTIONS = {
    1: "beginner level, use simple language and basic examples",
    2: "intermediate level, include worked examples and some formulas",
    3: "advanced level, include edge cases and challenge the student",
}


def _call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    """Central LLM call with graceful fallback if API unavailable."""
    if not _LLM_AVAILABLE:
        return "[LLM unavailable — install 'anthropic' and set ANTHROPIC_API_KEY]"
    try:
        msg = _CLIENT.messages.create(
            model      = MODEL,
            max_tokens = max_tokens,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ── 1. Topic Explanation ───────────────────────────────────────────────────────

def generate_explanation(
    topic_idx:  int,
    diff_idx:   int,
    mastery:    float,
    student_name: str = "Student",
) -> str:
    """
    Returns a beautifully formatted explanation from the static content file.
    """
    return get_static_explanation(topic_idx, diff_idx, student_name)


# ── 2. Question Generation ─────────────────────────────────────────────────────

def generate_question(topic_idx: int, diff_idx: int) -> dict:
    """
    Returns a high-quality quiz question from the static content file.
    """
    return get_static_question(topic_idx, diff_idx)


# ── 3. Answer Evaluation ───────────────────────────────────────────────────────

def evaluate_free_text_answer(
    question:        str,
    student_answer:  str,
    correct_answer:  str,
    topic_idx:       int,
) -> dict:
    """
    Evaluate a free-text student answer. Returns {score, feedback, is_correct}.
    """
    system = (
        "You are a strict but kind math tutor grading student answers. "
        "Respond with JSON only: {\"is_correct\": bool, \"score\": 0.0-1.0, \"feedback\": \"...\"}"
    )
    user = (
        f"Topic: {TOPICS[topic_idx].name}\n"
        f"Question: {question}\n"
        f"Expected answer: {correct_answer}\n"
        f"Student answer: {student_answer}\n\n"
        f"Grade the answer. Award partial credit if reasoning is correct but arithmetic slipped."
    )
    raw = _call_llm(system, user, max_tokens=250)
    try:
        return json.loads(raw)
    except Exception:
        is_correct = student_answer.strip().lower() == correct_answer.strip().lower()
        return {
            "is_correct": is_correct,
            "score":      1.0 if is_correct else 0.0,
            "feedback":   "Correct!" if is_correct else f"The answer was: {correct_answer}",
        }


# ── 4. Personalised Feedback ───────────────────────────────────────────────────

def generate_feedback(
    topic_idx:     int,
    diff_idx:      int,
    is_correct:    bool,
    mastery:       float,
    streak:        int,           # consecutive correct answers
    student_name:  str = "Student",
) -> str:
    """
    Returns encouraging static feedback.
    """
    if is_correct:
        if streak >= 3:
            return f"Excellent work, {student_name}! That's {streak} in a row. You're really getting the hang of {TOPICS[topic_idx].name}."
        return f"Spot on, {student_name}! Great job."
    else:
        if mastery < 0.3:
            return f"Don't worry {student_name}, {TOPICS[topic_idx].name} takes time. Let's look at the solution together."
        return f"Not quite, but you're close! Review the solution below to see where you went wrong."


# ── 5. Study Plan Generation ───────────────────────────────────────────────────

def generate_study_plan(
    performance_summary: dict,
    student_name: str = "Student",
    n_weeks: int = 2,
) -> str:
    """
    Generate a personalised study plan based on the student's performance.
    """
    system = (
        "You are an expert academic advisor. "
        "Create structured, actionable study plans. Use bullet points. Be specific."
    )
    user = (
        f"Create a {n_weeks}-week study plan for {student_name} based on:\n"
        f"{json.dumps(performance_summary, indent=2)}\n\n"
        f"Focus on weak topics first. Mention specific practice strategies. "
        f"Keep it motivating and realistic."
    )
    return _call_llm(system, user, max_tokens=500)


# ── 6. Decision Narration (bridges RL → human language) ───────────────────────

def narrate_agent_decision(
    strategy:        str,
    topic_name_str:  str,
    diff_str:        str,
    mastery:         float,
    recent_accuracy: float,
) -> str:
    """
    Translate the RL agent's decision into natural language for the student.
    More conversational than explain_action() — this is student-facing.
    """
    system = (
        "You are a friendly AI tutor narrating a study recommendation. "
        "Be conversational, warm, and motivating. One paragraph, under 80 words."
    )
    user = (
        f"The AI agent decided to assign: {topic_name_str} at {diff_str} difficulty.\n"
        f"Strategy: {strategy}\n"
        f"Student mastery: {mastery:.0%}, recent accuracy: {recent_accuracy:.0%}.\n"
        f"Narrate this recommendation in a friendly, student-facing way — "
        f"explain WHY this was chosen without sounding robotic."
    )
    return _call_llm(system, user, max_tokens=150)


if __name__ == "__main__":
    print("=== LLM Tutor Demo ===\n")
    print("Generating explanation for Linear Equations (Easy)…")
    expl = generate_explanation(topic_idx=2, diff_idx=0, mastery=0.3)
    print(expl)

    print("\nGenerating quiz question…")
    q = generate_question(topic_idx=2, diff_idx=1)
    print(f"Q: {q['question']}")
    for i, opt in enumerate(q['options']):
        marker = "✓" if i == q['correct_index'] else " "
        print(f"  [{marker}] {opt}")
    print(f"Explanation: {q['explanation']}")
