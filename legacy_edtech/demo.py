"""
demo.py
───────
Interactive CLI demo — the "student enters performance → agent responds" loop.

This is the missing piece that turns a lab experiment into a product.

Flow:
  1. Choose your learner profile (or enter custom mastery)
  2. Agent recommends first topic + difficulty + explanation
  3. You answer a generated quiz question (or enter your score)
  4. Agent sees your result → explains its next recommendation
  5. Repeat for N rounds
  6. End with personalised study plan

Run:
    python demo.py
    python demo.py --name "Alice" --learner fast --rounds 8
    python demo.py --compare            # show two learner types side by side
"""

import argparse
import time
import sys
import os

# ── Try to import optional deps gracefully ─────────────────────────────────────
try:
    from train import quick_train
    from student import make_real_student, LearnerType, LEARNER_CONFIGS
    from environment import InferenceSession, before_vs_after
    from curriculum import TOPICS, topic_name, diff_label
    from agent import idx_to_action
    from llm_tutor import (
        generate_explanation, generate_question,
        generate_feedback, generate_study_plan,
        narrate_agent_decision, _LLM_AVAILABLE,
    )
    from reward import explain_reward
except ModuleNotFoundError as e:
    print(f"[ERROR] Missing module: {e}")
    print("Make sure you're running from the adaptive_tutor/ directory.")
    sys.exit(1)


# ── ANSI colours (graceful fallback on Windows) ────────────────────────────────
def _c(code: str, text: str) -> str:
    if sys.platform == "win32" or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD   = lambda t: _c("1",    t)
GREEN  = lambda t: _c("32",   t)
YELLOW = lambda t: _c("33",   t)
CYAN   = lambda t: _c("36",   t)
RED    = lambda t: _c("31",   t)
DIM    = lambda t: _c("2",    t)
BLUE   = lambda t: _c("34",   t)


# ── Display helpers ────────────────────────────────────────────────────────────

def hr(char="─", width=65):
    print(DIM(char * width))

def print_header():
    os.system("cls" if sys.platform == "win32" else "clear")
    print(BOLD(CYAN("╔══════════════════════════════════════════════════════════════╗")))
    print(BOLD(CYAN("║         SELF-EVOLVING AI TUTOR  ·  Adaptive Learning          ║")))
    print(BOLD(CYAN("║         RL Agent (Q-Learning) + LLM Teaching Layer           ║")))
    print(BOLD(CYAN("╚══════════════════════════════════════════════════════════════╝")))
    print()

def print_mastery_bar(mastery: list, highlight_idx: int = -1):
    print(BOLD("\n  Topic Mastery:"))
    for i, topic in enumerate(TOPICS):
        pct   = mastery[i]
        filled = int(pct * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        pct_s  = f"{pct:.0%}"
        label  = f"  {topic.name:<25}"
        col    = GREEN if pct >= 0.8 else (YELLOW if pct >= 0.5 else RED)
        marker = " ◀ CURRENT" if i == highlight_idx else ""
        print(f"  {label} {col(bar)} {pct_s}{BOLD(marker)}")
    print()

def print_recommendation(rec: dict, step: int):
    r = rec["recommendation"]
    hr("═")
    print(BOLD(f"\n  📌 Step {step} — Agent Recommendation"))
    hr()
    print(f"  Topic      : {BOLD(CYAN(r['chosen_topic']))}")
    print(f"  Difficulty : {BOLD(r['chosen_difficulty'])}")
    print(f"  Strategy   : {YELLOW(r['strategy'].replace('_', ' ').title())}")
    _conf = f"{r['confidence']:.0%}"
    print(f"  Confidence : {GREEN(_conf)}")
    hr()
    print(f"\n  {BOLD('Why this recommendation:')}")
    print(f"  {r['reason_long']}")
    print()
    print(f"  {BOLD('Top alternatives considered:')}")
    for alt in r["top_alternatives"]:
        marker = GREEN("▶ CHOSEN") if alt["chosen"] else DIM("  ")
        print(f"    {marker}  {alt['topic']:<25} {alt['difficulty']:<8} Q={alt['q_value']:+.4f}")
    print()

def print_quiz_question(q: dict):
    hr()
    print(BOLD("\n  📝 Quiz Question:"))
    print(f"\n  {q['question']}\n")
    for i, opt in enumerate(q["options"]):
        label = chr(65 + i)   # A, B, C, D
        print(f"    [{label}]  {opt}")
    print()

def print_performance(perf: dict):
    print(BOLD("\n  📊 Your Performance:"))
    _acc = f"{perf['overall_accuracy']:.0%}"
    print(f"  Overall accuracy : {GREEN(_acc)}")
    print(f"  Questions done   : {perf['total_questions']}")
    print(f"  Weakest topics   : {RED(', '.join(perf['weakest'][:2]))}")
    print(f"  Strongest topics : {GREEN(', '.join(perf['strongest'][:2]))}")
    print()


# ── Core demo loop ─────────────────────────────────────────────────────────────

def run_demo(
    student_name:  str         = "Student",
    learner_type:  str         = "balanced",
    n_rounds:      int         = 6,
    use_llm:       bool        = True,
    quick:         bool        = False,
):
    print_header()
    print(f"  Hello, {BOLD(student_name)}! Preparing your personalised AI tutor…\n")
    print(f"  Learner profile : {YELLOW(learner_type.upper())}")
    print(f"  LLM teaching    : {'✅ ON' if (use_llm and _LLM_AVAILABLE) else '⚠️  OFF (no API key)'}")
    print()

    # Train agent
    n_eps = 200 if quick else 600
    print(f"  {DIM('Training RL agent...')}", end="", flush=True)
    agent = quick_train(n_episodes=n_eps)
    print(GREEN("  ✓ Ready\n"))

    # Create student profile
    lt      = LearnerType(learner_type)
    profile = make_real_student(student_name, learner_type)
    session = InferenceSession(agent, profile)

    # First recommendation
    rec         = session.get_first_recommendation()
    curr_topic  = rec["next_topic_idx"]
    curr_diff   = rec["next_diff_idx"]
    correct_streak = 0

    for step in range(1, n_rounds + 1):

        # Show mastery bars
        print_mastery_bar(profile.mastery, highlight_idx=curr_topic)

        # Show recommendation
        print_recommendation(rec, step)

        # LLM: explain the topic
        if use_llm and _LLM_AVAILABLE:
            print(BOLD("  📖 Topic Explanation:"))
            explanation = generate_explanation(
                topic_idx=curr_topic, diff_idx=curr_diff,
                mastery=profile.mastery[curr_topic], student_name=student_name
            )
            print(f"\n  {explanation}\n")

        # LLM: generate quiz question
        if use_llm and _LLM_AVAILABLE:
            q = generate_question(curr_topic, curr_diff)
            print_quiz_question(q)
            user_input = input(BOLD("  Your answer [A/B/C/D] (or Enter to skip): ")).strip().upper()

            if user_input in "ABCD" and user_input:
                chosen_idx = ord(user_input) - 65
                is_correct = (chosen_idx == q["correct_index"])
                score      = 1.0 if is_correct else 0.0
                correct_streak = (correct_streak + 1) if is_correct else 0

                result_str = GREEN("✓ Correct!") if is_correct else RED("✗ Incorrect")
                print(f"\n  {result_str}")
                if not is_correct:
                    correct_ans = q["options"][q["correct_index"]]
                    print(f"  Correct answer: {GREEN(correct_ans)}")
                print(f"  {DIM(q['explanation'])}")

                if use_llm:
                    fb = generate_feedback(
                        curr_topic, curr_diff, is_correct,
                        profile.mastery[curr_topic], correct_streak, student_name
                    )
                    print(f"\n  💬 {CYAN(fb)}")
            else:
                # User skipped — ask for manual score
                print(f"\n  {DIM('Question skipped.')}")
                raw = input(BOLD("  Enter your score for this topic [0–100%]: ")).strip()
                try:
                    score = float(raw.replace("%", "")) / 100.0
                except ValueError:
                    score = 0.5
                is_correct = score >= 0.5
        else:
            # No LLM: just ask for a score
            print(f"  Topic: {BOLD(topic_name(curr_topic))}  Difficulty: {BOLD(diff_label(curr_diff))}")
            raw = input(BOLD("  Enter your score for this topic [0–100]: ")).strip()
            try:
                score = float(raw) / 100.0
            except ValueError:
                score = 0.5

        print()
        hr()

        # Submit to session → get next recommendation
        rec        = session.submit_score(curr_topic, curr_diff, score)
        curr_topic = rec["next_topic_idx"]
        curr_diff  = rec["next_diff_idx"]

        print_performance(rec["performance"])
        input(DIM("  Press Enter for next step…"))
        print("\n" * 2)

    # ── Final summary ──────────────────────────────────────────────────────────
    hr("═")
    print(BOLD(CYAN("\n  🎓 SESSION COMPLETE\n")))
    print_mastery_bar(profile.mastery)

    if use_llm and _LLM_AVAILABLE:
        print(BOLD("  📅 Personalised Study Plan:\n"))
        plan = generate_study_plan(profile.performance_summary(), student_name, n_weeks=2)
        for line in plan.split("\n"):
            print(f"  {line}")
    else:
        perf = profile.performance_summary()
        print(BOLD("  📊 Final Performance Summary:"))
        _acc = f"{perf['overall_accuracy']:.0%}"
        print(f"  Accuracy  : {GREEN(_acc)}")
        print(f"  Weakest   : {RED(', '.join(perf['weakest']))}")
        print(f"  Strongest : {GREEN(', '.join(perf['strongest']))}")

    hr("═")
    print()


# ── Side-by-side comparison (WOW factor) ──────────────────────────────────────

def run_comparison(n_episodes: int = 400):
    """Show the trained agent teaching two different learner types simultaneously."""
    print_header()
    print(BOLD("  🔬 COMPARISON MODE — Fast Learner vs Slow Learner\n"))

    agent = quick_train(n_episodes=n_episodes)

    learner_types = ["fast", "slow"]
    profiles = {
        lt: make_real_student(lt.capitalize(), lt)
        for lt in learner_types
    }
    sessions = {
        lt: InferenceSession(agent, profiles[lt])
        for lt in learner_types
    }
    recs = {lt: sessions[lt].get_first_recommendation() for lt in learner_types}

    for step in range(1, 7):
        hr("═")
        print(BOLD(f"\n  STEP {step}\n"))

        for lt in learner_types:
            r     = recs[lt]["recommendation"]
            score = 0.8 if lt == "fast" else 0.4    # simulate answers
            print(f"  {BOLD(lt.upper()):20}  →  {CYAN(r['chosen_topic']):<25} "
                  f"[{r['chosen_difficulty']}]  strategy={YELLOW(r['strategy'])}")

            recs[lt] = sessions[lt].submit_score(
                recs[lt]["next_topic_idx"], recs[lt]["next_diff_idx"], score
            )

    print(BOLD("\n\n  FINAL MASTERY COMPARISON\n"))
    topics = [t.name for t in TOPICS]
    for topic in topics:
        fast_m = profiles["fast"].mastery[TOPICS.index(next(t for t in TOPICS if t.name == topic))]
        slow_m = profiles["slow"].mastery[TOPICS.index(next(t for t in TOPICS if t.name == topic))]
        bar_f  = "█" * int(fast_m * 15)
        bar_s  = "█" * int(slow_m * 15)
        print(f"  {topic:<25}  FAST:{GREEN(bar_f):<20}{fast_m:.0%}  SLOW:{YELLOW(bar_s):<20}{slow_m:.0%}")

    hr("═")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Learning AI Tutor — Demo")
    parser.add_argument("--name",    type=str, default="Student",  help="Your name")
    parser.add_argument("--learner", type=str, default="balanced",
                        choices=["balanced", "fast", "slow", "topic_weak", "inconsistent"],
                        help="Learner archetype")
    parser.add_argument("--rounds",  type=int, default=6,  help="Number of Q&A rounds")
    parser.add_argument("--no-llm",  action="store_true",  help="Disable LLM (no API key needed)")
    parser.add_argument("--quick",   action="store_true",  help="Fast training (fewer episodes)")
    parser.add_argument("--compare", action="store_true",  help="Run fast vs slow learner comparison")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        run_demo(
            student_name = args.name,
            learner_type = args.learner,
            n_rounds     = args.rounds,
            use_llm      = not args.no_llm,
            quick        = args.quick,
        )
