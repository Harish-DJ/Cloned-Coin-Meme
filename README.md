# Self-Evolving AI Tutor — Adaptive Learning Agent

> RL (Q-Learning) + LLM Teaching + Explainability + Personalization

---

## Architecture

```
curriculum.py     → Topics, prerequisite DAG, difficulty levels
student.py        → Learner archetypes, mastery tracking, retention decay
reward.py         → Multi-signal reward (improvement + difficulty fit + coverage + milestones)
agent.py          → Q-Learning + explainability engine + confidence scoring
environment.py    → RL training env + inference session (real students)
llm_tutor.py      → Anthropic API: explanations, quiz questions, feedback, study plans
train.py          → Training loop, before/after comparison, plots
demo.py           → Interactive CLI: student enters scores → agent responds
```

---

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional — works without LLM too

# Train + evaluate
python train.py --episodes 800 --compare

# Interactive demo (the product)
python demo.py --name "Alice" --learner balanced --rounds 8

# Demo without LLM (no API key needed)
python demo.py --no-llm

# Fast learner vs slow learner comparison
python demo.py --compare

# Train on all learner types
python train.py --all --episodes 600
```

---

## What each red flag fixes

| Problem | Fix |
|---|---|
| No real user interaction | `demo.py` — full input/output CLI loop |
| No explainability | `agent.explain_action()` — strategy detection + narration |
| No personalization | 5 learner archetypes in `student.py` |
| Reward too simple | 6-component reward in `reward.py` |
| No curriculum structure | Prerequisite DAG in `curriculum.py`, enforced in `agent.select_action()` |
| No teaching layer | `llm_tutor.py` — explanations, quiz generation, feedback |
| No before/after | `before_vs_after()` in `environment.py` + plot in `train.py` |

---

## Learner Types

| Type | Behaviour |
|---|---|
| `balanced` | Average learner, steady growth |
| `fast` | High initial mastery, learns quickly |
| `slow` | Low initial mastery, needs repetition |
| `topic_weak` | Strong overall but weak in specific topics |
| `inconsistent` | High performance variance |

---

## Reward Components

| Signal | Weight | Prevents |
|---|---|---|
| Score delta | 1.00 | Ignoring actual improvement |
| Difficulty fit (ZPD) | 0.40 | Staying in easy topics forever |
| Curriculum adherence | 0.30 | Skipping prerequisites |
| Topic coverage | 0.20 | Drilling one topic endlessly |
| Mastery milestone | 0.50 | Missing breakthrough moments |
| Jump penalty | −0.25 | Erratic topic switching |
