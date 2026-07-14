"""Offline LLM-as-a-judge evaluation of the app's own prompts and models.

While `evaluation.py` judges the *candidate's* answers at runtime (a product
feature), this script judges the *application's* outputs at development time:
it generates interviewer replies with different prompt techniques and/or
models on a fixed set of test cases, scores each reply with an independent
judge LLM, and prints a comparison table. The results justify which prompt
technique and model the app ships with.

Usage:
    python evals.py                                      # all 5 techniques, default model
    python evals.py --techniques Zero-Shot Few-Shot      # subset of techniques
    python evals.py --models openai/gpt-5-mini openai/gpt-5-nano
    python evals.py --judge-model openai/gpt-5           # override the judge
    python evals.py --pairwise                           # A/B compare techniques

Notes on judge reliability:
- The judge is a separate call with its own prompt and temperature 0.2.
- By default the judge model differs from the generation model to reduce
  self-preference bias (models tend to over-score their own style).
- Rationales are requested for every score so results can be spot-checked.
"""

import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from evaluation import _call_json_completion, format_transcript
from pricing import estimate_cost, format_cost
from prompts import PROMPT_TECHNIQUES, build_system_prompt

DEFAULT_GENERATION_MODEL = "openai/gpt-5-mini"
# Judge from a different model family than the GPT-5 chat models to reduce
# self-preference bias (a judge tends to over-score its own family's style).
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"
GENERATION_TEMPERATURE = 0.7  # matches the app's chat default
GENERATION_MAX_TOKENS = 8192  # GPT-5 models spend part of this on hidden reasoning

CRITERIA = ["relevance", "realism", "specificity", "helpfulness"]

# Fixed test cases so runs are reproducible and comparable. Each case is a
# conversation snippet ending with a candidate message; the app under test
# must produce the interviewer's next reply.
TEST_CASES = [
    {
        "name": "opening-question",
        "target_role": "Data Analyst",
        "interview_type": "General Mock Interview",
        "job_description": (
            "We are looking for a Data Analyst with strong SQL, dbt and Looker "
            "experience to own reporting for the growth team. You will design "
            "A/B test readouts and present insights to non-technical stakeholders."
        ),
        "messages": [
            {"role": "user", "content": "Ask me the first interview question."},
        ],
    },
    {
        "name": "vague-behavioral-answer",
        "target_role": "Product Manager",
        "interview_type": "Behavioral (STAR)",
        "job_description": (
            "Product Manager for a B2B SaaS platform. Owns roadmap "
            "prioritization, works with engineering and design, talks to "
            "customers weekly, and is accountable for activation metrics."
        ),
        "messages": [
            {
                "role": "assistant",
                "content": "Tell me about a time you had to say no to an important stakeholder.",
            },
            {
                "role": "user",
                "content": (
                    "Well, that happens a lot in my job. I usually just explain that "
                    "we can't do everything and people mostly understand. I'm pretty "
                    "good at communication so it works out fine."
                ),
            },
        ],
    },
    {
        "name": "strong-technical-answer",
        "target_role": "Backend Engineer",
        "interview_type": "Technical Q&A",
        "job_description": (
            "Backend Engineer (Python) building a high-throughput payments API. "
            "PostgreSQL, Redis, Kafka; strong focus on reliability and idempotency."
        ),
        "messages": [
            {
                "role": "assistant",
                "content": "How would you make a payment-creation endpoint safe to retry?",
            },
            {
                "role": "user",
                "content": (
                    "I'd require an idempotency key from the client, store it with the "
                    "request hash and response in Postgres with a unique constraint, and "
                    "return the stored response on repeats. Writes go in one transaction; "
                    "on conflict we read the existing row. Keys expire after ~24h, and I'd "
                    "return 409 if the same key arrives with a different payload."
                ),
            },
        ],
    },
    {
        "name": "subtly-wrong-answer",
        "target_role": "Data Engineer",
        "interview_type": "Technical Q&A",
        "job_description": (
            "Data Engineer maintaining Airflow pipelines and a Snowflake warehouse; "
            "cares about data quality, backfills, and cost control."
        ),
        "messages": [
            {
                "role": "assistant",
                "content": "What's the difference between incremental and full-refresh loads, and when do you pick each?",
            },
            {
                "role": "user",
                "content": (
                    "Incremental loads only new rows so it's always faster and cheaper, "
                    "and full refresh is basically legacy — you never really need it if "
                    "your incremental logic is correct, since late-arriving data will be "
                    "picked up in the next run anyway."
                ),
            },
        ],
    },
    {
        "name": "meta-question",
        "target_role": "UX Designer",
        "interview_type": "Questions to Ask the Interviewer",
        "job_description": (
            "Senior UX Designer at a fintech scale-up; design system ownership, "
            "user research, close collaboration with product and compliance."
        ),
        "messages": [
            {
                "role": "user",
                "content": "What should I ask the interviewer at the end of my onsite?",
            },
        ],
    },
]

# The judge sees the conversation and the generated reply, never the technique
# or model name — scores can't leak from labels.
EVAL_JUDGE_SYSTEM_PROMPT = """You are an impartial judge assessing the quality of an \
AI INTERVIEWER'S reply in a mock-interview practice app. The candidate is preparing \
for the position of {target_role}. You are evaluating the app's output — NOT the \
candidate's answer.

Score the interviewer's reply on four criteria, each 1-5:
- relevance: fits the role, the job posting, and this exact point in the conversation
- realism: sounds like a competent human interviewer or coach; appropriate difficulty; \
catches mistakes or vagueness in the candidate's answer instead of praising it blindly
- specificity: engages with what the candidate actually said; no generic filler
- helpfulness: moves the candidate forward — a clear question, concrete feedback, or \
actionable advice

Calibrate every score against these anchors:
- 1-2: fails the criterion
- 3: adequate — does the job, nothing more
- 4: good — a concrete, noteworthy strength beyond adequate
- 5: exceptional — reserve for a reply you could not realistically improve; rare

Most competent replies should score 3-4. Before scoring, identify at least one \
concrete way the reply could be better; if you truly cannot, only then award a 5.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "rubric": {{
    "relevance":   {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "realism":     {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "specificity": {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "helpfulness": {{"score": <integer 1-5>, "rationale": "<one sentence>"}}
  }},
  "average_score": <float>,
  "verdict": "<one-sentence overall judgment of the reply>"
}}

Be strict and consistent. A reply that ignores an error in the candidate's answer, or \
that could have been written without reading the conversation, must not score above 3 \
on realism or specificity."""


# Pairwise mode: absolute 1-5 scores saturate near the top when every
# technique is decent (see README), so this judge is forced to CHOOSE between
# two replies instead of scoring them independently — far more discriminative.
PAIRWISE_JUDGE_SYSTEM_PROMPT = """You are an impartial judge comparing two AI INTERVIEWER \
replies in a mock-interview practice app. The candidate is preparing for the position of \
{target_role}. Both replies respond to the same conversation.

Decide which reply is the better next turn from an interviewer/coach, considering: \
fit to the role, the job posting, and this exact point in the conversation; realism; \
engagement with what the candidate actually said (including catching mistakes or \
vagueness); and how much it moves the candidate forward.

You MUST prefer one reply unless they are truly indistinguishable in quality — \
"tie" is a last resort, not a safe default. Do not reward length for its own sake.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "winner": "A" | "B" | "tie",
  "rationale": "<1-2 sentences naming the deciding difference>"
}}"""


def get_openrouter_client() -> OpenAI:
    """Standalone OpenRouter client (no Streamlit; API key from the environment)."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (add it to .env).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def build_eval_system_prompt(technique: str, case: Dict) -> str:
    """The exact prompt the app ships — via the shared builder in prompts.py."""
    return build_system_prompt(
        technique=technique,
        interview_type=case["interview_type"],
        target_role=case["target_role"],
        job_description=case["job_description"],
    )


def generate_reply(
    client: OpenAI,
    model: str,
    technique: str,
    case: Dict,
) -> Tuple[Optional[str], Optional[object]]:
    """Produce the interviewer's next reply for a test case under one variant.

    Retries a couple of times on empty content: OpenRouter occasionally returns
    finish_reason="error" from the upstream provider on an otherwise fine request.
    """
    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": build_eval_system_prompt(technique, case)}]
            + case["messages"],
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return reply, response.usage
    return None, response.usage


def judge_reply(
    client: OpenAI,
    judge_model: str,
    case: Dict,
    reply: str,
) -> Tuple[Optional[dict], Optional[object]]:
    """Score one generated interviewer reply with the independent judge.

    Retries on unparseable output — usually a transient empty response from
    the upstream provider rather than genuinely malformed JSON.
    """
    system_prompt = EVAL_JUDGE_SYSTEM_PROMPT.format(target_role=case["target_role"])
    user_content = (
        f"JOB POSTING:\n---\n{case['job_description']}\n---\n\n"
        f"CONVERSATION SO FAR:\n---\n{format_transcript(case['messages'])}\n---\n\n"
        f"INTERVIEWER'S REPLY TO EVALUATE:\n---\n{reply}\n---"
    )
    for _ in range(3):
        judgment, usage = _call_json_completion(client, judge_model, system_prompt, user_content)
        if judgment is not None and "rubric" in judgment:
            return judgment, usage
    return judgment, usage


def usage_cost(model: str, usage: Optional[object]) -> float:
    """Estimated USD cost of one call; 0.0 when pricing or usage is unavailable."""
    if usage is None:
        return 0.0
    cost = estimate_cost(
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
    return cost or 0.0


def run_evals(
    client: OpenAI,
    techniques: List[str],
    models: List[str],
    judge_model: str,
) -> Tuple[List[Dict], float]:
    """Evaluate every (technique, model) variant on every test case."""
    results = []
    total_cost = 0.0
    variants = [(t, m) for t in techniques for m in models]

    for technique, model in variants:
        print(f"\n=== {technique} / {model} ===")
        for case in TEST_CASES:
            reply, gen_usage = generate_reply(client, model, technique, case)
            total_cost += usage_cost(model, gen_usage)
            if reply is None:
                print(f"  {case['name']}: EMPTY REPLY — skipped")
                continue

            judgment, judge_usage = judge_reply(client, judge_model, case, reply)
            total_cost += usage_cost(judge_model, judge_usage)
            if judgment is None or "rubric" not in judgment:
                print(f"  {case['name']}: judge returned unparseable output — skipped")
                continue

            scores = {
                criterion: judgment["rubric"].get(criterion, {}).get("score")
                for criterion in CRITERIA
            }
            results.append(
                {
                    "technique": technique,
                    "model": model,
                    "case": case["name"],
                    "scores": scores,
                    "average": judgment.get("average_score"),
                    "verdict": judgment.get("verdict", ""),
                    "reply": reply,
                    "rubric": judgment["rubric"],
                }
            )
            score_line = ", ".join(f"{c}={scores[c]}" for c in CRITERIA)
            print(f"  {case['name']}: {score_line} → avg {judgment.get('average_score')}")

    return results, total_cost


def summarize(results: List[Dict]) -> None:
    """Print a per-variant summary table (mean score per criterion + overall)."""
    variants = sorted({(r["technique"], r["model"]) for r in results})
    if not variants:
        print("\nNo successful evaluations to summarize.")
        return

    header = ["technique", "model"] + CRITERIA + ["overall", "cases"]
    rows = []
    for technique, model in variants:
        rs = [r for r in results if (r["technique"], r["model"]) == (technique, model)]
        means = []
        for criterion in CRITERIA:
            values = [r["scores"][criterion] for r in rs if r["scores"][criterion] is not None]
            means.append(sum(values) / len(values) if values else float("nan"))
        overall = sum(means) / len(means)
        rows.append(
            [technique, model]
            + [f"{m:.2f}" for m in means]
            + [f"{overall:.2f}", str(len(rs))]
        )

    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    print("\n" + " | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-|-".join("-" * w for w in widths))
    for row in sorted(rows, key=lambda r: r[-2], reverse=True):
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def judge_pair(
    client: OpenAI,
    judge_model: str,
    case: Dict,
    reply_a: str,
    reply_b: str,
) -> Tuple[Optional[dict], Optional[object]]:
    """Ask the judge which of two replies is the better next interviewer turn."""
    system_prompt = PAIRWISE_JUDGE_SYSTEM_PROMPT.format(target_role=case["target_role"])
    user_content = (
        f"JOB POSTING:\n---\n{case['job_description']}\n---\n\n"
        f"CONVERSATION SO FAR:\n---\n{format_transcript(case['messages'])}\n---\n\n"
        f"REPLY A:\n---\n{reply_a}\n---\n\n"
        f"REPLY B:\n---\n{reply_b}\n---"
    )
    judgment, usage = None, None
    for _ in range(3):
        judgment, usage = _call_json_completion(client, judge_model, system_prompt, user_content)
        if judgment is not None and judgment.get("winner") in {"A", "B", "tie"}:
            return judgment, usage
    return judgment, usage


def run_pairwise(
    client: OpenAI,
    techniques: List[str],
    model: str,
    judge_model: str,
) -> Tuple[List[Dict], float]:
    """Pairwise A/B comparison of techniques on one model.

    Each pair of techniques is compared on every test case. Presentation
    order (A vs B) is randomized per comparison to cancel position bias.
    """
    total_cost = 0.0

    # One generation per (technique, case), reused across all its pairings.
    print(f"\nGenerating replies with {model}...")
    replies: Dict[Tuple[str, str], str] = {}
    for technique in techniques:
        for case in TEST_CASES:
            reply, usage = generate_reply(client, model, technique, case)
            total_cost += usage_cost(model, usage)
            if reply is None:
                print(f"  {technique} / {case['name']}: EMPTY REPLY — excluded")
            else:
                replies[(technique, case["name"])] = reply

    results = []
    print(f"\nJudging pairs with {judge_model}...")
    for case in TEST_CASES:
        for tech_1, tech_2 in itertools.combinations(techniques, 2):
            reply_1 = replies.get((tech_1, case["name"]))
            reply_2 = replies.get((tech_2, case["name"]))
            if reply_1 is None or reply_2 is None:
                continue

            # Randomize which technique is shown as "A" to cancel position bias.
            if random.random() < 0.5:
                shown_a, shown_b = (tech_1, reply_1), (tech_2, reply_2)
            else:
                shown_a, shown_b = (tech_2, reply_2), (tech_1, reply_1)

            judgment, usage = judge_pair(client, judge_model, case, shown_a[1], shown_b[1])
            total_cost += usage_cost(judge_model, usage)
            if judgment is None or judgment.get("winner") not in {"A", "B", "tie"}:
                print(f"  {case['name']}: {tech_1} vs {tech_2} — judge failed, skipped")
                continue

            label = judgment["winner"]
            winner = shown_a[0] if label == "A" else shown_b[0] if label == "B" else None
            results.append(
                {
                    "case": case["name"],
                    "model": model,
                    "technique_1": tech_1,
                    "technique_2": tech_2,
                    "winner": winner,  # None means tie
                    "rationale": judgment.get("rationale", ""),
                }
            )
            outcome = winner or "tie"
            print(f"  {case['name']}: {tech_1} vs {tech_2} → {outcome}")

    return results, total_cost


def summarize_pairwise(results: List[Dict], techniques: List[str]) -> None:
    """Print win rates per technique (a tie counts as half a win for both)."""
    if not results:
        print("\nNo successful comparisons to summarize.")
        return

    wins: Dict[str, float] = defaultdict(float)
    games: Dict[str, int] = defaultdict(int)
    for result in results:
        t1, t2, winner = result["technique_1"], result["technique_2"], result["winner"]
        games[t1] += 1
        games[t2] += 1
        if winner is None:
            wins[t1] += 0.5
            wins[t2] += 0.5
        else:
            wins[winner] += 1.0

    print(f"\n{'technique'.ljust(24)} | win rate | comparisons")
    print(f"{'-' * 24}-|----------|------------")
    ranked = sorted(
        (t for t in techniques if games[t]),
        key=lambda t: wins[t] / games[t],
        reverse=True,
    )
    for technique in ranked:
        rate = wins[technique] / games[technique]
        print(f"{technique.ljust(24)} | {rate:8.0%} | {games[technique]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge evaluation of the app's prompt techniques and models."
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=list(PROMPT_TECHNIQUES.keys()),
        choices=list(PROMPT_TECHNIQUES.keys()),
        help="Prompt techniques to compare (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_GENERATION_MODEL],
        help=f"Generation models to compare (default: {DEFAULT_GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model, kept separate from generation (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help=(
            "A/B-compare techniques instead of absolute 1-5 scoring — "
            "discriminates better when all techniques score near the top. "
            "Uses the first --models entry."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pairwise A/B order shuffling (default: 42).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write full results as JSON "
            "(default: evals_results.json, or evals_pairwise_results.json with --pairwise)."
        ),
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = "evals_pairwise_results.json" if args.pairwise else "evals_results.json"

    client = get_openrouter_client()

    if args.pairwise:
        random.seed(args.seed)
        model = args.models[0]
        n_pairs = len(list(itertools.combinations(args.techniques, 2))) * len(TEST_CASES)
        print(
            f"Pairwise: {len(args.techniques)} technique(s) on {len(TEST_CASES)} cases "
            f"= {len(args.techniques) * len(TEST_CASES)} generations + {n_pairs} "
            f"comparisons (model: {model}, judge: {args.judge_model})"
        )
        results, total_cost = run_pairwise(client, args.techniques, model, args.judge_model)
        summarize_pairwise(results, args.techniques)
    else:
        n_runs = len(args.techniques) * len(args.models) * len(TEST_CASES)
        print(
            f"Evaluating {len(args.techniques)} technique(s) x {len(args.models)} model(s) "
            f"on {len(TEST_CASES)} cases = {n_runs} generations + {n_runs} judge calls "
            f"(judge: {args.judge_model})"
        )
        results, total_cost = run_evals(client, args.techniques, args.models, args.judge_model)
        summarize(results)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results (replies, rationales, verdicts): {args.out}")
    print(f"Estimated total cost: {format_cost(total_cost)}")


if __name__ == "__main__":
    main()
