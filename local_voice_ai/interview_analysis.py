"""Post-interview transcript grading.

After a call ends the agent hands the saved transcript, JD, and resume to the
same local LLM that ran the interview and asks it to score the candidate
against a fixed rubric. The result is persisted to ``interview_analyses``
(see db/interview_analyses.sql) via
``interview_context.save_interview_analysis``.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger("interview.analysis")

# Rubrics for an L1 screening call. ``key`` is the stable identifier stored in
# the DB, ``weight`` fractions sum to 1.0, and each dimension is scored 1-5.
RUBRICS: list[dict] = [
    {
        "key": "technical_knowledge",
        "name": "Technical Knowledge",
        "weight": 0.30,
        "description": (
            "Correctness and depth of technical answers relative to the "
            "skills the job description requires."
        ),
    },
    {
        "key": "experience_relevance",
        "name": "Experience Relevance",
        "weight": 0.20,
        "description": (
            "How well the candidate's described work maps to the role, and "
            "whether it is consistent with the resume claims."
        ),
    },
    {
        "key": "problem_solving",
        "name": "Problem Solving",
        "weight": 0.20,
        "description": (
            "Structured reasoning, awareness of trade-offs, and the quality "
            "of answers when the interviewer probes with follow-ups."
        ),
    },
    {
        "key": "communication",
        "name": "Communication",
        "weight": 0.20,
        "description": (
            "Clarity, conciseness, and coherence of spoken answers; ability "
            "to explain technical topics simply."
        ),
    },
    {
        "key": "professionalism",
        "name": "Professionalism & Engagement",
        "weight": 0.10,
        "description": (
            "Attitude, engagement with the interviewer, and professional "
            "conduct throughout the call."
        ),
    },
]

RECOMMENDATIONS = ("strong_yes", "yes", "maybe", "no")


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    lines = []
    for turn in transcript:
        speaker = "Interviewer" if turn.get("role") == "interviewer" else "Candidate"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


def build_analysis_prompt(
    role: str,
    job_description: str,
    resume: str,
    transcript: list[dict[str, str]],
) -> str:
    rubric_lines = "\n".join(
        f'- "{r["key"]}" ({r["name"]}): {r["description"]}' for r in RUBRICS
    )
    score_fields = ", ".join(
        f'"{r["key"]}": {{"score": <1-5>, "justification": "<1-2 sentences>"}}'
        for r in RUBRICS
    )
    return f"""You are an expert technical recruiter evaluating a completed L1 screening interview for the {role} position.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume}

INTERVIEW TRANSCRIPT:
{_format_transcript(transcript)}

Score the candidate on each rubric from 1 (poor) to 5 (excellent). Judge only what the transcript shows; do not give credit for resume claims the interview never touched. If the interview ended before a dimension could be assessed, score it low and say so in the justification.
{rubric_lines}

Respond with ONLY a JSON object, no markdown and no other text, in exactly this shape:
{{"scores": {{{score_fields}}}, "strengths": ["<short bullet>", "..."], "areas_of_concern": ["<short bullet>", "..."], "summary": "<3-4 sentence overall assessment>", "recommendation": "<one of: strong_yes, yes, maybe, no>"}}"""


def _extract_json(text: str) -> dict:
    """Small local models wrap JSON in fences or prose; dig the object out."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _overall_score(scores_by_key: dict[str, int]) -> float:
    weighted = sum(r["weight"] * (scores_by_key[r["key"]] / 5) for r in RUBRICS)
    return round(weighted * 100, 2)


def _fallback_recommendation(overall: float) -> str:
    if overall >= 85:
        return "strong_yes"
    if overall >= 70:
        return "yes"
    if overall >= 50:
        return "maybe"
    return "no"


def analyze_interview(
    role: str,
    job_description: str,
    resume: str,
    transcript: list[dict[str, str]],
) -> dict:
    """Grade a finished interview with the local LLM.

    Returns the payload ``save_interview_analysis`` expects. Raises on LLM or
    parse failure — the caller decides whether that is fatal.
    """
    base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    model = os.getenv("ANALYSIS_MODEL") or os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")
    timeout = float(os.getenv("ANALYSIS_TIMEOUT", "120"))

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    messages = [
        {
            "role": "user",
            "content": build_analysis_prompt(role, job_description, resume, transcript),
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some OpenAI-compatible servers reject response_format; ask plainly.
        logger.warning("json response_format rejected; retrying without it")
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2
        )
    data = _extract_json(resp.choices[0].message.content or "")

    scores = data.get("scores") or {}
    rubric_scores: list[dict] = []
    scores_by_key: dict[str, int] = {}
    for rubric in RUBRICS:
        entry = scores.get(rubric["key"])
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"model output missing score for {rubric['key']!r}")
        score = min(5, max(1, int(entry["score"])))
        scores_by_key[rubric["key"]] = score
        rubric_scores.append(
            {
                "key": rubric["key"],
                "name": rubric["name"],
                "weight": rubric["weight"],
                "score": score,
                "justification": str(entry.get("justification", "")).strip(),
            }
        )

    overall = _overall_score(scores_by_key)
    recommendation = str(data.get("recommendation", "")).strip().lower()
    if recommendation not in RECOMMENDATIONS:
        recommendation = _fallback_recommendation(overall)

    return {
        "rubric_scores": rubric_scores,
        "overall_score": overall,
        "recommendation": recommendation,
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(s) for s in data.get("strengths") or []],
        "areas_of_concern": [str(s) for s in data.get("areas_of_concern") or []],
        "model": model,
    }
