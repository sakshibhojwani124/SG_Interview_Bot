"""Interview data access: the PostgreSQL ``candidates``,
``interview_transcripts``, and ``interview_analyses`` tables.

Single seam between the interview product and its data source. Both the
FastAPI gate (`/api/interview/{uid}`) and the agent (JD/resume for the prompt,
status transitions, transcript persistence) go through here.

Schema (see repo docs): candidates(uid uuid unique, name, role_title, jd_text,
resume_text, status 'not_started'|'in_progress'|'completed', ...);
interview_transcripts(id, candidate_uid uuid fk, transcript jsonb,
started_at, ended_at, created_at).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger("interview")

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"


@dataclass(frozen=True)
class CandidateSummary:
    """Just enough to gate the UI — no JD/resume, safe to send to the browser."""

    uid: str
    name: str
    role_title: str
    status: str  # not_started | in_progress | completed


@dataclass(frozen=True)
class InterviewContext:
    """Everything the agent needs to build its interviewer prompt."""

    uid: str
    candidate_name: str
    role: str
    job_description: str
    resume: str


def _connect() -> psycopg.Connection:
    dsn = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return psycopg.connect(dsn, connect_timeout=5)


def _parse_uid(uid: str | None) -> uuid.UUID | None:
    """The uid column is a uuid; a link with anything else is simply unknown."""
    if not uid:
        return None
    try:
        return uuid.UUID(uid)
    except ValueError:
        return None


def fetch_candidate_summary(uid: str | None) -> CandidateSummary | None:
    """Look up a candidate by uid. None = no such interview link."""
    key = _parse_uid(uid)
    if key is None:
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, role_title, status FROM candidates WHERE uid = %s", (key,))
        row = cur.fetchone()
    if row is None:
        return None
    name, role_title, status = row
    return CandidateSummary(uid=str(key), name=name, role_title=role_title, status=status)


def fetch_interview_context(uid: str | None) -> InterviewContext | None:
    """Full interview context (JD + resume) for the agent prompt. None = unknown uid."""
    key = _parse_uid(uid)
    if key is None:
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, role_title, jd_text, resume_text FROM candidates WHERE uid = %s",
            (key,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    name, role_title, jd_text, resume_text = row
    return InterviewContext(
        uid=str(key),
        candidate_name=name,
        role=role_title,
        job_description=jd_text or "(no job description on file)",
        resume=resume_text or "(no resume on file)",
    )


def save_interview_transcript(
    uid: str | None,
    transcript: list[dict[str, str]],
    started_at: datetime | None,
    ended_at: datetime | None,
) -> int | None:
    """Store a finished call's transcript: a list of {role, content} turns,
    role being 'interviewer' or 'candidate'. Returns the new row's id, or
    None if nothing was written.

    ``candidate_uid`` is a FK to candidates(uid), so this only works for
    interviews that came from the DB — not the static fallback.
    """
    key = _parse_uid(uid)
    if key is None:
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO interview_transcripts (candidate_uid, transcript, started_at, ended_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (key, Jsonb(transcript), started_at, ended_at),
        )
        row = cur.fetchone()
    return row[0] if row else None


def save_interview_analysis(
    uid: str | None,
    transcript_id: int | None,
    analysis: dict,
) -> bool:
    """Store the LLM grading of a finished interview (see
    interview_analysis.analyze_interview for the payload shape).
    Returns True if a row was written."""
    key = _parse_uid(uid)
    if key is None:
        return False
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO interview_analyses "
            "(candidate_uid, transcript_id, rubric_scores, overall_score, "
            " recommendation, summary, strengths, areas_of_concern, model) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                key,
                transcript_id,
                Jsonb(analysis["rubric_scores"]),
                analysis["overall_score"],
                analysis["recommendation"],
                analysis["summary"],
                Jsonb(analysis["strengths"]),
                Jsonb(analysis["areas_of_concern"]),
                analysis["model"],
            ),
        )
    return True


def fetch_analyses() -> list[dict]:
    """All analyses joined with candidate identity, newest first. Rows are
    JSON-shaped (camelCase) for the /api/analyses route backing the
    /analysis page."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, c.uid, c.name, c.role_title, a.overall_score, "
            "       a.recommendation, a.created_at "
            "FROM interview_analyses a "
            "JOIN candidates c ON c.uid = a.candidate_uid "
            "ORDER BY a.created_at DESC"
        )
        rows = cur.fetchall()
    return [
        {
            "analysisId": row[0],
            "uid": str(row[1]),
            "candidateName": row[2],
            "roleTitle": row[3],
            "overallScore": float(row[4]),
            "recommendation": row[5],
            "createdAt": row[6].isoformat() if row[6] else None,
        }
        for row in rows
    ]


def fetch_analysis_detail(analysis_id: int) -> dict | None:
    """One analysis with full rubric detail plus its transcript (when the
    transcript row was saved). None = no such analysis."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, c.uid, c.name, c.role_title, a.rubric_scores, "
            "       a.overall_score, a.recommendation, a.summary, a.strengths, "
            "       a.areas_of_concern, a.model, a.created_at, "
            "       t.transcript, t.started_at, t.ended_at "
            "FROM interview_analyses a "
            "JOIN candidates c ON c.uid = a.candidate_uid "
            "LEFT JOIN interview_transcripts t ON t.id = a.transcript_id "
            "WHERE a.id = %s",
            (analysis_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "analysisId": row[0],
        "uid": str(row[1]),
        "candidateName": row[2],
        "roleTitle": row[3],
        "rubricScores": row[4],
        "overallScore": float(row[5]),
        "recommendation": row[6],
        "summary": row[7],
        "strengths": row[8] or [],
        "areasOfConcern": row[9] or [],
        "model": row[10],
        "createdAt": row[11].isoformat() if row[11] else None,
        "transcript": row[12],
        "startedAt": row[13].isoformat() if row[13] else None,
        "endedAt": row[14].isoformat() if row[14] else None,
    }


def mark_interview_started(uid: str | None) -> bool:
    """Transition not_started → in_progress. Returns True if a row changed."""
    key = _parse_uid(uid)
    if key is None:
        return False
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE candidates SET status = 'in_progress' "
            "WHERE uid = %s AND status = 'not_started'",
            (key,),
        )
        return cur.rowcount > 0


def mark_interview_completed(uid: str | None) -> bool:
    """Transition in_progress → completed. Returns True if a row changed."""
    key = _parse_uid(uid)
    if key is None:
        return False
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE candidates SET status = 'completed' "
            "WHERE uid = %s AND status = 'in_progress'",
            (key,),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Static fallback — used by the agent when the DB has no row for the uid (or
# is unreachable), so a joined room never sits silent. Also handy in dev.
# ---------------------------------------------------------------------------

STATIC_JOB_DESCRIPTION = dedent(
    """\
    Role: Software Engineer — Backend (L1 screening)
    Company: Saint-Gobain
    Location: Mumbai, India (hybrid)

    We are looking for a backend engineer to build and maintain services that
    power Saint-Gobain's digital platforms.

    Responsibilities:
    - Design, build, and operate REST APIs in Python (FastAPI or Django).
    - Model and query relational data in PostgreSQL; keep queries efficient.
    - Write unit and integration tests; participate in code reviews.
    - Deploy and monitor services on AWS using Docker and CI/CD pipelines.
    - Collaborate with frontend and data teams on API contracts.

    Requirements:
    - 2-4 years of professional backend development experience.
    - Strong Python fundamentals: data structures, error handling, typing.
    - Solid SQL and data-modeling skills.
    - Working knowledge of Git, Docker, and at least one cloud provider.
    - Clear spoken communication and a collaborative working style.
    """
)

STATIC_RESUME = dedent(
    """\
    Name: Priya Sharma
    Title: Backend Developer, 3 years experience

    Experience:
    - Backend Developer, Meridian Software (2023-present): Built FastAPI
      microservices for an e-commerce order platform handling 50k orders/day.
      Designed PostgreSQL schemas, added Redis caching that cut p95 API
      latency from 480ms to 120ms, and containerized services with Docker on
      AWS ECS.
    - Junior Developer, TechNova Solutions (2022-2023): Maintained Django
      monolith for an internal HR tool; wrote Celery jobs for payroll report
      generation and raised test coverage from 40% to 75% with pytest.

    Skills: Python, FastAPI, Django, PostgreSQL, Redis, Celery, Docker,
    AWS (ECS, S3, RDS), Git, pytest.

    Education: B.E. Computer Engineering, University of Pune, 2022.
    """
)


def static_interview_context(uid: str | None) -> InterviewContext:
    return InterviewContext(
        uid=uid or "unknown",
        candidate_name="Priya Sharma",
        role="Software Engineer — Backend",
        job_description=STATIC_JOB_DESCRIPTION,
        resume=STATIC_RESUME,
    )
