"""FastAPI app served from the supervisor process.

Two responsibilities:
  1. ``POST /api/connection-details`` — mints a LiveKit access token. This is
     the Python port of ``frontend/app/api/connection-details/route.ts``.
  2. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .interview_context import (
    fetch_analyses,
    fetch_analysis_detail,
    fetch_candidate_summary,
)

logger = logging.getLogger("api")


def _mint_token(cfg: Config, agent_name: str | None, uid: str | None = None) -> dict[str, Any]:
    participant_name = "user"
    rand = random.randint(0, 9999)
    if uid:
        # Room/identity names have a restricted charset; the raw uid still
        # travels in the participant metadata below.
        safe_uid = re.sub(r"[^A-Za-z0-9_-]", "", uid)[:32] or "candidate"
        participant_identity = f"candidate_{safe_uid}_{rand}"
        room_name = f"interview_{safe_uid}_{rand}"
    else:
        participant_identity = f"voice_assistant_user_{rand}"
        room_name = f"voice_assistant_room_{rand}"

    token = (
        lk_api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=15))
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
    )

    if agent_name:
        token = token.with_room_config(
            lk_api.RoomConfiguration(agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)])
        )

    if uid:
        # The agent reads this to look up the candidate's JD and resume.
        token = token.with_metadata(json.dumps({"uid": uid}))

    return {
        "serverUrl": cfg.livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


def build_app(
    cfg: Config,
    status_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        """Per-child readiness, polled by the frontend's first-boot splash.

        The web server starts before the children are ready (first boot can
        spend a long time downloading model weights), so this is how the UI
        knows whether the stack is usable yet.
        """
        children = status_provider() if status_provider is not None else []
        return {
            "ready": all(c["ready"] for c in children),
            "children": children,
            # Lets the frontend hint "say the wake phrase" when enabled.
            "wake_word": cfg.wake_word,
        }

    @app.get("/api/interview/{uid}")
    async def interview_status(uid: str) -> dict[str, Any]:
        """Frontend gate: does this interview link exist, and was it already taken?

        Returns found=False for unknown (or non-uuid) uids; 503 when the
        database is unreachable so the UI can show a retry screen.
        """
        try:
            summary = await asyncio.to_thread(fetch_candidate_summary, uid)
        except Exception as exc:
            logger.exception("candidate lookup failed for uid=%r", uid)
            raise HTTPException(status_code=503, detail="candidate lookup failed") from exc

        if summary is None:
            return {"found": False, "status": None, "candidateName": None, "roleTitle": None}
        return {
            "found": True,
            "status": summary.status,
            "candidateName": summary.name,
            "roleTitle": summary.role_title,
        }

    @app.get("/api/analyses")
    async def analyses() -> dict[str, Any]:
        """List view for the /analysis page: every graded interview, newest first."""
        try:
            items = await asyncio.to_thread(fetch_analyses)
        except Exception as exc:
            logger.exception("analysis list lookup failed")
            raise HTTPException(status_code=503, detail="analysis lookup failed") from exc
        return {"analyses": items}

    @app.get("/api/analyses/{analysis_id}")
    async def analysis_detail(analysis_id: int) -> dict[str, Any]:
        """Full rubric breakdown + transcript for one analysis row."""
        try:
            detail = await asyncio.to_thread(fetch_analysis_detail, analysis_id)
        except Exception as exc:
            logger.exception("analysis detail lookup failed for id=%s", analysis_id)
            raise HTTPException(status_code=503, detail="analysis lookup failed") from exc
        if detail is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        return detail

    @app.post("/api/connection-details")
    async def connection_details(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: str | None = None
        try:
            agent_name = body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
        except (AttributeError, IndexError, TypeError):
            agent_name = None

        uid: str | None = None
        if isinstance(body, dict) and body.get("uid"):
            uid = str(body["uid"])

        if uid:
            try:
                summary = await asyncio.to_thread(fetch_candidate_summary, uid)
            except Exception:
                # Fail open on DB outage: the UI gate already failed closed,
                # and the agent falls back to static context.
                logger.exception("candidate lookup failed; skipping status gate")
            else:
                if summary is None:
                    raise HTTPException(status_code=404, detail="unknown interview id")
                if summary.status != "not_started":
                    raise HTTPException(
                        status_code=409, detail="interview already started or completed"
                    )

        try:
            data = _mint_token(cfg, agent_name, uid)
        except Exception as exc:
            logger.exception("token minting failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if cfg.frontend_dir:
        # SPA-style: serve static export, falling back to index.html for unknown paths.
        static = StaticFiles(directory=cfg.frontend_dir, html=True)

        @app.get("/{path:path}")
        async def spa(path: str, request: Request) -> Any:
            try:
                return await static.get_response(path or "index.html", request.scope)
            except Exception:
                # trailingSlash:false exports real pages as <name>.html
                # (e.g. /analysis → analysis.html); try that before falling
                # back to the SPA entry point.
                try:
                    return await static.get_response(f"{path}.html", request.scope)
                except Exception:
                    return FileResponse(f"{cfg.frontend_dir}/index.html")

    return app
