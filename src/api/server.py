"""
REST API Server - Communication Bridge Between User Chatbot and Admin.

This FastAPI server provides endpoints for:
1. Submitting reservation requests (called by the user chatbot)
2. Listing pending reservations (called by the admin agent)
3. Approving/rejecting reservations (called by the admin agent)

WHY A REST API?
The Stage 2 task says: "Chat bot should be able to send a reservation request
to administrator and get confirm/refuse response from him (e.g. via email
server, messenger, rest api)."

We use REST API because:
- Decouples the user chatbot from the admin agent (they don't need to run together)
- Standard HTTP interface (can be called from CLI, web UI, or other agents)
- Easy to test and debug
- Can be extended with email/webhook notifications

ENDPOINTS:
    POST   /api/reservations              - Submit a new reservation
    GET    /api/reservations              - List reservations (filter by ?status=pending)
    GET    /api/reservations/{id}         - Get a specific reservation
    PUT    /api/reservations/{id}/approve - Admin approves a reservation
    PUT    /api/reservations/{id}/reject  - Admin rejects a reservation
    GET    /api/health                    - Health check

HOW TO RUN:
    python -m uvicorn src.api.server:app --reload --port 8000
    Then visit http://localhost:8000/docs for interactive Swagger UI
"""

import os
import sys
import traceback as _traceback

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.database.postgres import get_db_info, validate_connection
from src.database.session import db_session
from src.database.sql_store import SQLStore
from src.notifications.email_service import EmailService
from src.services.booking_service import BookingService, BookingServiceError
from src.services.booking_validation_service import BookingValidationError
from src.services.parking_service import ParkingService
from src.services.payment_service import (
    AlreadyPaidError,
    PaymentError,
    PaymentExpiredError,
    PaymentNotFoundError,
    PaymentService,
)
from src.utils.logging_config import setup_logging
from src.utils.masking import mask_email

# Initialize structured logging
setup_logging()

import logging as _logging_module
_log = _logging_module.getLogger(__name__)


# ========================
# PYDANTIC REQUEST/RESPONSE MODELS
# ========================
# These define the shape of JSON data sent/received by the API.
# FastAPI uses them for validation AND auto-generated Swagger docs.


class ReservationRequest(BaseModel):
    """JSON body for creating a new reservation (sent by chatbot)."""

    first_name: str = Field(..., json_schema_extra={"example": "John"})
    last_name: str = Field(..., json_schema_extra={"example": "Smith"})
    email: Optional[str] = Field(None, json_schema_extra={"example": "john@example.com"})
    car_number: str = Field(..., json_schema_extra={"example": "ABC-1234"})
    space_type: str = Field(..., json_schema_extra={"example": "standard"})
    start_datetime: str = Field(..., json_schema_extra={"example": "2026-05-10 09:00"})
    end_datetime: str = Field(..., json_schema_extra={"example": "2026-05-10 18:00"})


class ReservationResponse(BaseModel):
    """JSON response when returning reservation data."""

    id: int
    first_name: str
    last_name: str
    email: Optional[str] = None
    car_number: str
    space_type: str
    start_datetime: str
    end_datetime: str
    status: str
    admin_notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    approved_at: Optional[str] = None


class AdminActionRequest(BaseModel):
    """JSON body for admin approve/reject action."""

    admin_notes: Optional[str] = Field(None, json_schema_extra={"example": "Approved - VIP customer"})


class StatusResponse(BaseModel):
    """Generic status response."""

    success: bool
    message: str


# ========================
# FASTAPI APPLICATION
# ========================

app = FastAPI(
    title="ParkSmart Reservation API",
    description=(
        "REST API for the ParkSmart Parking Reservation System.\n\n"
        "Used for communication between the user-facing chatbot and "
        "the admin agent for reservation approval."
    ),
    version="1.0.0",
)

# CORS — configurable via settings.cors_origins
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # all Vercel preview/prod URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared database instance (initialized once when the server starts)
sql_store = SQLStore()
sql_store.initialize_default_data()

# Production service layer (PostgreSQL-backed)
_parking_svc = ParkingService()
_booking_svc = BookingService()
_payment_svc = PaymentService()


@app.on_event("startup")
def _startup_db_check() -> None:
    """Kick off background pipeline init, then run DB health/seed checks."""
    import logging

    _log = logging.getLogger(__name__)

    # ── CRITICAL: Start the AI pipeline thread FIRST ──────────────────────
    # Must happen before any DB checks that may fail or return early.
    # Previously the thread start was at the END of this function, so if
    # validate_connection() failed (e.g. cold PostgreSQL on Render) the
    # thread was never started and /api/ready returned 503 forever.
    app.state.initialized = False
    app.state.pipeline_error = None
    _log.info("[STARTUP] ── Server startup (PID=%d) ──", os.getpid())
    _log.info("[STARTUP] Launching background pipeline init thread...")
    _init_thread = _threading.Thread(
        target=_init_pipeline_background, daemon=True, name="pipeline-init"
    )
    _init_thread.start()
    _log.info(
        "[STARTUP] Thread launched — name=%s  alive=%s",
        _init_thread.name,
        _init_thread.is_alive(),
    )

    # ── DB health / seed checks (informational — never block pipeline) ────
    # Ensure payment table exists (idempotent — safe to run every startup)
    try:
        from src.database.base import Base
        from src.database.postgres import get_engine
        import src.models.payment  # noqa: F401 — registers Payment with Base.metadata

        Base.metadata.create_all(bind=get_engine(), checkfirst=True)
        _log.info("Payment table ensured (create_all checkfirst=True)")
    except Exception as _exc:
        _log.warning("Could not ensure payment table: %s", _exc)
    info = get_db_info()

    _log.info("─" * 48)
    _log.info("Database URL    : %s", info["database_url"])
    if info["absolute_path"]:
        _log.info("Absolute Path   : %s", info["absolute_path"])
    _log.info("Driver          : %s", info["driver"])

    ok = validate_connection()
    if not ok:
        _log.error("Production DB UNREACHABLE — continuing with degraded availability")
        _log.info("─" * 48)
        return

    # Count rows in production tables so we know whether seed ran
    try:
        from src.models.parking_slot import ParkingSlot
        from src.models.parking_type import ParkingType

        with db_session() as db:
            type_count = db.query(ParkingType).count()
            slot_count = db.query(ParkingSlot).count()

        _log.info("Parking Types   : %d", type_count)
        _log.info("Total Slots     : %d", slot_count)
        _log.info("─" * 48)

        if type_count == 0:
            _log.warning("Database empty — auto-seeding parking types and slots...")
            _auto_seed(_log)
        else:
            # Sync available_slots counters from actual slot statuses.
            # This corrects any drift caused by crashes, seed resets, or manual edits.
            _sync_slot_counters()
    except Exception as exc:  # pragma: no cover
        _log.warning("Could not query production tables: %s", exc)
        _log.info("─" * 48)

    # NOTE: thread is started at the TOP of this function (above DB checks)
    # so it is already running by this point.


def _auto_seed(_log) -> None:
    """
    Run automatically on first boot when the database is empty.
    Seeds parking types and all physical slot rows, then syncs counters.
    Safe to call multiple times — all operations are idempotent.
    """
    try:
        from src.database.seeder import seed_parking_types, seed_slots
        from src.models.parking_slot import ParkingSlot
        from src.models.parking_type import ParkingType

        with db_session() as db:
            types = seed_parking_types(db)
            created = seed_slots(db, types)

        _log.info("Auto-seed complete: %d new slot rows created", created)
        _sync_slot_counters()
    except Exception as exc:
        _log.error("Auto-seed failed: %s", exc)


def _sync_slot_counters() -> None:
    """
    Recompute each parking_type.available_slots from the actual count of
    parking_slots rows with status='available'.

    Called at startup to correct any counter drift caused by:
    - seed script re-runs (which used to reset the counter)
    - server crashes mid-transaction
    - manual DB edits
    """
    try:
        from src.models.parking_slot import ParkingSlot
        from src.models.parking_type import ParkingType

        with db_session() as db:
            types = db.query(ParkingType).all()
            synced = 0
            for pt in types:
                actual = (
                    db.query(ParkingSlot)
                    .filter_by(parking_type_id=pt.id, status="available")
                    .count()
                )
                if pt.available_slots != actual:
                    _log.info(
                        "[sync] %s: counter %d → actual %d",
                        pt.slug.upper(), pt.available_slots, actual,
                    )
                    pt.available_slots = actual
                    synced += 1
        if synced:
            _log.info("Counter sync: fixed %d parking type(s)", synced)
        else:
            _log.info("Counter sync: all counters correct")
    except Exception as exc:
        _log.warning("Counter sync failed (non-fatal): %s", exc)

# Email notification service
email_service = EmailService()

# ---------------------------------------------------------------------------
# Pipeline state — managed by background init thread
# ---------------------------------------------------------------------------
import threading as _threading

_pipeline = None
_pipeline_ready = _threading.Event()   # set when pipeline finishes (success OR failure)
_pipeline_init_error: Exception | None = None
_sessions: dict[str, dict] = {}  # session_id -> pipeline_state

# app.state.initialized / app.state.pipeline_error are set inside the thread.
# app.state.vector_store is None until the lazy VS loader finishes.

# Vector-store lazy-load state ─────────────────────────────────────────────
_vs_loading: bool = False          # True while the background VS thread is running
_vs_load_lock = _threading.Lock()  # prevents duplicate VS loading attempts


def _init_pipeline_background() -> None:
    """
    Daemon thread: lightweight pipeline init — SQL, chatbot (SQL-only), agents, graph.

    Vector store (Pinecone/HuggingFace) is intentionally EXCLUDED from this path.
    It is heavy (30-90s on cold Render start) and blocks readiness.
    Instead, it is loaded by _load_vector_store_background() AFTER this thread
    marks the system ready, so /api/ready returns 200 without waiting for it.

    Flow:
        SQL → RAG (SQL-only) → Chatbot (SQL-only) → Agents → Graph → READY
        ↓ (after READY)
        Vector store loads in background → injected into live RAGChain (hot-swap)
    """
    global _pipeline, _pipeline_init_error

    pid = os.getpid()
    tid = _threading.get_ident()
    _log.info("[PIPELINE] START  (PID=%d  TID=%d)", pid, tid)

    try:
        # ── Stage 1: SQL store ───────────────────────────────────────────────
        _log.info("[PIPELINE] SQL INIT START")
        from src.database.sql_store import SQLStore as _SQLStore  # noqa: PLC0415
        _sql = sql_store or _SQLStore()
        _sql.initialize_default_data()
        _log.info("[PIPELINE] SQL INIT DONE")

        # ── Stage 2: RAG chain (SQL-only — no VectorStore yet) ───────────────
        # Vector store will be injected later via RAGChain.set_vector_store().
        _log.info("[PIPELINE] RAG INIT START (SQL-only mode — vector store loads lazily after ready)")
        from src.chatbot.rag_chain import RAGChain  # noqa: PLC0415
        _rag = RAGChain(sql_store=_sql, skip_vector_store=True)
        _log.info("[PIPELINE] RAG DONE")

        # ── Stage 3: Chatbot ─────────────────────────────────────────────────
        _log.info("[PIPELINE] CHATBOT INIT START")
        from src.chatbot.chatbot import ParkingChatbot  # noqa: PLC0415
        _bot = ParkingChatbot(skip_vector_store=True)
        _bot.sql_store = _sql
        _bot.rag_chain = _rag
        _log.info("[PIPELINE] CHATBOT INIT DONE")

        # ── Stage 4: Email service & MCP client ─────────────────────────────
        _log.info("[PIPELINE] EMAIL + MCP INIT START")
        from src.notifications.email_service import EmailService as _EmailSvc  # noqa: PLC0415
        from src.mcp.mcp_client import MCPClient  # noqa: PLC0415
        _email = email_service or _EmailSvc()
        _mc = MCPClient()
        _log.info("[PIPELINE] EMAIL + MCP DONE")

        # ── Stage 5: Admin agent ─────────────────────────────────────────────
        _log.info("[PIPELINE] AGENTS INIT START")
        from src.agents.admin_agent import AdminAgent  # noqa: PLC0415
        _ag = AdminAgent(sql_store=_sql)
        _log.info("[PIPELINE] AGENTS DONE")

        # ── Stage 6: Build LangGraph pipeline ────────────────────────────────
        _log.info("[PIPELINE] BUILDING GRAPH")
        from src.graph.pipeline import create_pipeline  # noqa: PLC0415
        _built = create_pipeline(
            chatbot=_bot,
            sql_store=_sql,
            email_service=_email,
            mcp_client=_mc,
            admin_agent=_ag,
        )

        if _built is None:
            raise RuntimeError("create_pipeline() returned None — graph compilation failed")

        # ── Mark ready BEFORE touching the vector store ───────────────────────
        _pipeline = _built
        app.state.initialized = True
        app.state.pipeline_error = None
        app.state.vector_store = None  # populated by _load_vector_store_background
        _log.info("[PIPELINE] LIGHTWEIGHT STARTUP COMPLETE ✓  (PID=%d  TID=%d)", pid, tid)
        _log.info("[PIPELINE] Chat available now (SQL-only mode). Full RAG loading in background...")

    except Exception as exc:
        _pipeline_init_error = exc
        app.state.initialized = False
        app.state.pipeline_error = str(exc)
        _log.error("[PIPELINE] FAILED: %s", exc)
        _log.error("[PIPELINE] Full traceback:\n%s", _traceback.format_exc())

    finally:
        # Always signal so the frontend is never stuck polling forever.
        _pipeline_ready.set()
        _log.info(
            "[PIPELINE] Ready-event set  initialized=%s  error=%s",
            getattr(app.state, "initialized", False),
            getattr(app.state, "pipeline_error", None),
        )

    # ── Kick off vector store lazy load AFTER signalling ready ────────────
    # Runs only if pipeline init succeeded.
    if getattr(app.state, "initialized", False):
        _trigger_vector_store_load()


def _trigger_vector_store_load() -> None:
    """Start the vector-store background loader if not already running."""
    global _vs_loading
    with _vs_load_lock:
        if _vs_loading or getattr(app.state, "vector_store", None) is not None:
            return  # already loading or already loaded
        _vs_loading = True
        _t = _threading.Thread(
            target=_load_vector_store_background, daemon=True, name="vs-lazy-load"
        )
        _t.start()
        _log.info("[VECTOR] Lazy loading started (background thread)")


def _load_vector_store_background() -> None:
    """
    Background daemon thread: load Pinecone/HuggingFace vector store.

    Runs AFTER the pipeline is already marked ready, so it never blocks
    /api/ready or the chat endpoint.  Once loaded it hot-swaps the retriever
    inside the live RAGChain via RAGChain.set_vector_store().

    This thread is the ONLY place torch / langchain_huggingface are imported,
    so there are no import-lock conflicts with the lightweight pipeline thread.
    """
    global _vs_loading
    _log.info("[VECTOR] Lazy loading START (PID=%d  TID=%d)", os.getpid(), _threading.get_ident())
    try:
        from src.database.vector_store import VectorStore as _VS  # noqa: PLC0415
        _log.info("[VECTOR] Importing HuggingFace + Pinecone (may take 30-90s on cold start)...")
        _vs = _VS()
        _log.info("[VECTOR] VectorStore initialised — injecting into live RAGChain...")

        # Cache in app.state
        app.state.vector_store = _vs

        # Hot-swap retriever in the live chatbot's RAGChain
        from src.graph import nodes as _nodes  # noqa: PLC0415
        if _nodes._chatbot and hasattr(_nodes._chatbot, "rag_chain"):
            _nodes._chatbot.rag_chain.set_vector_store(_vs)
            _nodes._chatbot.vector_store = _vs
            _log.info("[VECTOR] Loaded successfully ✓ — chatbot upgraded to full RAG mode")
        else:
            _log.warning("[VECTOR] Loaded but chatbot not available — VS cached for future use")

    except Exception as exc:
        _log.error("[VECTOR] Lazy load FAILED: %s", exc)
        _log.error("[VECTOR] Full traceback:\n%s", _traceback.format_exc())
    finally:
        _vs_loading = False


def _get_pipeline():
    """
    Return the shared pipeline if ready, or None if still initializing.
    Never blocks — callers must handle the None case gracefully.
    """
    if not _pipeline_ready.is_set():
        return None  # still loading
    if _pipeline_init_error:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline initialization failed: {_pipeline_init_error}"
        )
    return _pipeline


def _get_session_state(session_id: str) -> dict:
    """Get or create pipeline state for a session."""
    from src.graph.pipeline import create_initial_state

    if session_id not in _sessions:
        state = create_initial_state()
        state["session_id"] = session_id
        _sessions[session_id] = state
    return _sessions[session_id]


class ChatRequest(BaseModel):
    """JSON body for a chat message."""

    message: str = Field(..., json_schema_extra={"example": "What are your parking rates?"})
    session_id: Optional[str] = Field(None, json_schema_extra={"example": "abc-123"})


class ChatResponse(BaseModel):
    """JSON response from the chatbot."""

    response: str
    is_booking_flow: bool = False
    reservation_id: Optional[int] = None
    session_id: Optional[str] = None
    booking_progress: Optional[dict] = None


# ========================
# CHAT ENDPOINT
# ========================


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to the ParkSmart chatbot and receive a response.
    Supports general Q&A and the full reservation booking flow.
    Each session_id gets its own isolated conversation state.
    """
    import uuid

    from src.graph.pipeline import run_user_message

    session_id = request.session_id or str(uuid.uuid4())
    _log.info("[CHAT] Received message from session %s: %s", session_id[:8], request.message[:50])

    pipeline = _get_pipeline()

    # If pipeline is still warming up, return immediately with a friendly message.
    # The frontend will show this and the user can retry in a few seconds.
    if pipeline is None:
        _log.info("[CHAT] Pipeline not ready yet — returning warm-up response")
        return ChatResponse(
            response=(
                "⏳ I'm still loading my AI components (this takes ~30 seconds on first start). "
                "Please try again in a moment!"
            ),
            is_booking_flow=False,
            session_id=session_id,
        )

    state = _get_session_state(session_id)

    try:
        # Ensure session_id flows through pipeline state
        state["session_id"] = session_id
        result = run_user_message(pipeline, request.message, state)
        _sessions[session_id] = result

        # Get booking progress from the chatbot's session state
        from src.graph.nodes import _chatbot

        booking_progress = None
        if _chatbot:
            booking_progress = _chatbot.get_booking_progress(session_id)

        return ChatResponse(
            response=result.get("bot_response", "Sorry, I couldn't process your request."),
            is_booking_flow=result.get("is_booking_flow", False),
            reservation_id=result.get("reservation_id") or None,
            session_id=session_id,
            booking_progress=booking_progress,
        )
    except Exception as e:
        _log.error("[CHAT] Error processing message: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/chat/reset", response_model=StatusResponse)
def chat_reset(session_id: Optional[str] = None):
    """
    Reset conversation state for a session.
    If session_id is provided, resets that session (pipeline + chatbot).
    If not provided, creates a fresh session.
    """
    from src.graph.nodes import _chatbot

    if session_id and session_id in _sessions:
        del _sessions[session_id]
    # Also reset the chatbot's internal session state
    if session_id and _chatbot:
        _chatbot.reset_session(session_id)
    return StatusResponse(success=True, message="Chat session reset successfully")


@app.post("/api/chat/cancel-booking", response_model=ChatResponse)
def cancel_booking(session_id: Optional[str] = None):
    """Cancel an in-progress booking for the given session."""
    from src.graph.nodes import _chatbot

    if not _chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    sid = session_id or "default"
    message = _chatbot.cancel_booking(sid)
    booking_progress = _chatbot.get_booking_progress(sid)

    # Update pipeline state if it exists
    if sid in _sessions:
        _sessions[sid]["is_booking_flow"] = False

    return ChatResponse(
        response=message,
        is_booking_flow=False,
        session_id=sid,
        booking_progress=booking_progress,
    )


@app.get("/api/chat/sessions")
def chat_sessions():
    """List active chat session IDs."""
    return {
        "sessions": [
            {"session_id": sid, "phase": state.get("conversation_phase", "unknown")} for sid, state in _sessions.items()
        ]
    }


# ========================
# API ENDPOINTS
# ========================


@app.get("/api/health")
def health_check():
    """
    Health check endpoint.
    Returns OK if the server is running and DB is accessible.
    """
    return {"status": "healthy", "service": "ParkSmart Reservation API"}


@app.get("/")
def root():
    """
    Root endpoint — returns 200 so Render's default healthcheck succeeds.
    Use /api/ready for AI pipeline readiness; use /api/health for liveness.
    """
    return {"service": "ParkSmart API", "status": "running"}


@app.get("/api/ready")
def ready_check():
    """
    Readiness endpoint — tells the frontend whether the AI pipeline is loaded.
    Returns HTTP 200 + {ready: true}  when fully initialized.
    Returns HTTP 503 + {ready: false} while still loading (frontend polls this).
    """
    from fastapi.responses import JSONResponse

    initialized = getattr(app.state, "initialized", False)
    error = getattr(app.state, "pipeline_error", None)
    healthy = initialized and _pipeline is not None and _pipeline_init_error is None

    if not healthy:
        if error or _pipeline_init_error:
            msg = f"Initialization failed: {error or _pipeline_init_error}"
        else:
            msg = "AI pipeline is loading, please wait..."
        return JSONResponse(
            status_code=503,
            content={"ready": False, "status": "initializing", "message": msg},
        )

    return {"ready": True, "status": "ready", "message": "All systems operational"}


@app.get("/debug/state")
def debug_state():
    """
    Diagnostic endpoint — exposes internal initialization state.
    Useful for confirming the background thread ran and what stage it reached.
    """
    return {
        "pid": os.getpid(),
        "pipeline_thread_running": any(
            t.name == "pipeline-init" for t in _threading.enumerate()
        ),
        "pipeline_ready_event": _pipeline_ready.is_set(),
        "pipeline_is_none": _pipeline is None,
        "pipeline_init_error": str(_pipeline_init_error) if _pipeline_init_error else None,
        "app_state_initialized": getattr(app.state, "initialized", False),
        "app_state_pipeline_error": getattr(app.state, "pipeline_error", None),
        # Vector store lazy-load status
        "vector_store_loaded": getattr(app.state, "vector_store", None) is not None,
        "vector_store_loading": _vs_loading,
        "vs_thread_running": any(
            t.name == "vs-lazy-load" for t in _threading.enumerate()
        ),
        "active_sessions": len(_sessions),
    }


@app.get("/api/health/detailed")
def health_check_detailed():
    """
    Detailed health check — validates database connectivity and vector DB config.
    Used by monitoring systems and Docker HEALTHCHECK.
    """
    checks = {
        "service": "ParkSmart Reservation API",
        "status": "healthy",
        "database": "unknown",
        "vector_db": "unknown",
        "email": "unknown",
    }

    # Check SQL database connectivity
    try:
        sql_store.get_reservations(status="pending")
        checks["database"] = "connected"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        checks["status"] = "degraded"

    # Check Pinecone config presence
    from config.settings import settings

    if settings.pinecone_api_key:
        checks["vector_db"] = "configured"
    else:
        checks["vector_db"] = "not configured"
        checks["status"] = "degraded"

    # Check email config
    if settings.smtp_host and settings.smtp_username:
        checks["email"] = "configured"
    else:
        checks["email"] = "console fallback"

    return checks


@app.post("/api/reservations", response_model=ReservationResponse, status_code=201)
def create_reservation(request: ReservationRequest):
    """
    Submit a new reservation request.

    Called by the user-facing chatbot when a user confirms their booking.
    Validates slot availability BEFORE creating the reservation — returns
    HTTP 409 if the requested parking type is full or inactive.
    """
    # ── pre-flight availability check ────────────────────────────────────────
    space_type = request.space_type.lower()
    try:
        from src.models.parking_type import ParkingType as _PT
        with db_session() as _db:
            pt = _db.query(_PT).filter_by(slug=space_type).first()
            if pt is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown parking type: '{space_type}'. "
                           "Valid types: standard, large, ev, vip, disabled, bike",
                )
            if not pt.is_active:
                raise HTTPException(
                    status_code=409,
                    detail=f"'{pt.name}' is currently disabled. Please choose another type.",
                )
            if pt.available_slots <= 0:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"No {pt.name} slots available right now. "
                        f"Total capacity: {pt.total_slots}. "
                        "All slots are either reserved or occupied."
                    ),
                )
    except HTTPException:
        raise
    except Exception as e:
        _log.warning("Pre-booking availability check failed (non-fatal): %s", e)
        # Non-fatal — fall through and let save_reservation handle it

    # ── create the reservation record ────────────────────────────────────────
    reservation_data = request.model_dump()
    try:
        reservation_id = sql_store.save_reservation(reservation_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reservation: {str(e)}")

    # Fetch the saved reservation to return it
    reservation = sql_store.get_reservation_by_id(reservation_id)

    # Notify admin via email (non-blocking — doesn't fail if email fails)
    try:
        email_service.notify_new_reservation(reservation)
    except Exception as e:
        print(f"⚠ Email notification failed: {e}")

    return reservation


@app.get("/api/reservations", response_model=list[ReservationResponse])
def list_reservations(status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected")):
    """
    List all reservations, optionally filtered by status.

    The admin agent calls this to see pending reservations that need review.
    """
    valid_statuses = ["pending", "approved", "rejected", None]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: pending, approved, rejected")

    reservations = sql_store.get_reservations(status=status)
    return reservations


@app.get("/api/reservations/{reservation_id}", response_model=ReservationResponse)
def get_reservation(reservation_id: int):
    """
    Get details of a specific reservation by ID.
    """
    reservation = sql_store.get_reservation_by_id(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation #{reservation_id} not found")
    return reservation


@app.put("/api/reservations/{reservation_id}/approve", response_model=StatusResponse)
def approve_reservation(reservation_id: int, request: AdminActionRequest = None):
    """
    Admin approves a pending reservation.

    Updates the status to 'approved' and records the timestamp.
    Optionally includes admin notes.
    """
    # Check if reservation exists
    reservation = sql_store.get_reservation_by_id(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation #{reservation_id} not found")

    if reservation["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Reservation #{reservation_id} is already {reservation['status']}")

    admin_notes = request.admin_notes if request else None
    success = sql_store.update_reservation_status(reservation_id, "approved", admin_notes)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update reservation")

    # Transition the pre-reserved slot: reserved → occupied.
    # Counter was ALREADY decremented when the booking was created (pending stage).
    # Never decrement again here — doing so would double-count.
    slot_number = sql_store.occupy_reserved_slot(reservation_id)

    # Legacy table fallback: if the slot was never pre-reserved (old reservation
    # created before this fix), decrement the legacy parking_availability table only.
    if slot_number is None:
        try:
            sql_store.update_availability(reservation["space_type"], delta=-1)
        except Exception as e:
            _log.warning("Legacy availability update failed: %s", e)
        # Also decrement production counter for old-style reservations
        try:
            from src.database.session import db_session as _dbs
            from src.repositories.parking_repository import ParkingRepository as _PR
            with _dbs() as _db:
                pt = _PR().get_by_slug(_db, reservation["space_type"])
                if pt and pt.available_slots > 0:
                    pt.available_slots = pt.available_slots - 1
        except Exception as e:
            _log.warning("Production counter update failed: %s", e)

    _log.info(
        "[APPROVED] Reservation #%d — %s  slot=%s",
        reservation_id, reservation["space_type"].upper(), slot_number or "(legacy)",
    )

    # ── Create payment record and send approval email with payment link ───
    payment_token: str = ""
    try:
        # Calculate the amount for this reservation
        try:
            bd = _parking_svc.calculate_price(
                reservation["space_type"],
                reservation["start_datetime"],
                reservation["end_datetime"],
            )
            amount_inr = bd["total_inr"]
        except Exception:
            try:
                total, _, _ = sql_store.calculate_price(
                    reservation["space_type"],
                    reservation["start_datetime"],
                    reservation["end_datetime"],
                )
                amount_inr = total
            except Exception:
                amount_inr = 0.0

        from decimal import Decimal
        payment_data = _payment_svc.create_payment(
            reservation_id=reservation_id,
            amount_inr=Decimal(str(amount_inr)),
            user_name=f"{reservation['first_name']} {reservation['last_name']}",
            user_email=reservation.get("email"),
            vehicle_number=reservation.get("car_number", ""),
            space_type=reservation["space_type"],
            start_datetime=reservation["start_datetime"],
            end_datetime=reservation["end_datetime"],
        )
        payment_token = payment_data["payment_token"]
        _log.info("[APPROVED] Payment created token=%s... amount=₹%.2f", payment_token[:8], amount_inr)
    except Exception as e:
        _log.warning("[APPROVED] Payment creation failed (non-fatal): %s", e)

    # Notify the user via email with the payment link
    try:
        updated = sql_store.get_reservation_by_id(reservation_id)
        payment_link = (
            f"{settings.app_base_url}/payment/{payment_token}" if payment_token else ""
        )
        email_service.send_user_approval(updated, payment_link=payment_link)
    except Exception as e:
        print(f"⚠ User notification failed: {e}")

    # Write approved reservation to file via MCP server (Stage 3)
    try:
        from src.mcp.mcp_client import MCPClient

        mcp_client = MCPClient()
        mcp_client.write_reservation_to_file(
            name=f"{reservation['first_name']} {reservation['last_name']}",
            car_number=reservation["car_number"],
            reservation_period=(f"{reservation['start_datetime']} - {reservation['end_datetime']}"),
        )
    except Exception as e:
        print(f"⚠ MCP file write failed: {e}")

    return StatusResponse(
        success=True,
        message=f"Reservation #{reservation_id} approved for {reservation['first_name']} {reservation['last_name']}",
    )


@app.put("/api/reservations/{reservation_id}/reject", response_model=StatusResponse)
def reject_reservation(reservation_id: int, request: AdminActionRequest = None):
    """
    Admin rejects a pending reservation.

    Updates the status to 'rejected' with optional reason.
    Since the slot was never allocated (only approved bookings claim a slot),
    no availability change is needed here.
    """
    reservation = sql_store.get_reservation_by_id(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation #{reservation_id} not found")

    if reservation["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Reservation #{reservation_id} is already {reservation['status']}")

    admin_notes = request.admin_notes if request else None
    success = sql_store.update_reservation_status(reservation_id, "rejected", admin_notes)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update reservation")

    # Release the pre-reserved slot back to available and increment counter.
    released = sql_store.release_reserved_slot(reservation_id)

    # Fallback: if no physical slot was pre-reserved (e.g. the slot reservation
    # step failed silently at booking time), still increment available_slots so
    # the counter stays consistent with reality.
    if not released:
        try:
            from src.models.parking_type import ParkingType as _PT2
            with db_session() as _db2:
                pt2 = _db2.query(_PT2).filter_by(slug=reservation["space_type"]).first()
                if pt2:
                    # Only increment if counter is strictly below total
                    actual_available = (
                        _db2.query(__import__("src.models.parking_slot", fromlist=["ParkingSlot"]).ParkingSlot)
                        .filter_by(parking_type_id=pt2.id, status="available")
                        .count()
                    )
                    if pt2.available_slots < pt2.total_slots:
                        pt2.available_slots = min(pt2.total_slots, actual_available + 1)
                        _log.info(
                            "[REJECTED] Reservation #%d — no pre-reserved slot found; "
                            "counter restored: %s available_slots → %d",
                            reservation_id, reservation["space_type"].upper(), pt2.available_slots,
                        )
        except Exception as _e:
            _log.warning("Rejection counter fallback failed for #%d: %s", reservation_id, _e)

    _log.info(
        "[REJECTED] Reservation #%d — slot released=%s",
        reservation_id, released,
    )

    # Notify the user via email about the rejection
    try:
        updated = sql_store.get_reservation_by_id(reservation_id)
        email_service.notify_user_status_change(updated)
    except Exception as e:
        print(f"⚠ User notification failed: {e}")

    return StatusResponse(
        success=True,
        message=f"Reservation #{reservation_id} rejected for {reservation['first_name']} {reservation['last_name']}",
    )


# ========================
# PARKING AVAILABILITY & PRICING ENDPOINTS
# ========================

# Master config: parking types with INR pricing and feature metadata.
# Prices mirror initialize_default_data() in sql_store.py.
_PARKING_TYPE_META = {
    "standard": {
        "name": "Standard Parking",
        "description": "Hatchbacks, sedans & compact SUVs",
        "hourly_price": 50,
        "daily_price": 350,
        "monthly_price": 4500,
        "features": ["CCTV monitored", "Covered parking", "Elevator access"],
    },
    "large": {
        "name": "Large Vehicle",
        "description": "SUVs, pickup trucks & vans",
        "hourly_price": 80,
        "daily_price": 550,
        "monthly_price": 7000,
        "features": ["Extra-wide lanes", "High roof clearance", "Floor P4"],
    },
    "ev": {
        "name": "EV Charging",
        "description": "Level 2 + fast charging included",
        "hourly_price": 120,
        "daily_price": 800,
        "monthly_price": 9500,
        "features": ["Free charging", "Fast charge", "24/7 access"],
    },
    "vip": {
        "name": "VIP Premium",
        "description": "Closest to exit, dedicated valet",
        "hourly_price": 200,
        "daily_price": 1500,
        "monthly_price": 18000,
        "features": ["Covered premium area", "Priority access", "Valet support"],
    },
    "disabled": {
        "name": "Disabled",
        "description": "Wheelchair-accessible near elevators",
        "hourly_price": 30,
        "daily_price": 120,
        "monthly_price": 1200,
        "features": ["Wheelchair access", "Extra-wide", "Elevator priority"],
    },
    "bike": {
        "name": "Bike / 2-Wheeler",
        "description": "Bikes, scooters & electric two-wheelers",
        "hourly_price": 20,
        "daily_price": 120,
        "monthly_price": 1200,
        "features": ["Covered area", "EV bike charging", "Helmet lockers"],
    },
}


@app.get("/api/parking/availability")
def get_parking_availability():
    """
    Get live slot availability for all parking types.

    Returns counts per type with percentage and status label.
    Frontend polls this endpoint every 30 s for real-time updates.
    """
    import datetime as _dt
    import logging

    _log = logging.getLogger(__name__)
    types = _parking_svc.get_all_types()
    _log.info("[availability] DB query returned %d parking types", len(types))

    result = []
    for t in types:
        available = t["available_slots"]
        total = t["total_slots"]
        pct = t["availability_percentage"]
        status = t["availability_status"]
        _log.info(
            "[availability]   %-10s  avail=%d  total=%d  pct=%d%%  status=%s",
            t["slug"], available, total, pct, status,
        )
        result.append(
            {
                "space_type": t["slug"],
                "available": available,
                "total": total,
                "percentage": pct,
                "status": status,
            }
        )
    return {"availability": result, "timestamp": _dt.datetime.utcnow().isoformat()}


@app.get("/api/parking/types")
def get_parking_types():
    """
    Get all parking types with INR pricing and live availability.

    Reads from the production `parking_types` table (seeded by
    `scripts/seed_parking_data.py`).  Returns an empty list when the
    table is unpopulated — run the seed script to fix this.

    Used by the frontend ParkingCards component.
    """
    import logging

    _log = logging.getLogger(__name__)
    types = _parking_svc.get_all_types()
    _log.info("[parking/types] returning %d types from production DB", len(types))
    for t in types:
        _log.debug(
            "[parking/types]   %-10s  avail=%d/%d",
            t["slug"], t["available_slots"], t["total_slots"],
        )
    if not types:
        _log.warning(
            "[parking/types] No parking types found — "
            "run: python scripts/seed_parking_data.py"
        )
    return {"types": types}


class PriceCalculateRequest(BaseModel):
    """Body for dynamic price calculation."""

    space_type: str
    start_datetime: str
    end_datetime: str


@app.post("/api/parking/calculate-price")
def calculate_price_endpoint(request: PriceCalculateRequest):
    """
    Calculate total INR cost for a booking before confirmation.

    Tries the production service first; falls back to the legacy SQL store
    so that chatbot legacy types (standard / large / ev / vip) always work.
    """
    _log.info(
        "[pricing] Request: type=%s  start=%s  end=%s",
        request.space_type, request.start_datetime, request.end_datetime,
    )
    # ── Try production service (parking_types table) ──────────────────────
    try:
        bd = _parking_svc.calculate_price(
            request.space_type, request.start_datetime, request.end_datetime
        )
        result = {
            "space_type": request.space_type,
            "total_inr": bd["total_inr"],
            "duration_label": bd["duration_label"],
            "unit_price": bd["unit_price"],
            "duration_hours": bd["duration_hours"],
            "currency": "INR",
            "formatted": bd["formatted"],
        }
        _log.info("[pricing] Calculated total (production): %s", result["formatted"])
        return result
    except Exception as exc:
        _log.debug("[pricing] Production service unavailable (%s), trying legacy store", exc)

    # ── Fallback: legacy SQL store (parking_prices table) ─────────────────
    try:
        total, label, unit = sql_store.calculate_price(
            request.space_type, request.start_datetime, request.end_datetime
        )
        # Compute duration_hours from the request strings
        from datetime import datetime as _dt
        _fmt = "%Y-%m-%d %H:%M"
        _s = _dt.strptime(request.start_datetime[:16].replace("T", " "), _fmt)
        _e = _dt.strptime(request.end_datetime[:16].replace("T", " "), _fmt)
        hours = round((_e - _s).total_seconds() / 3600, 2)
        result = {
            "space_type": request.space_type,
            "total_inr": total,
            "duration_label": label,
            "unit_price": unit,
            "duration_hours": hours,
            "currency": "INR",
            "formatted": f"\u20b9{total:,.0f}",
        }
        _log.info("[pricing] Calculated total (legacy): %s", result["formatted"])
        return result
    except Exception as exc:
        _log.error("[pricing] Both services failed for type=%s: %s", request.space_type, exc)
        raise HTTPException(status_code=400, detail=f"Price calculation failed: {exc}")


# POST /api/pricing/calculate — alternate spec-aligned endpoint
class PricingCalculateRequest(BaseModel):
    """Body for POST /api/pricing/calculate."""
    parking_type: str = Field(..., json_schema_extra={"example": "vip"})
    start_time: str = Field(..., json_schema_extra={"example": "2026-06-01 09:00"})
    end_time: str = Field(..., json_schema_extra={"example": "2026-06-01 18:00"})


@app.post("/api/pricing/calculate", summary="Calculate booking price")
def pricing_calculate(request: PricingCalculateRequest):
    """
    Calculate booking cost.

    Request:  { parking_type, start_time, end_time }
    Response: { duration_hours, pricing_model, rate, estimated_total, ... }

    Automatically picks the cheapest valid pricing tier:
      < 24 h  → hourly   |   1-29 days  → daily   |   ≥ 30 days  → monthly
    """
    try:
        bd = _parking_svc.calculate_price(
            request.parking_type, request.start_time, request.end_time
        )
        label = bd["duration_label"]
        model = "monthly" if "month" in label else ("daily" if "day" in label else "hourly")
        return {
            "parking_type": request.parking_type,
            "duration_hours": bd["duration_hours"],
            "pricing_model": model,
            "rate": bd["unit_price"],
            "estimated_total": bd["total_inr"],
            "formatted": bd["formatted"],
            "duration_label": label,
            "currency": "INR",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Price calculation failed: {exc}")


@app.post("/api/parking/check-availability")
def check_availability_endpoint(space_type: str):
    """
    Check if slots are available for a specific parking type.
    Returns availability details and alternative suggestions when full.
    """
    return _parking_svc.check_availability(space_type)


@app.get("/api/debug/database-status")
def debug_database_status():
    """
    Admin debug endpoint — returns live DB diagnostics.

    Reports the active database URL/path, table row counts,
    and available slot totals.  Used to verify seeded data is
    visible to the backend.
    """
    import logging

    _log = logging.getLogger(__name__)
    info = get_db_info()

    try:
        from src.models.booking import Booking
        from src.models.parking_slot import ParkingSlot
        from src.models.parking_type import ParkingType

        with db_session() as db:
            type_count = db.query(ParkingType).count()
            slot_count = db.query(ParkingSlot).count()
            available_slots = (
                db.query(ParkingType.available_slots)
                .filter(ParkingType.is_active.is_(True))
                .all()
            )
            total_available = sum(r[0] for r in available_slots if r[0])
            booking_count = db.query(Booking).count()

        result = {
            "database_url": info["database_url"],
            "absolute_path": info["absolute_path"],
            "driver": info["driver"],
            "parking_type_count": type_count,
            "total_slot_count": slot_count,
            "available_slots_total": total_available,
            "booking_count": booking_count,
            "seed_needed": type_count == 0,
        }
        _log.info("[debug/db-status] %s", result)
        return result
    except Exception as exc:
        _log.error("[debug/db-status] query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"DB status query failed: {exc}")


@app.post("/api/admin/sync-counters")
def admin_sync_counters():
    """
    Admin: manually trigger a slot-counter resync.
    Recomputes parking_type.available_slots from actual parking_slot statuses.
    """
    _sync_slot_counters()
    return {"success": True, "message": "Slot counters synced from actual slot statuses"}


@app.get("/api/parking/slot-availability/{slug}")
def get_slot_availability(slug: str):
    """
    Real-time availability for a single parking type.
    Counts directly from parking_slots table (authoritative) rather than
    relying on the counter column, so the result is always accurate.
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.models.parking_type import ParkingType as _PT
    from sqlalchemy import case, func, select

    with _dbs() as db:
        pt = db.execute(select(_PT).where(_PT.slug == slug)).scalar_one_or_none()
        if not pt:
            raise HTTPException(status_code=404, detail=f"Parking type '{slug}' not found")

        counts = db.execute(
            select(
                func.count().label("total"),
                func.sum(case((_PS.status == "available", 1), else_=0)).label("available"),
                func.sum(case((_PS.status == "reserved", 1), else_=0)).label("reserved"),
                func.sum(case((_PS.status == "occupied", 1), else_=0)).label("occupied"),
                func.sum(case((_PS.status == "maintenance", 1), else_=0)).label("maintenance"),
            ).where(_PS.parking_type_id == pt.id)
        ).first()

        available = counts.available or 0
        reserved = counts.reserved or 0
        occupied = counts.occupied or 0
        total = counts.total or 0
        # "effective" available = only truly free slots (not reserved by pending bookings)
        pct = round((available / total) * 100) if total else 0

    return {
        "slug": slug,
        "name": pt.name,
        "total": total,
        "available": available,
        "reserved_pending": reserved,
        "occupied": occupied,
        "maintenance": counts.maintenance or 0,
        "effective_available": available,  # = slots with status 'available'
        "availability_percentage": pct,
        "status": "full" if available == 0 else ("limited" if pct < 20 else "available"),
        "is_bookable": available > 0 and pt.is_active,
    }


# ========================
# ADMIN SLOT MANAGEMENT APIs
# ========================

class ParkingTypeUpdateRequest(BaseModel):
    """Body for PUT /api/admin/parking-types/{slug}."""
    name: Optional[str] = None
    description: Optional[str] = None
    hourly_price: Optional[float] = None
    daily_price: Optional[float] = None
    monthly_price: Optional[float] = None
    features: Optional[list] = None
    is_active: Optional[bool] = None


class SlotsAddRequest(BaseModel):
    """Body for POST /api/admin/parking-types/{slug}/slots/add."""
    count: int = Field(..., ge=1, le=500, description="Number of slots to add")


class SlotsRemoveRequest(BaseModel):
    """Body for POST /api/admin/parking-types/{slug}/slots/remove."""
    count: int = Field(..., ge=1, le=500, description="Number of available slots to remove")


@app.get("/api/admin/parking-types")
def admin_list_parking_types():
    """
    Admin: list all parking types with full slot stats.
    Includes total/available/occupied/maintenance counts per type.
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.models.parking_type import ParkingType as _PT
    from sqlalchemy import select, func, case

    with _dbs() as db:
        rows = db.execute(select(_PT).order_by(_PT.id)).scalars().all()
        result = []
        for pt in rows:
            # Count slots by status for this type
            counts = db.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((_PS.status == "available", 1), else_=0)).label("available"),
                    func.sum(case((_PS.status == "occupied", 1), else_=0)).label("occupied"),
                    func.sum(case((_PS.status == "maintenance", 1), else_=0)).label("maintenance"),
                ).where(_PS.parking_type_id == pt.id)
            ).first()
            result.append({
                "id": pt.id,
                "slug": pt.slug,
                "name": pt.name,
                "description": pt.description,
                "hourly_price": float(pt.hourly_price) if pt.hourly_price else 0,
                "daily_price": float(pt.daily_price) if pt.daily_price else 0,
                "monthly_price": float(pt.monthly_price) if pt.monthly_price else 0,
                "features": pt.features or [],
                "is_active": pt.is_active,
                "total_slots": counts.total or 0,
                "available_slots": counts.available or 0,
                "occupied_slots": counts.occupied or 0,
                "maintenance_slots": counts.maintenance or 0,
                "availability_status": pt.availability_status,
                "availability_percentage": pt.availability_percentage,
            })
    return {"parking_types": result}


@app.put("/api/admin/parking-types/{slug}")
def admin_update_parking_type(slug: str, req: ParkingTypeUpdateRequest):
    """
    Admin: update pricing, description, features, or active flag for a parking type.
    Does NOT change slot counts — use the /slots/add or /slots/remove endpoints for that.
    """
    from src.database.session import db_session as _dbs
    from src.repositories.parking_repository import ParkingRepository as _PR

    repo = _PR()
    with _dbs() as db:
        pt = repo.get_by_slug(db, slug)
        if not pt:
            raise HTTPException(status_code=404, detail=f"Parking type '{slug}' not found")
        update_fields = req.model_dump(exclude_none=True)
        for k, v in update_fields.items():
            setattr(pt, k, v)
        db.flush()
    return {"success": True, "message": f"Parking type '{slug}' updated"}


@app.post("/api/admin/parking-types/{slug}/slots/add")
def admin_add_slots(slug: str, req: SlotsAddRequest):
    """
    Admin: add N new parking slots to a type.
    Slots are auto-numbered (prefix-NNN) and marked AVAILABLE.
    The parking_type.available_slots counter is incremented atomically.
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.models.parking_type import ParkingType as _PT
    from src.repositories.parking_repository import ParkingRepository as _PR
    from sqlalchemy import select, func

    repo = _PR()
    with _dbs() as db:
        pt = repo.get_by_slug(db, slug)
        if not pt:
            raise HTTPException(status_code=404, detail=f"Parking type '{slug}' not found")

        # Determine prefix from existing slots or fall back to slug[:3].upper()
        existing = db.execute(
            select(_PS.slot_number)
            .where(_PS.parking_type_id == pt.id)
            .order_by(_PS.id.desc())
            .limit(1)
        ).scalar_one_or_none()

        if existing:
            prefix = existing.rsplit("-", 1)[0]  # e.g. "VIP" from "VIP-010"
            last_num = int(existing.rsplit("-", 1)[1])
        else:
            prefix = slug[:3].upper()
            last_num = 0

        new_slots = []
        for i in range(req.count):
            num = last_num + i + 1
            new_slots.append(_PS(
                slot_number=f"{prefix}-{num:03d}",
                parking_type_id=pt.id,
                floor_number=getattr(pt, "floor_number", "Ground") or "Ground",
                status="available",
            ))
        db.add_all(new_slots)

        # Sync counters
        pt.total_slots = pt.total_slots + req.count
        pt.available_slots = pt.available_slots + req.count
        db.flush()

    _log.info(
        "[ADMIN] Added %d slots to %s  (new total: %d)",
        req.count, slug.upper(), pt.total_slots,
    )
    return {"success": True, "added": req.count, "new_total": pt.total_slots}


@app.post("/api/admin/parking-types/{slug}/slots/remove")
def admin_remove_slots(slug: str, req: SlotsRemoveRequest):
    """
    Admin: logically remove N AVAILABLE slots from a type.
    Only removes slots whose status is 'available' (never occupied/reserved).
    Updates total_slots and available_slots counters.
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.repositories.parking_repository import ParkingRepository as _PR
    from sqlalchemy import select

    repo = _PR()
    with _dbs() as db:
        pt = repo.get_by_slug(db, slug)
        if not pt:
            raise HTTPException(status_code=404, detail=f"Parking type '{slug}' not found")

        avail_slots = db.execute(
            select(_PS)
            .where(_PS.parking_type_id == pt.id)
            .where(_PS.status == "available")
            .order_by(_PS.id.desc())
            .limit(req.count)
        ).scalars().all()

        if not avail_slots:
            raise HTTPException(status_code=409, detail="No available slots to remove")

        actually_removed = len(avail_slots)
        for slot in avail_slots:
            db.delete(slot)

        pt.total_slots = max(0, pt.total_slots - actually_removed)
        pt.available_slots = max(0, pt.available_slots - actually_removed)
        db.flush()

    _log.info(
        "[ADMIN] Removed %d slots from %s  (new total: %d)",
        actually_removed, slug.upper(), pt.total_slots,
    )
    return {"success": True, "removed": actually_removed, "new_total": pt.total_slots}


@app.get("/api/admin/dashboard")
def admin_dashboard():
    """
    Admin: aggregate live dashboard stats.
    Returns parking type inventory + reservation counts in one call.
    Polled every 10 s by the admin frontend.
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.models.parking_type import ParkingType as _PT
    from sqlalchemy import select, func, case

    with _dbs() as db:
        # Per-type slot stats (include reserved/pending slots in the breakdown)
        type_rows = db.execute(select(_PT).order_by(_PT.id)).scalars().all()
        slot_stats = []
        total_available = 0
        total_occupied = 0
        total_reserved = 0
        total_slots = 0
        for pt in type_rows:
            c = db.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((_PS.status == "available", 1), else_=0)).label("available"),
                    func.sum(case((_PS.status == "occupied", 1), else_=0)).label("occupied"),
                    func.sum(case((_PS.status == "reserved", 1), else_=0)).label("reserved"),
                ).where(_PS.parking_type_id == pt.id)
            ).first()
            t, a, o, r = (c.total or 0), (c.available or 0), (c.occupied or 0), (c.reserved or 0)
            total_slots += pt.total_slots   # use the authoritative counter, not physical count
            total_available += a
            total_occupied += o
            total_reserved += r
            slot_stats.append({
                "slug": pt.slug,
                "name": pt.name,
                "total": pt.total_slots,
                "available": a,
                "occupied": o,
                "reserved_pending": r,
                "hourly_price": float(pt.hourly_price) if pt.hourly_price else 0,
                "is_active": pt.is_active,
            })

        # Reservation counts (legacy table)
        res_all = sql_store.get_reservations()
        pending = sum(1 for r in res_all if r["status"] == "pending")
        approved = sum(1 for r in res_all if r["status"] == "approved")
        rejected = sum(1 for r in res_all if r["status"] == "rejected")
        paid = sum(1 for r in res_all if r["status"] == "paid")

    return {
        "slot_stats": slot_stats,
        "totals": {
            "total_slots": total_slots,
            "available_slots": total_available,
            "occupied_slots": total_occupied,
            "pending_slots": total_reserved,
        },
        "reservations": {
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "paid": paid,
            "total": len(res_all),
        },
    }


# ── GET /api/admin/slot-stats ────────────────────────────────────────────────
@app.get("/api/admin/slot-stats")
def admin_slot_stats():
    """
    Admin: granular real-time slot statistics.

    Returns per-type breakdown of total / available / reserved(pending) /
    occupied / maintenance counts, along with system-wide totals.

    Availability rule:
      AVAILABLE = total_slots − occupied − pending(reserved)
    """
    from src.database.session import db_session as _dbs
    from src.models.parking_slot import ParkingSlot as _PS
    from src.models.parking_type import ParkingType as _PT
    from sqlalchemy import select, func, case

    with _dbs() as db:
        types = db.execute(select(_PT).order_by(_PT.id)).scalars().all()
        stats = []
        totals = {"total": 0, "available": 0, "reserved": 0, "occupied": 0, "maintenance": 0}

        for pt in types:
            c = db.execute(
                select(
                    func.sum(case((_PS.status == "available", 1), else_=0)).label("available"),
                    func.sum(case((_PS.status == "reserved", 1), else_=0)).label("reserved"),
                    func.sum(case((_PS.status == "occupied", 1), else_=0)).label("occupied"),
                    func.sum(case((_PS.status == "maintenance", 1), else_=0)).label("maintenance"),
                ).where(_PS.parking_type_id == pt.id)
            ).first()

            entry = {
                "slug": pt.slug,
                "name": pt.name,
                "is_active": pt.is_active,
                "total_slots": pt.total_slots,
                "available": c.available or 0,
                "reserved_pending": c.reserved or 0,
                "occupied": c.occupied or 0,
                "maintenance": c.maintenance or 0,
                # available_slots counter (may differ by 1 during a transaction)
                "counter_available": pt.available_slots,
                "availability_pct": (
                    round(((c.available or 0) / pt.total_slots) * 100)
                    if pt.total_slots else 0
                ),
                "is_bookable": (c.available or 0) > 0 and pt.is_active,
            }
            stats.append(entry)
            totals["total"] += pt.total_slots
            totals["available"] += c.available or 0
            totals["reserved"] += c.reserved or 0
            totals["occupied"] += c.occupied or 0
            totals["maintenance"] += c.maintenance or 0

    return {"slot_stats": stats, "totals": totals}


# ── POST /api/admin/update-capacity ─────────────────────────────────────────
class UpdateCapacityRequest(BaseModel):
    """Body for POST /api/admin/update-capacity."""
    slug: str = Field(..., json_schema_extra={"example": "vip"})
    add_slots: Optional[int] = Field(None, ge=0, description="Add N available slots")
    remove_slots: Optional[int] = Field(None, ge=0, description="Remove N available slots")
    set_active: Optional[bool] = Field(None, description="Enable or disable this type")


@app.post("/api/admin/update-capacity")
def admin_update_capacity(req: UpdateCapacityRequest):
    """
    Admin: convenience endpoint to adjust capacity in a single call.

    Supports adding slots, removing slots, and toggling active status.
    Equivalent to calling the individual /slots/add, /slots/remove, and
    PUT /parking-types/{slug} endpoints separately.
    """
    results = {}

    if req.add_slots and req.add_slots > 0:
        r = admin_add_slots(req.slug, SlotsAddRequest(count=req.add_slots))
        results["added"] = r

    if req.remove_slots and req.remove_slots > 0:
        r = admin_remove_slots(req.slug, SlotsRemoveRequest(count=req.remove_slots))
        results["removed"] = r

    if req.set_active is not None:
        r = admin_update_parking_type(req.slug, ParkingTypeUpdateRequest(is_active=req.set_active))
        results["active_updated"] = r

    if not results:
        raise HTTPException(status_code=400, detail="No changes requested")

    return {"success": True, "slug": req.slug, "changes": results}


# ========================
# POST ADMIN ROUTES
# ========================


@app.post("/admin/approve/{reservation_id}", response_model=StatusResponse)
async def admin_approve_reservation(reservation_id: int, request: AdminActionRequest = None):
    """
    Admin approves a pending reservation (POST).

    Updates the status to 'approved', triggers email notification,
    and writes to MCP file.
    """
    import logging

    logger = logging.getLogger(__name__)

    reservation = sql_store.get_reservation_by_id(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation #{reservation_id} not found")

    if reservation["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Reservation #{reservation_id} is already {reservation['status']}")

    admin_notes = request.admin_notes if request else None
    success = sql_store.update_reservation_status(reservation_id, "approved", admin_notes)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update reservation")

    # Transition pre-reserved slot: reserved → occupied.
    # Counter was ALREADY decremented when the booking was created (pending stage).
    # Never decrement again here — doing so would double-count.
    slot_number = sql_store.occupy_reserved_slot(reservation_id)

    # Legacy fallback: for reservations created before pre-reservation was added,
    # no physical slot exists so we still need to decrement the counter.
    if slot_number is None:
        try:
            from src.database.session import db_session as _dbs_a
            from src.repositories.parking_repository import ParkingRepository as _PR_a
            with _dbs_a() as _db_a:
                _pt_a = _PR_a().get_by_slug(_db_a, reservation["space_type"])
                if _pt_a and _pt_a.available_slots > 0:
                    _pt_a.available_slots = _pt_a.available_slots - 1
        except Exception as e:
            logger.warning("Legacy counter fallback failed for #%d: %s", reservation_id, e)

    logger.info(
        "[APPROVED] Reservation #%d approved | user=%s  slot=%s",
        reservation_id,
        mask_email(reservation.get("email", "")),
        slot_number or "(legacy-fallback)",
    )

    # Notify user via email (async, non-blocking)
    try:
        updated = sql_store.get_reservation_by_id(reservation_id)

        # Create payment record and build payment link
        _pay_token: str = ""
        try:
            from decimal import Decimal
            try:
                bd = _parking_svc.calculate_price(
                    reservation["space_type"],
                    reservation["start_datetime"],
                    reservation["end_datetime"],
                )
                _amount = bd["total_inr"]
            except Exception:
                _amount, _, _ = sql_store.calculate_price(
                    reservation["space_type"],
                    reservation["start_datetime"],
                    reservation["end_datetime"],
                )
            pd = _payment_svc.create_payment(
                reservation_id=reservation_id,
                amount_inr=Decimal(str(_amount)),
                user_name=f"{reservation['first_name']} {reservation['last_name']}",
                user_email=reservation.get("email"),
                vehicle_number=reservation.get("car_number", ""),
                space_type=reservation["space_type"],
                start_datetime=reservation["start_datetime"],
                end_datetime=reservation["end_datetime"],
            )
            _pay_token = pd["payment_token"]
        except Exception as _pe:
            logger.warning("Payment creation failed for #%d (non-fatal): %s", reservation_id, _pe)

        payment_link = f"{settings.app_base_url}/payment/{_pay_token}" if _pay_token else ""
        await email_service.send_user_approval_async(updated, payment_link=payment_link)
    except Exception as e:
        logger.warning("User approval email failed for #%d: %s", reservation_id, e)

    # Write approved reservation to MCP file
    try:
        from src.mcp.mcp_client import MCPClient

        mcp_client = MCPClient()
        mcp_client.write_reservation_to_file(
            name=f"{reservation['first_name']} {reservation['last_name']}",
            car_number=reservation["car_number"],
            reservation_period=(f"{reservation['start_datetime']} - {reservation['end_datetime']}"),
        )
    except Exception as e:
        logger.warning("MCP file write failed for #%d: %s", reservation_id, e)

    return StatusResponse(
        success=True,
        message=f"Reservation #{reservation_id} approved for {reservation['first_name']} {reservation['last_name']}",
    )


@app.post("/admin/reject/{reservation_id}", response_model=StatusResponse)
async def admin_reject_reservation(reservation_id: int, request: AdminActionRequest = None):
    """
    Admin rejects a pending reservation (POST).

    Updates the status to 'rejected' and sends rejection email to user.
    """
    import logging

    logger = logging.getLogger(__name__)

    reservation = sql_store.get_reservation_by_id(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation #{reservation_id} not found")

    if reservation["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Reservation #{reservation_id} is already {reservation['status']}")

    admin_notes = request.admin_notes if request else None
    success = sql_store.update_reservation_status(reservation_id, "rejected", admin_notes)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update reservation")

    # Release the pre-reserved slot → available and restore available_slots counter.
    released = sql_store.release_reserved_slot(reservation_id)

    # Fallback: if no physical slot was pre-reserved (legacy reservation created
    # before pre-reservation was introduced), still restore the counter.
    if not released:
        try:
            from src.models.parking_type import ParkingType as _PT2r
            with db_session() as _db2r:
                _pt2r = _db2r.query(_PT2r).filter_by(slug=reservation["space_type"]).first()
                if _pt2r and _pt2r.available_slots < _pt2r.total_slots:
                    _actual = (
                        _db2r.query(
                            __import__("src.models.parking_slot", fromlist=["ParkingSlot"]).ParkingSlot
                        )
                        .filter_by(parking_type_id=_pt2r.id, status="available")
                        .count()
                    )
                    _pt2r.available_slots = min(_pt2r.total_slots, _actual + 1)
                    logger.info(
                        "[REJECTED] Reservation #%d — no pre-reserved slot; "
                        "counter restored: %s available_slots → %d",
                        reservation_id, reservation["space_type"].upper(), _pt2r.available_slots,
                    )
        except Exception as _fe:
            logger.warning("Rejection counter fallback failed for #%d: %s", reservation_id, _fe)

    logger.info(
        "[REJECTED] Reservation #%d rejected | user=%s  slot_released=%s",
        reservation_id,
        mask_email(reservation.get("email", "")),
        released,
    )

    # Notify user via email (async, non-blocking)
    try:
        updated = sql_store.get_reservation_by_id(reservation_id)
        await email_service.send_user_rejection_async(updated)
    except Exception as e:
        logger.warning("User rejection email failed for #%d: %s", reservation_id, e)

    return StatusResponse(
        success=True,
        message=f"Reservation #{reservation_id} rejected for {reservation['first_name']} {reservation['last_name']}",
    )


# ========================
# PRODUCTION BOOKING APIs (PostgreSQL-backed)
# ========================


class BookingCreateRequest(BaseModel):
    """Request body for POST /api/bookings."""

    parking_type: str = Field(..., json_schema_extra={"example": "standard"})
    user_name: str = Field(..., json_schema_extra={"example": "Sanyam Sachan"})
    email: Optional[str] = Field(None, json_schema_extra={"example": "user@example.com"})
    vehicle_number: str = Field(..., json_schema_extra={"example": "TS09AB1234"})
    vehicle_type: Optional[str] = Field(None, json_schema_extra={"example": "car"})
    start_time: str = Field(..., json_schema_extra={"example": "2026-06-01 09:00"})
    end_time: str = Field(..., json_schema_extra={"example": "2026-06-01 18:00"})


class BookingAdminRequest(BaseModel):
    """Request body for admin approve/reject booking endpoints."""

    admin_notes: Optional[str] = None
    admin_email: Optional[str] = None


class CheckAvailabilityRequest(BaseModel):
    space_type: str


@app.post("/api/bookings", summary="Create a new booking (production)")
def create_booking(request: BookingCreateRequest):
    """
    Create a new production booking.

    Validates availability and business rules, calculates the INR price,
    and persists the booking as 'pending'.
    """
    from datetime import datetime

    try:
        fmt_options = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]

        def _parse(s: str):
            for fmt in fmt_options:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime: {s!r}")

        start = _parse(request.start_time)
        end = _parse(request.end_time)

        result = _booking_svc.create(
            parking_type_slug=request.parking_type,
            user_name=request.user_name,
            email=request.email,
            vehicle_number=request.vehicle_number,
            start_time=start,
            end_time=end,
            vehicle_type=request.vehicle_type,
        )
        return {"success": True, "booking": result}
    except BookingValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "alternatives": exc.alternatives},
        )
    except BookingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Booking creation failed: {exc}")


@app.get("/api/bookings", summary="List bookings (production)")
def list_bookings(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all production bookings with optional status filter."""
    try:
        bookings = _booking_svc.list(status=status, limit=limit, offset=offset)
        return {"bookings": bookings, "total": len(bookings)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/bookings/by-email", summary="Get booking history for an email")
def bookings_by_email(email: str = Query(...)):
    """Return all bookings associated with an email address."""
    try:
        bookings = _booking_svc.list_by_email(email)
        return {"bookings": bookings, "email": email}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/bookings/reference/{reference}", summary="Get booking by reference code")
def get_booking_by_reference(reference: str):
    """Get a production booking by its human-readable reference (e.g. PS-A3X8K)."""
    result = _booking_svc.get_by_reference(reference)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Booking '{reference}' not found")
    return result


@app.get("/api/bookings/{booking_id}", summary="Get a single booking")
def get_booking(booking_id: int):
    """Get a production booking by numeric ID."""
    result = _booking_svc.get(booking_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Booking #{booking_id} not found")
    return result


@app.post("/api/bookings/{booking_id}/approve", summary="Admin: approve a booking")
def approve_booking(booking_id: int, request: BookingAdminRequest = None):
    """
    Approve a pending production booking.

    Atomically:
      1. Allocates a physical parking slot
      2. Decrements parking_type.available_slots
      3. Updates booking status to 'approved'
      4. Logs AdminAction audit record
    """
    try:
        notes = request.admin_notes if request else None
        admin_email = request.admin_email if request else None
        result = _booking_svc.approve(booking_id, admin_email=admin_email, admin_notes=notes)
        return {"success": True, "booking": result}
    except BookingValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "alternatives": exc.alternatives},
        )
    except BookingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/bookings/{booking_id}/reject", summary="Admin: reject a booking")
def reject_booking(booking_id: int, request: BookingAdminRequest = None):
    """Reject a pending production booking (no slot allocated)."""
    try:
        notes = request.admin_notes if request else None
        admin_email = request.admin_email if request else None
        result = _booking_svc.reject(booking_id, admin_email=admin_email, admin_notes=notes)
        return {"success": True, "booking": result}
    except BookingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/bookings/{booking_id}/cancel", summary="Cancel a booking")
def cancel_booking(booking_id: int, reason: Optional[str] = Query(None)):
    """
    Cancel an approved or pending booking.

    Releases the assigned slot and restores availability.
    """
    try:
        result = _booking_svc.cancel(booking_id, reason=reason)
        return {"success": True, "booking": result}
    except BookingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/parking/live-status", summary="Live availability with colour indicators")
def get_live_status():
    """
    Return parking type availability with colour-coded status.

    Adds `status_colour` = "green" | "yellow" | "red" per type.
    """
    try:
        return {"types": _parking_svc.get_live_status()}
    except Exception:
        # Gracefully fall back to legacy sql_store data
        summary = sql_store.get_total_availability()
        types = []
        for slug, counts in summary.items():
            pct = round((counts["available"] / counts["total"]) * 100) if counts["total"] else 0
            types.append({
                "slug": slug,
                "available_slots": counts["available"],
                "total_slots": counts["total"],
                "availability_percentage": pct,
                "status_colour": "green" if pct >= 50 else ("yellow" if pct > 0 else "red"),
            })
        return {"types": types, "source": "legacy"}


@app.post("/api/bookings/check-availability", summary="Check availability + alternatives")
def check_booking_availability(request: CheckAvailabilityRequest):
    """
    Check if slots are available for a parking type.

    Returns alternatives when the requested type is full.
    """
    try:
        result = _parking_svc.check_availability(request.space_type)
        return result
    except Exception:
        # Fall back to sql_store
        avail = sql_store.check_availability(request.space_type)
        if not avail["is_available"]:
            summary = sql_store.get_total_availability()
            avail["alternatives"] = [
                {"space_type": k, "available": v["available"]}
                for k, v in summary.items()
                if k != request.space_type and v["available"] > 0
            ]
        return avail


# ========================
# PAYMENT ENDPOINTS
# ========================


class PaymentProcessRequest(BaseModel):
    """Body for POST /api/payment/{token}/process."""

    payment_method: str = Field(
        ...,
        json_schema_extra={"example": "upi"},
        description="One of: upi, card, netbanking, wallet",
    )
    # UPI
    upi_id: Optional[str] = Field(None, json_schema_extra={"example": "user@upi"})
    # Card
    card_last4: Optional[str] = Field(None, json_schema_extra={"example": "4242"})
    card_holder: Optional[str] = Field(None, json_schema_extra={"example": "John Smith"})
    # Wallet
    wallet_provider: Optional[str] = Field(None, json_schema_extra={"example": "paytm"})


@app.get("/api/payment/{token}", summary="Get payment details by token")
def get_payment_by_token(token: str):
    """
    Return booking details and payment status for the payment page.

    Called by the frontend /payment/[token] page on load to populate the
    booking summary and determine whether the user can still pay.

    Returns 404 if the token is not recognised.
    """
    payment = _payment_svc.get_payment(token)
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment link not found or has expired.")
    return payment


@app.post("/api/payment/{token}/process", summary="Process a payment")
def process_payment(token: str, request: PaymentProcessRequest):
    """
    Process a payment via the mock gateway.

    - Validates the token, checks idempotency and expiry.
    - Marks the payment as 'paid' and stores the transaction ID.
    - Updates the reservation status to 'paid' in the legacy table.
    - Sends a payment-confirmation email to the admin.

    Returns the updated payment dict including transaction_id and paid_at.

    Error codes:
      400 — invalid payment method
      404 — token not found
      409 — already paid (idempotency guard)
      410 — payment link expired
    """
    try:
        payment = _payment_svc.process_payment(
            token=token,
            payment_method=request.payment_method,
            upi_id=request.upi_id,
            card_last4=request.card_last4,
            wallet_provider=request.wallet_provider,
        )
    except PaymentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except AlreadyPaidError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except PaymentExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    except PaymentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Update the legacy reservation status to 'paid'
    try:
        sql_store.update_reservation_status(
            payment["reservation_id"], "paid", "Payment completed online"
        )
        _log.info("[payment] Reservation #%d status → paid", payment["reservation_id"])
    except Exception as e:
        _log.warning("[payment] Could not update legacy reservation status: %s", e)

    # Notify admin about the payment
    try:
        email_service.send_payment_confirmation_to_admin(payment)
    except Exception as e:
        _log.warning("[payment] Admin payment notification failed (non-fatal): %s", e)

    return payment


@app.get("/api/payment/{token}/status", summary="Poll payment status")
def get_payment_status(token: str):
    """
    Lightweight status-only poll for the frontend to check after redirect.

    Returns { status, transaction_id, paid_at } without full booking details.
    """
    payment = _payment_svc.get_payment(token)
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment link not found.")
    return {
        "token": token,
        "status": payment["status"],
        "transaction_id": payment.get("transaction_id"),
        "paid_at": payment.get("paid_at"),
        "amount_inr": payment.get("amount_inr"),
        "is_payable": payment.get("is_payable", False),
        "is_expired": payment.get("is_expired", False),
    }

