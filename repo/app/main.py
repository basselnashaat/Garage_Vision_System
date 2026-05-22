"""
FastAPI application entry point.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""

    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting up...")

    try:
        from .pipeline.coordinator import LPRPipeline
        app.state.pipeline = LPRPipeline()
        logger.info("✓ Pipeline initialized")
    except Exception as e:
        logger.error(f"Pipeline failed to initialize: {e}")
        app.state.pipeline = None

    try:
        from .database.connection import Database
        from .database.logger import VisitLogger
        db                     = Database()
        app.state.db           = db
        app.state.visit_logger = VisitLogger(db)
        logger.info("✓ Supabase connected")
    except Exception as e:
        logger.error(f"Database failed to connect: {e}")
        app.state.db           = None
        app.state.visit_logger = None

    yield  # app runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    if hasattr(app.state, "db") and app.state.db:
        app.state.db.close()
    if hasattr(app.state, "pipeline") and app.state.pipeline:
        del app.state.pipeline
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vehicle Intelligence Dashboard — LPR API",
    description="License plate detection, OCR, vehicle classification, and purchasing power scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React CRA
        "http://localhost:5173",   # Vite
        "http://localhost:8080",   # your dashboard
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

from .api.routes import detect, entries, stats, analytics, leaderboard, cameras

app.include_router(detect.router,      prefix="/api")
app.include_router(entries.router,     prefix="/api")
app.include_router(stats.router,       prefix="/api")
app.include_router(analytics.router,   prefix="/api")
app.include_router(leaderboard.router, prefix="/api")
app.include_router(cameras.router,     prefix="/api")

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "Vehicle Intelligence Dashboard LPR API",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {
        "pipeline": app.state.pipeline is not None,
        "database": app.state.db is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)