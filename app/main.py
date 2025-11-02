# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.analyze import router as analyze_router
from app.api.routers.health import router as health_router
from app.api.routers.batch import router as batch_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG-ABSA")

    # Allow your frontend dev origins
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,      # use ["*"] only for quick local testing
        allow_credentials=True,
        allow_methods=["*"],        # includes OPTIONS for preflight
        allow_headers=["*"],        # e.g., Content-Type, Authorization
    )

    # Routers
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(batch_router)

    return app


app = create_app()
