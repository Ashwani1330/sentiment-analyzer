# app/main.py
from fastapi import FastAPI
from app.api.routers.analyze import router as analyze_router
from app.api.routers.health import router as health_router
from app.api.routers.batch import router as batch_router

def create_app():
    app = FastAPI(title="RAG-ABSA")
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(batch_router)
    return app

app = create_app()
app.include_router(batch_router)
