from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_client import BailianVisionClient
from .config import ModelConfig
from .lab import router as lab_router
from .agent3.router import router as agent3_router

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="L'Oréal Product Trust Agent", version="0.1.0")
app.include_router(lab_router)
app.include_router(agent3_router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "lab.html")


class ModelTestRequest(BaseModel):
    config_payload: dict = Field(default_factory=dict, alias="model_config")
    prompt: str
    system: str = ""
    model_config = {"populate_by_name": True}


@app.post("/api/model/test")
def model_test(request: ModelTestRequest):
    config = ModelConfig.from_dict(request.config_payload)
    try:
        client = BailianVisionClient(
            api_key=config.api_key, model=config.model, base_url=config.base_url,
            timeout_seconds=config.timeout_seconds, temperature=config.temperature,
        )
        result = client.complete_text(request.prompt, request.system)
        return {"status": "ok", "model": config.model, "request_id": client.last_request_id, "text": result}
    except Exception as exc:
        return {"status": "error", "model": config.model, "error": str(exc)}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "bailian_key_configured": bool(os.getenv("DASHSCOPE_API_KEY")),
        "mcp_configured": bool(os.getenv("BAILIAN_MCP_URL")),
    }
