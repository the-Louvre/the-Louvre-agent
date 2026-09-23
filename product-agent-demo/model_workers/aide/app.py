from __future__ import annotations

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


ROOT = Path(os.getenv("AIDE_REPO", "~/aide-deploy/AIDE-main")).expanduser()
CHECKPOINT = Path(
    os.getenv("AIDE_CHECKPOINT", "~/aide-deploy/checkpoints/GenImage_train.pth")
).expanduser()
MODEL_VERSION = os.getenv("AIDE_MODEL_VERSION", "aide-genimage-v1")
DEVICE = torch.device(os.getenv("AIDE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MODEL = None
DCT = None

TO_TENSOR = transforms.ToTensor()
RESIZE = transforms.Resize((256, 256))
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _load_model() -> None:
    """Load the official AIDE checkpoint without downloading any base weights."""
    global MODEL, DCT
    if not ROOT.is_dir():
        raise RuntimeError(f"AIDE source directory does not exist: {ROOT}")
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"AIDE checkpoint does not exist: {CHECKPOINT}")
    sys.path.insert(0, str(ROOT))
    from data.dct import DCT_base_Rec_Module
    from models.AIDE import AIDE

    model = AIDE(resnet_path=None, convnext_path=None)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("AIDE checkpoint does not match the official model definition")
    MODEL = model.to(DEVICE).eval()
    DCT = DCT_base_Rec_Module().eval()


def _prepare(image: Image.Image) -> torch.Tensor:
    """Match AIDE's deterministic TestDataset preprocessing exactly."""
    if DCT is None:
        raise RuntimeError("DCT preprocessor is not loaded")
    source = TO_TENSOR(image.convert("RGB"))
    with torch.inference_mode():
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = DCT(source)
    inputs = [x_minmin, x_maxmax, x_minmin1, x_maxmax1, source]
    return torch.stack([NORMALIZE(RESIZE(item)) for item in inputs], dim=0).unsqueeze(0).to(DEVICE)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="AIDE Worker", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if MODEL is not None else "unavailable",
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
    }


@app.post("/internal/v1/predict")
async def predict(image: UploadFile = File(...), request_id: str = Form(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "model is not loaded"})
    started = time.perf_counter()
    try:
        data = await image.read()
        with Image.open(io.BytesIO(data)) as source:
            batch = _prepare(source)
        with torch.inference_mode():
            logits = MODEL(batch)
            probabilities = torch.softmax(logits, dim=1)[0].detach().float().cpu()
        # Official TestDataset labels: 0_real -> class 0, 1_fake -> class 1.
        real_probability, ai_probability = (float(probabilities[0]), float(probabilities[1]))
        return {
            "request_id": request_id,
            "status": "ok",
            "model_version": MODEL_VERSION,
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "request_id": request_id,
            "status": "error",
            "model_version": MODEL_VERSION,
            "error": {"code": "inference_failed", "message": str(exc)[:180], "retryable": True},
        }
