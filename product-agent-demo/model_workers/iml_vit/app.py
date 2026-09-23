from __future__ import annotations

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


ROOT = Path(os.getenv("IML_VIT_REPO", "~/the-louvre-imlvit/IML-ViT")).expanduser()
CHECKPOINT = Path(os.getenv("IML_VIT_CHECKPOINT", str(ROOT / "checkpoints/iml-vit_checkpoint.pth"))).expanduser()
INPUT_SIZE = int(os.getenv("IML_VIT_INPUT_SIZE", "1024"))
MODEL_VERSION = os.getenv("IML_VIT_MODEL_VERSION", "iml-vit-v1")
DEVICE = torch.device(os.getenv("IML_VIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MODEL = None


def _load_model():
    global MODEL
    sys.path.insert(0, str(ROOT))
    from iml_vit_model import iml_vit_model

    model = iml_vit_model(input_size=INPUT_SIZE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=True)
    MODEL = model.to(DEVICE).eval()


def _prepare(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int, int, int]]:
    original_width, original_height = image.size
    scale = min(1.0, INPUT_SIZE / max(original_width, original_height))
    work_width = max(1, round(original_width * scale))
    work_height = max(1, round(original_height * scale))
    if (work_width, work_height) != image.size:
        image = image.resize((work_width, work_height), Image.Resampling.BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    canvas[:work_height, :work_width] = array
    canvas = (canvas - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor, (original_width, original_height), (work_width, work_height, 0, 0)


def _encode_mask(mask: np.ndarray) -> str:
    clipped = np.clip(mask, 0.0, 1.0)
    image = Image.fromarray(np.round(clipped * 65535.0).astype(np.uint16), mode="I;16")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="IML-ViT Worker", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready" if MODEL is not None else "unavailable", "model_version": MODEL_VERSION, "device": str(DEVICE)}


@app.post("/internal/v1/predict")
async def predict(image: UploadFile = File(...), request_id: str = Form(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "model is not loaded"})
    started = time.perf_counter()
    try:
        data = await image.read()
        with Image.open(io.BytesIO(data)) as source:
            work, original_size, transform = _prepare(source)
        masks = torch.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), device=DEVICE)
        edge_masks = torch.ones_like(masks)
        with torch.inference_mode():
            _, prediction, _ = MODEL(work, masks, edge_masks)
        output = prediction[0, 0].detach().float().cpu().numpy()
        work_width, work_height, _, _ = transform
        output = output[:work_height, :work_width]
        output = np.asarray(Image.fromarray(output).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        return {
            "request_id": request_id,
            "status": "ok",
            "model_version": MODEL_VERSION,
            "mask_encoding": "png_gray16_base64",
            "mask_width": original_size[0],
            "mask_height": original_size[1],
            "probability_mask": _encode_mask(output),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {"request_id": request_id, "status": "error", "model_version": MODEL_VERSION, "error": {"code": "inference_failed", "message": str(exc)[:180], "retryable": True}}
