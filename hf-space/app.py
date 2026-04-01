"""FastAPI backend for electrical symbol detection on Hugging Face Spaces."""

import io
import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import SymbolDetectionPredictor

# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model_epoch_400.pth")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "8"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.50"))
CATEGORIES_FILE = os.environ.get("CATEGORIES_FILE", "model/categories.json")

predictor: SymbolDetectionPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    cat_file = CATEGORIES_FILE if Path(CATEGORIES_FILE).exists() else None
    predictor = SymbolDetectionPredictor(
        checkpoint_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        categories_file=cat_file,
        device="cpu",
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    yield
    predictor = None


app = FastAPI(title="Electrical Symbol Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/detect")
async def detect(image: UploadFile = File(...), confidence: float | None = None):
    """
    Detect electrical symbols in an uploaded floor plan image.

    Returns detections and a Bill of Quantities (BOQ).
    """
    if predictor is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    # Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Run inference
    conf = confidence if confidence is not None else CONFIDENCE_THRESHOLD
    detections = predictor.predict(img, conf_threshold=conf)

    # Build BOQ from detections
    symbol_counts = Counter(d["class_name"] for d in detections)
    boq = [
        {"id": idx + 1, "symbol": name, "quantity": count, "unit": "nos"}
        for idx, (name, count) in enumerate(sorted(symbol_counts.items()))
    ]

    return {
        "detections": detections,
        "boq": boq,
        "total_detections": len(detections),
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    }
