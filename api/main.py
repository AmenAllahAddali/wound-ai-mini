"""
API Wound-AI Mini
Lancer depuis la racine du projet : uvicorn api.main:app --port 8000
Documentation interactive : http://localhost:8000/docs
"""
import os
import base64
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.predict import load_model, predict_mask, read_image_bytes, MODEL_PATH
from src.measure import measure_wound, make_overlay
from src.gradcam import gradcam_overlay

MAX_SIZE_MB = 10
state = {}
gradcam_lock = threading.Lock()  # Grad-CAM pose des "hooks" sur le modèle : une requête à la fois


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargé une seule fois au démarrage du serveur
    state["model"] = load_model()
    yield
    state.clear()


app = FastAPI(
    title="Wound-AI Mini API",
    description="Segmentation de plaies (U-Net) + mesure de surface + explicabilité (Grad-CAM). "
                "Outil de démonstration, pas un dispositif médical.",
    version="0.2.0",
    lifespan=lifespan,
)

# Autorise les appels depuis une démo web ou mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_base64_png(img):
    """Image numpy (RGB ou niveaux de gris) -> chaîne PNG en base64."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Échec de l'encodage PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.get("/")
def root():
    return {
        "name": "Wound-AI Mini API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (option ?explain=true pour le Grad-CAM)",
    }


@app.get("/health")
def health():
    return {"status": "ok", "trained_model": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="Photo de la plaie (PNG ou JPG)"),
    threshold: float = Query(0.5, ge=0.05, le=0.95, description="Seuil de probabilité plaie / fond"),
    px_per_cm: Optional[float] = Query(None, gt=0, description="Pixels pour 1 cm (optionnel, pour avoir des cm²)"),
    explain: bool = Query(False, description="Ajouter la carte Grad-CAM (plus lent)"),
):
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide")
    if len(data) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image trop lourde (max {MAX_SIZE_MB} Mo)")

    try:
        image = read_image_bytes(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide")

    mask, prob = predict_mask(state["model"], image, threshold=threshold)
    result, contours = measure_wound(mask, px_per_cm=px_per_cm)
    overlay = make_overlay(image, contours, result["bbox"])

    wound = mask > 0
    confidence = round(float(prob[wound].mean()), 3) if wound.any() else None

    response = {
        "filename": file.filename,
        "trained_model": os.path.exists(MODEL_PATH),
        "measurements": result,
        "mean_confidence": confidence,
        "overlay_png_base64": to_base64_png(overlay),
        "mask_png_base64": to_base64_png(mask),
    }

    if explain:
        with gradcam_lock:
            cam_overlay, _ = gradcam_overlay(state["model"], image, threshold=threshold)
        response["gradcam_png_base64"] = to_base64_png(cam_overlay)

    return response
