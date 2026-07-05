import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_service import (
    CLASS_NAMES,
    DISEASE_MODEL_PATH,
    OBJECT_MODEL_PATH,
    get_disease_metadata,
    get_disease_model,
    get_object_metadata,
    get_object_model,
    get_object_threshold,
    get_training_summary,
    predict_image,
)
from .recommendations import CLASS_LABELS

app = FastAPI(
    title="ChickPoo API",
    description="Backend prediksi kesehatan ayam berbasis foto feses.",
    version="0.1.0",
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "")
    configured_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return DEFAULT_CORS_ORIGINS + configured_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "object_model_exists": OBJECT_MODEL_PATH.exists(),
        "object_model_path": str(OBJECT_MODEL_PATH),
        "object_threshold": get_object_threshold(),
        "object_model_name": get_object_metadata().get("best_model_name"),
        "object_test_metrics": get_object_metadata().get("test_metrics", {}),
        "disease_model_exists": DISEASE_MODEL_PATH.exists(),
        "disease_model_path": str(DISEASE_MODEL_PATH),
        "disease_model_name": get_disease_metadata().get("best_model_name"),
        "disease_test_metrics": get_disease_metadata().get("test_metrics", {}),
        "training_summary": get_training_summary(),
        "classes": [{"key": key, "label": CLASS_LABELS[key]} for key in CLASS_NAMES],
    }


@app.post("/warmup")
def warmup():
    get_object_model()
    get_disease_model()
    return {"status": "ready", "pipeline": ["object_validation", "disease_classification"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    try:
        return predict_image(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
