from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_service import CLASS_NAMES, MODEL_PATH, REFERENCE_STATS_PATH, get_model, predict_image
from .recommendations import CLASS_LABELS

app = FastAPI(
    title="ChickPoo API",
    description="Backend prediksi kesehatan ayam berbasis foto feses.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "reference_stats_exists": REFERENCE_STATS_PATH.exists(),
        "reference_stats_path": str(REFERENCE_STATS_PATH),
        "classes": [{"key": key, "label": CLASS_LABELS[key]} for key in CLASS_NAMES],
    }


@app.post("/warmup")
def warmup():
    get_model()
    return {"status": "ready"}


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
