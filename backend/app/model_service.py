import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from .recommendations import CLASS_LABELS, build_invalid_input_recommendation, build_recommendation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISEASE_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
OBJECT_MODEL_PATH = PROJECT_ROOT / "models" / "object_validation_model.keras"
OBJECT_METADATA_PATH = PROJECT_ROOT / "models" / "object_validation_metadata.json"

CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
OBJECT_CLASS_NAMES = ["not_chicken_feces", "chicken_feces"]
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60
MEDIUM_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_OBJECT_THRESHOLD = 0.69
FECES_LIKE_VISUAL_THRESHOLD = 0.35

_disease_model = None
_object_model = None
_object_metadata = None


def get_object_metadata() -> dict:
    global _object_metadata
    if _object_metadata is None:
        if OBJECT_METADATA_PATH.exists():
            with OBJECT_METADATA_PATH.open("r", encoding="utf-8") as file:
                _object_metadata = json.load(file)
        else:
            _object_metadata = {
                "best_threshold": DEFAULT_OBJECT_THRESHOLD,
                "class_names": OBJECT_CLASS_NAMES,
                "decision_rule": "Accept as chicken_feces if predicted probability >= threshold",
            }
    return _object_metadata


def get_object_threshold() -> float:
    return float(get_object_metadata().get("best_threshold", DEFAULT_OBJECT_THRESHOLD))


def get_disease_model():
    global _disease_model
    if _disease_model is None:
        _disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)
    return _disease_model


def get_object_model():
    global _object_model
    if _object_model is None:
        _object_model = tf.keras.models.load_model(OBJECT_MODEL_PATH, compile=False)
    return _object_model


def load_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("File tidak bisa dibaca sebagai gambar.") from exc


def prepare_image(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _rgb_to_hsv(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = array / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    max_channel = rgb.max(axis=-1)
    min_channel = rgb.min(axis=-1)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    mask = delta != 0
    red_mask = mask & (max_channel == red)
    green_mask = mask & (max_channel == green)
    blue_mask = mask & (max_channel == blue)
    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4
    hue *= 60

    saturation = np.where(max_channel == 0, 0, delta / max_channel)
    value = max_channel
    return hue, saturation, value


def assess_feces_like_visuals(image: Image.Image) -> dict:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    hue, saturation, value = _rgb_to_hsv(array)

    green = (hue >= 55) & (hue <= 165) & (saturation > 0.20) & (value > 0.18)
    white_urate = (saturation < 0.25) & (value > 0.58)
    gray_texture = (saturation < 0.30) & (value > 0.25) & (value <= 0.75)
    brown_texture = (((hue <= 55) | (hue >= 330)) & (saturation > 0.18) & (value > 0.12) & (value < 0.70))
    dark_texture = value < 0.32

    center = np.zeros_like(value, dtype=bool)
    center[45:179, 45:179] = True
    center_size = float(center.sum())
    center_non_green = float(((~green) & center).sum() / center_size)
    center_textured = float(((brown_texture | dark_texture | gray_texture) & center).sum() / center_size)
    center_urate = float((white_urate & center).sum() / center_size)

    green_ratio = float(green.mean())
    urate_ratio = float(white_urate.mean())
    textured_ratio = float((brown_texture | gray_texture | dark_texture).mean())
    looks_like_outdoor_feces = (
        green_ratio >= 0.18
        and center_non_green >= 0.35
        and center_textured >= 0.20
        and (urate_ratio >= 0.08 or center_urate >= 0.12)
    )

    score = min(
        1.0,
        (green_ratio / 0.42) * 0.22
        + (center_non_green / 0.70) * 0.28
        + (center_textured / 0.72) * 0.30
        + (max(urate_ratio, center_urate) / 0.34) * 0.20,
    )

    return {
        "looks_like_outdoor_feces": bool(looks_like_outdoor_feces),
        "support_score": round(score * 100, 2),
        "green_ratio": round(green_ratio * 100, 2),
        "urate_ratio": round(urate_ratio * 100, 2),
        "textured_ratio": round(textured_ratio * 100, 2),
        "center_non_green_ratio": round(center_non_green * 100, 2),
        "center_texture_ratio": round(center_textured * 100, 2),
        "center_urate_ratio": round(center_urate * 100, 2),
    }


def run_object_validation(batch: np.ndarray, image: Image.Image) -> dict:
    threshold = get_object_threshold()
    probability = float(get_object_model().predict(batch, verbose=0).reshape(-1)[0])
    visual_support = assess_feces_like_visuals(image)
    model_accepts = probability >= threshold
    visual_accepts = probability >= FECES_LIKE_VISUAL_THRESHOLD or visual_support["looks_like_outdoor_feces"]
    is_chicken_feces = model_accepts or visual_accepts
    if model_accepts:
        status = "accepted"
        decision_source = "object_model"
    elif visual_accepts:
        status = "likely_feces"
        decision_source = "visual_support"
    else:
        status = "rejected"
        decision_source = "object_model"

    return {
        "model": get_object_metadata().get("best_model_name", "efficientnetb0_finetuned"),
        "status": status,
        "decision_source": decision_source,
        "is_chicken_feces": bool(is_chicken_feces),
        "probability_chicken_feces": probability,
        "percentage_chicken_feces": round(probability * 100, 2),
        "threshold": threshold,
        "visual_support": visual_support,
        "classes": [
            {
                "key": "not_chicken_feces",
                "label": "Bukan feses ayam",
                "probability": 1.0 - probability,
                "percentage": round((1.0 - probability) * 100, 2),
            },
            {
                "key": "chicken_feces",
                "label": "Feses ayam",
                "probability": probability,
                "percentage": round(probability * 100, 2),
            },
        ],
    }


def build_probability_rows(probabilities: np.ndarray) -> list[dict]:
    rows = [
        {
            "key": class_key,
            "label": CLASS_LABELS[class_key],
            "probability": float(probabilities[index]),
            "percentage": round(float(probabilities[index]) * 100, 2),
        }
        for index, class_key in enumerate(CLASS_NAMES)
    ]
    rows.sort(key=lambda item: item["probability"], reverse=True)
    return rows


def predict_image(file_bytes: bytes) -> dict:
    image = load_image(file_bytes)
    batch = prepare_image(image)
    object_validation = run_object_validation(batch, image)

    if not object_validation["is_chicken_feces"]:
        return {
            "status": "invalid_input",
            "status_message": "Foto belum cukup sesuai untuk dibaca. Ambil ulang dengan kotoran ayam sebagai objek utama.",
            "predicted_class": "unknown",
            "predicted_label": "Foto belum sesuai",
            "confidence": 0.0,
            "confidence_percentage": 0.0,
            "probabilities": [],
            "object_validation": object_validation,
            "input_assessment": {
                "is_plausible_feces": False,
                "flags": [
                    {
                        "code": "object_validation_rejected",
                        "severity": "blocker",
                        "message": (
                            "Foto belum melewati penyaring awal "
                            f"({object_validation['percentage_chicken_feces']:.2f}% dari ambang "
                            f"{object_validation['threshold'] * 100:.0f}%)."
                        ),
                    }
                ],
            },
            "recommendation": build_invalid_input_recommendation(object_validation),
            "thresholds": {
                "object_validation": object_validation["threshold"],
                "low_confidence": LOW_CONFIDENCE_THRESHOLD,
                "high_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
            },
        }

    disease_probabilities = get_disease_model().predict(batch, verbose=0)[0]
    disease_probabilities = np.asarray(disease_probabilities, dtype=np.float64)
    best_index = int(np.argmax(disease_probabilities))
    label = CLASS_NAMES[best_index]
    confidence = float(disease_probabilities[best_index])
    probability_rows = build_probability_rows(disease_probabilities)

    likely_feces = object_validation["status"] == "likely_feces"

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        status = "needs_retake"
        status_message = "Foto bisa dibaca, tetapi tingkat keyakinan masih rendah. Ambil foto ulang dari jarak lebih dekat."
    elif confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        status = "review"
        status_message = "Hasil dapat dibaca sebagai indikasi awal dan tetap perlu diamati."
    elif likely_feces:
        status = "review"
        status_message = "Foto terindikasi sesuai untuk dianalisis. Baca hasil sebagai indikasi awal dan tetap amati kondisi ayam."
    else:
        status = "accepted"
        status_message = "Hasil terbaca jelas dan memiliki tingkat keyakinan yang baik."

    return {
        "status": status,
        "status_message": status_message,
        "predicted_class": label,
        "predicted_label": CLASS_LABELS[label],
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "probabilities": probability_rows,
        "object_validation": object_validation,
        "input_assessment": {
            "is_plausible_feces": True,
            "flags": [],
        },
        "recommendation": build_recommendation(label, confidence),
        "thresholds": {
            "object_validation": object_validation["threshold"],
            "low_confidence": LOW_CONFIDENCE_THRESHOLD,
            "high_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
        },
    }
