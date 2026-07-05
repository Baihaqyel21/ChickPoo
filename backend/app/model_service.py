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
MODELS_DIR = PROJECT_ROOT / "models"

DISEASE_MODEL_PATH = MODELS_DIR / "best_disease_classifier_model.keras"
OBJECT_MODEL_PATH = MODELS_DIR / "best_object_validation_model.keras"
OBJECT_METADATA_PATH = MODELS_DIR / "object_validation_metadata.json"
DISEASE_METADATA_PATH = MODELS_DIR / "disease_classifier_metadata.json"
TRAINING_SUMMARY_PATH = MODELS_DIR / "final_training_summary.json"

CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
OBJECT_CLASS_NAMES = ["not_chicken_feces", "chicken_feces"]
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60
MEDIUM_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_OBJECT_THRESHOLD = 0.42

_disease_model = None
_object_model = None
_object_metadata = None
_disease_metadata = None
_training_summary = None


def _read_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_object_metadata() -> dict:
    global _object_metadata
    if _object_metadata is None:
        _object_metadata = _read_json(
            OBJECT_METADATA_PATH,
            {
                "task": "object_validation",
                "class_names": OBJECT_CLASS_NAMES,
                "positive_class": "chicken_feces",
                "img_size": IMAGE_SIZE,
                "threshold": DEFAULT_OBJECT_THRESHOLD,
                "best_model_name": "best_object_validation_model",
            },
        )
    return _object_metadata


def get_disease_metadata() -> dict:
    global _disease_metadata
    if _disease_metadata is None:
        _disease_metadata = _read_json(
            DISEASE_METADATA_PATH,
            {
                "task": "disease_classification",
                "class_names": CLASS_NAMES,
                "img_size": IMAGE_SIZE,
                "best_model_name": "best_disease_classifier_model",
            },
        )
    return _disease_metadata


def get_training_summary() -> dict:
    global _training_summary
    if _training_summary is None:
        _training_summary = _read_json(TRAINING_SUMMARY_PATH, {})
    return _training_summary


def get_object_threshold() -> float:
    metadata = get_object_metadata()
    return float(metadata.get("threshold", metadata.get("best_threshold", DEFAULT_OBJECT_THRESHOLD)))


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
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _sigmoid_probability(prediction: np.ndarray) -> float:
    value = float(np.asarray(prediction, dtype=np.float64).reshape(-1)[0])
    return float(np.clip(value, 0.0, 1.0))


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

    saturation = np.divide(delta, max_channel, out=np.zeros_like(delta), where=max_channel != 0)
    value = max_channel
    return hue, saturation, value


def assess_outdoor_feces_visuals(image: Image.Image) -> dict:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    hue, saturation, value = _rgb_to_hsv(array)

    green = (hue >= 45) & (hue <= 170) & (saturation > 0.18) & (value > 0.15)
    white_urate = (saturation < 0.32) & (value > 0.52)
    gray_texture = (saturation < 0.34) & (value > 0.20) & (value <= 0.78)
    brown_texture = (((hue <= 58) | (hue >= 330)) & (saturation > 0.16) & (value > 0.10) & (value < 0.74))
    red_wet_texture = (((hue <= 18) | (hue >= 345)) & (saturation > 0.28) & (value > 0.18) & (value < 0.90))
    cyan_cast_texture = ((hue >= 165) & (hue <= 235) & (saturation > 0.12) & (value > 0.22) & (value < 0.95))
    dark_texture = value < 0.30
    feces_like = (white_urate | gray_texture | brown_texture | dark_texture) & (~green)

    center = np.zeros_like(value, dtype=bool)
    center[36:188, 36:188] = True
    center_size = float(center.sum())

    grayscale = (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]) / 255.0
    gradient_y = np.abs(np.diff(grayscale, axis=0, append=grayscale[-1:, :]))
    gradient_x = np.abs(np.diff(grayscale, axis=1, append=grayscale[:, -1:]))
    textured_edges = (gradient_x + gradient_y) > 0.055

    green_ratio = float(green.mean())
    center_non_green = float(((~green) & center).sum() / center_size)
    center_feces_like = float((feces_like & center).sum() / center_size)
    center_white_urate = float((white_urate & center).sum() / center_size)
    center_brown_gray_dark = float(((brown_texture | gray_texture | dark_texture) & center).sum() / center_size)
    center_red_wet = float((red_wet_texture & center).sum() / center_size)
    center_cyan_cast = float((cyan_cast_texture & center).sum() / center_size)
    center_edge_ratio = float((textured_edges & center).sum() / center_size)

    urate_or_dense_texture = center_white_urate >= 0.22 or (
        center_brown_gray_dark >= 0.70 and center_edge_ratio >= 0.60
    )
    grass_surface_feces = (
        green_ratio >= 0.30
        and center_non_green >= 0.52
        and center_feces_like >= 0.52
        and center_edge_ratio >= 0.38
        and urate_or_dense_texture
    )
    dry_surface_urate_feces = (
        green_ratio <= 0.20
        and center_white_urate >= 0.62
        and center_feces_like >= 0.80
        and center_edge_ratio >= 0.28
    )
    dry_surface_mixed_feces = (
        green_ratio <= 0.20
        and center_white_urate >= 0.22
        and center_brown_gray_dark >= 0.84
        and center_feces_like >= 0.76
        and center_edge_ratio >= 0.42
    )
    red_wet_feces = (
        green_ratio <= 0.30
        and center_red_wet >= 0.28
        and center_white_urate >= 0.16
        and center_brown_gray_dark >= 0.72
        and center_feces_like >= 0.76
        and center_edge_ratio >= 0.28
    )
    cyan_cast_feces = (
        green_ratio <= 0.18
        and center_cyan_cast >= 0.30
        and center_white_urate >= 0.28
        and center_brown_gray_dark >= 0.66
        and center_feces_like >= 0.82
        and center_edge_ratio >= 0.25
    )
    looks_like_outdoor_feces = (
        grass_surface_feces
        or dry_surface_urate_feces
        or dry_surface_mixed_feces
        or red_wet_feces
        or cyan_cast_feces
    )
    if grass_surface_feces:
        support_reason = "visual_support_grass_surface"
    elif dry_surface_urate_feces:
        support_reason = "visual_support_dry_surface_urate"
    elif dry_surface_mixed_feces:
        support_reason = "visual_support_dry_surface_mixed"
    elif red_wet_feces:
        support_reason = "visual_support_red_wet_pattern"
    elif cyan_cast_feces:
        support_reason = "visual_support_cyan_cast_pattern"
    else:
        support_reason = "none"
    support_score = min(
        1.0,
        (green_ratio / 0.62) * 0.18
        + (center_non_green / 0.64) * 0.24
        + (center_feces_like / 0.64) * 0.24
        + (center_edge_ratio / 0.62) * 0.18
        + (max(center_white_urate / 0.35, center_brown_gray_dark / 0.78)) * 0.16,
    )

    return {
        "looks_like_outdoor_feces": bool(looks_like_outdoor_feces),
        "support_reason": support_reason,
        "support_score": round(support_score * 100, 2),
        "grass_surface_feces": bool(grass_surface_feces),
        "dry_surface_urate_feces": bool(dry_surface_urate_feces),
        "dry_surface_mixed_feces": bool(dry_surface_mixed_feces),
        "red_wet_feces": bool(red_wet_feces),
        "cyan_cast_feces": bool(cyan_cast_feces),
        "green_ratio": round(green_ratio * 100, 2),
        "center_non_green_ratio": round(center_non_green * 100, 2),
        "center_feces_like_ratio": round(center_feces_like * 100, 2),
        "center_white_urate_ratio": round(center_white_urate * 100, 2),
        "center_brown_gray_dark_ratio": round(center_brown_gray_dark * 100, 2),
        "center_red_wet_ratio": round(center_red_wet * 100, 2),
        "center_cyan_cast_ratio": round(center_cyan_cast * 100, 2),
        "center_edge_ratio": round(center_edge_ratio * 100, 2),
    }


def run_object_validation(batch: np.ndarray, image: Image.Image) -> dict:
    threshold = get_object_threshold()
    probability = _sigmoid_probability(get_object_model().predict(batch, verbose=0))
    visual_support = assess_outdoor_feces_visuals(image)
    model_accepts = probability >= threshold
    visual_accepts = visual_support["looks_like_outdoor_feces"]
    is_chicken_feces = model_accepts or visual_accepts
    status = "accepted" if is_chicken_feces else "rejected"
    decision_source = "object_validation_model" if model_accepts else (
        visual_support.get("support_reason", "visual_support") if visual_accepts else "object_validation_model"
    )

    return {
        "model": get_object_metadata().get("best_model_name", "best_object_validation_model"),
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


def _model_info() -> dict:
    return {
        "object_validation": {
            "name": get_object_metadata().get("best_model_name"),
            "threshold": get_object_threshold(),
            "test_metrics": get_object_metadata().get("test_metrics", {}),
        },
        "disease_classification": {
            "name": get_disease_metadata().get("best_model_name"),
            "test_metrics": get_disease_metadata().get("test_metrics", {}),
        },
    }


def predict_image(file_bytes: bytes) -> dict:
    image = load_image(file_bytes)
    batch = prepare_image(image)
    object_validation = run_object_validation(batch, image)

    if not object_validation["is_chicken_feces"]:
        return {
            "status": "invalid_input",
            "status_message": "Foto belum terdeteksi sebagai feses ayam. Ambil ulang dengan objek kotoran ayam yang lebih jelas dan dekat.",
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
            "model_info": _model_info(),
        }

    disease_probabilities = get_disease_model().predict(batch, verbose=0)[0]
    disease_probabilities = np.asarray(disease_probabilities, dtype=np.float64)
    best_index = int(np.argmax(disease_probabilities))
    label = CLASS_NAMES[best_index]
    confidence = float(disease_probabilities[best_index])
    probability_rows = build_probability_rows(disease_probabilities)
    visual_override = str(object_validation["decision_source"]).startswith("visual_support_")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        status = "needs_retake"
        status_message = "Foto bisa dibaca, tetapi tingkat keyakinan masih rendah. Ambil foto ulang dari jarak lebih dekat."
    elif confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        status = "review"
        status_message = "Hasil dapat dibaca sebagai indikasi awal dan tetap perlu diamati."
    elif visual_override:
        status = "review"
        status_message = "Foto terindikasi sebagai feses ayam dari pola visual. Baca hasil sebagai indikasi awal dan tetap amati kondisi ayam."
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
            "flags": [
                {
                    "code": "outdoor_visual_support",
                    "severity": "info",
                    "message": "Foto lolos karena pola visual feses terdeteksi kuat.",
                }
            ]
            if visual_override
            else [],
        },
        "recommendation": build_recommendation(label, confidence),
        "thresholds": {
            "object_validation": object_validation["threshold"],
            "low_confidence": LOW_CONFIDENCE_THRESHOLD,
            "high_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
        },
        "model_info": _model_info(),
    }
