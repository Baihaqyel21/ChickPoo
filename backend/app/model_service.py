import io
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from PIL import Image, ImageOps

from .recommendations import CLASS_LABELS, build_invalid_input_recommendation, build_recommendation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
REFERENCE_STATS_PATH = PROJECT_ROOT / "models" / "feces_reference_stats.npz"
CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60
MEDIUM_CONFIDENCE_THRESHOLD = 0.80
FEATURE_LAYER_NAME = "global_average_pooling2d_3"

_model = None
_feature_model = None
_reference_stats = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def get_feature_model():
    global _feature_model
    if _feature_model is None:
        model = get_model()
        _feature_model = keras.Model(inputs=model.input, outputs=model.get_layer(FEATURE_LAYER_NAME).output)
    return _feature_model


def get_reference_stats():
    global _reference_stats
    if _reference_stats is None:
        if not REFERENCE_STATS_PATH.exists():
            _reference_stats = {}
        else:
            loaded = np.load(REFERENCE_STATS_PATH, allow_pickle=False)
            threshold_keys = [str(key) for key in loaded["threshold_keys"]]
            threshold_values = [float(value) for value in loaded["thresholds"]]
            _reference_stats = {
                "features": loaded["features"].astype(np.float32),
                "labels": np.asarray([str(label) for label in loaded["labels"]]),
                "centroids": loaded["centroids"].astype(np.float32),
                "class_names": [str(name) for name in loaded["class_names"]],
                "thresholds": dict(zip(threshold_keys, threshold_values)),
            }
    return _reference_stats


def load_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("File tidak bisa dibaca sebagai gambar.") from exc


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * -4
    laplacian = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(laplacian))


def assess_quality(image: Image.Image) -> dict:
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = _laplacian_variance(gray)

    flags = []
    if min(width, height) < 224:
        flags.append(
            {
                "code": "small_image",
                "severity": "warning",
                "message": "Resolusi foto cukup kecil. Jika memungkinkan, ambil foto ulang dari jarak lebih dekat.",
            }
        )
    if brightness < 45:
        flags.append(
            {
                "code": "dark_image",
                "severity": "warning",
                "message": "Foto terlihat gelap. Coba ambil ulang dengan pencahayaan lebih terang.",
            }
        )
    if brightness > 235:
        flags.append(
            {
                "code": "overexposed_image",
                "severity": "warning",
                "message": "Foto terlihat terlalu terang. Hindari pantulan flash atau cahaya langsung berlebihan.",
            }
        )
    if contrast < 18:
        flags.append(
            {
                "code": "low_contrast",
                "severity": "warning",
                "message": "Kontras foto rendah. Pastikan objek feses terlihat jelas dari latar.",
            }
        )
    if blur_score < 18:
        flags.append(
            {
                "code": "blurry_image",
                "severity": "warning",
                "message": "Foto tampak kurang tajam. Pegang kamera stabil dan fokuskan ke objek feses.",
            }
        )

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "flags": flags,
    }


def assess_object_relevance(image: Image.Image) -> dict:
    resized = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    max_channel = np.max(array, axis=2)
    min_channel = np.min(array, axis=2)
    brightness = np.mean(array, axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1)
    channel_sum = np.maximum(red + green + blue, 1)
    red_ratio = red / channel_sum
    green_ratio = green / channel_sum
    blue_ratio = blue / channel_sum

    white_background = (red > 235) & (green > 235) & (blue > 235) & ((max_channel - min_channel) < 18)
    bright_plain = (brightness > 220) & (saturation < 0.11)
    skin_like = (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & (red > green)
        & (red > blue)
        & ((red - green) > 12)
        & ((red - blue) > 24)
        & (red_ratio > 0.36)
        & (green_ratio > 0.25)
        & (blue_ratio < 0.33)
        & (brightness > 70)
    )
    cooked_food_like = (
        (red > 125)
        & (green > 70)
        & (blue < 135)
        & (red > green * 1.04)
        & (green > blue * 1.02)
        & (saturation > 0.16)
        & (brightness > 85)
        & (brightness < 235)
    )
    plate_like = (
        (brightness > 115)
        & (brightness < 238)
        & (saturation < 0.13)
        & ((max_channel - min_channel) < 30)
    )
    garnish_like = (
        (green > red * 1.04)
        & (green > blue * 1.08)
        & (saturation > 0.18)
        & (brightness > 45)
        & (brightness < 180)
    )
    dark_organic = (brightness < 105) & (saturation > 0.12)
    brown_green_organic = (
        (brightness > 35)
        & (brightness < 215)
        & (saturation > 0.12)
        & (((green >= red * 0.72) & (green > blue * 0.9)) | ((red >= green) & (green >= blue * 0.72)))
    )
    feces_like = (dark_organic | brown_green_organic) & (~white_background)

    white_fraction = float(np.mean(white_background))
    plain_fraction = float(np.mean(bright_plain))
    skin_fraction = float(np.mean(skin_like))
    cooked_food_fraction = float(np.mean(cooked_food_like))
    plate_like_fraction = float(np.mean(plate_like))
    garnish_like_fraction = float(np.mean(garnish_like))
    feces_like_fraction = float(np.mean(feces_like))
    foreground_fraction = float(np.mean(~bright_plain))

    flags = []
    if white_fraction > 0.42 and skin_fraction > 0.055:
        flags.append(
            {
                "code": "human_skin_detected",
                "severity": "blocker",
                "message": "Foto terindikasi berisi objek kulit atau tangan pada latar terang, bukan feses ayam.",
            }
        )
    if plain_fraction > 0.72 and foreground_fraction < 0.24:
        flags.append(
            {
                "code": "plain_non_feces_scene",
                "severity": "blocker",
                "message": "Foto didominasi latar polos dan objek utama tidak cukup menyerupai feses ayam.",
            }
        )
    if plain_fraction > 0.55 and feces_like_fraction < 0.13:
        flags.append(
            {
                "code": "low_feces_visual_signal",
                "severity": "blocker",
                "message": "Sinyal warna dan tekstur feses pada foto terlalu rendah untuk diprediksi sebagai kotoran ayam.",
            }
        )
    if cooked_food_fraction > 0.20 and plate_like_fraction > 0.18 and feces_like_fraction < 0.78:
        flags.append(
            {
                "code": "plated_food_pattern",
                "severity": "blocker",
                "message": "Foto terindikasi menampilkan makanan atau ayam matang di piring, bukan feses ayam.",
            }
        )
    if cooked_food_fraction > 0.26 and garnish_like_fraction > 0.015 and feces_like_fraction < 0.72:
        flags.append(
            {
                "code": "cooked_food_garnish_pattern",
                "severity": "blocker",
                "message": "Pola warna foto lebih menyerupai makanan matang dengan garnish daripada feses ayam.",
            }
        )

    return {
        "is_plausible_feces": len(flags) == 0,
        "metrics": {
            "white_background_fraction": round(white_fraction, 4),
            "plain_bright_fraction": round(plain_fraction, 4),
            "skin_like_fraction": round(skin_fraction, 4),
            "cooked_food_fraction": round(cooked_food_fraction, 4),
            "plate_like_fraction": round(plate_like_fraction, 4),
            "garnish_like_fraction": round(garnish_like_fraction, 4),
            "feces_like_fraction": round(feces_like_fraction, 4),
            "foreground_fraction": round(foreground_fraction, 4),
        },
        "flags": flags,
    }


def prepare_image(image: Image.Image) -> np.ndarray:
    resized = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _normalize_features(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)


def assess_feature_relevance(batch: np.ndarray) -> dict:
    stats = get_reference_stats()
    if not stats:
        return {
            "is_plausible_feces": True,
            "metrics": {"reference_available": False},
            "flags": [],
        }

    embedding = get_feature_model().predict(batch, verbose=0).astype(np.float32)
    embedding = _normalize_features(embedding)[0]

    reference_features = stats["features"]
    centroids = stats["centroids"]
    nearest_scores = reference_features @ embedding
    centroid_scores = centroids @ embedding
    nearest_index = int(np.argmax(nearest_scores))
    centroid_index = int(np.argmax(centroid_scores))
    nearest_similarity = float(nearest_scores[nearest_index])
    centroid_similarity = float(centroid_scores[centroid_index])
    thresholds = stats["thresholds"]

    nearest_threshold = thresholds.get("nearest_similarity", 0.60)
    centroid_threshold = thresholds.get("centroid_similarity", 0.48)
    strict_nearest_threshold = thresholds.get("strict_nearest_similarity", 0.52)
    strict_centroid_threshold = thresholds.get("strict_centroid_similarity", 0.38)

    flags = []
    strict_mismatch = nearest_similarity < strict_nearest_threshold or centroid_similarity < strict_centroid_threshold
    soft_mismatch = nearest_similarity < nearest_threshold and centroid_similarity < centroid_threshold
    if strict_mismatch or soft_mismatch:
        flags.append(
            {
                "code": "feature_manifold_mismatch",
                "severity": "blocker",
                "message": "Pola visual foto berada di luar distribusi fitur feses ayam pada dataset ChickPoo.",
            }
        )

    return {
        "is_plausible_feces": len(flags) == 0,
        "metrics": {
            "reference_available": True,
            "nearest_similarity": round(nearest_similarity, 4),
            "centroid_similarity": round(centroid_similarity, 4),
            "nearest_threshold": round(nearest_threshold, 4),
            "centroid_threshold": round(centroid_threshold, 4),
            "nearest_reference_class": str(stats["labels"][nearest_index]),
            "nearest_centroid_class": stats["class_names"][centroid_index],
        },
        "flags": flags,
    }


def combine_relevance_checks(visual_relevance: dict, feature_relevance: dict) -> dict:
    flags = visual_relevance["flags"] + feature_relevance["flags"]
    return {
        "is_plausible_feces": visual_relevance["is_plausible_feces"] and feature_relevance["is_plausible_feces"],
        "flags": flags,
        "visual_metrics": visual_relevance["metrics"],
        "feature_metrics": feature_relevance["metrics"],
    }


def predict_image(file_bytes: bytes) -> dict:
    image = load_image(file_bytes)
    quality = assess_quality(image)
    visual_relevance = assess_object_relevance(image)
    batch = prepare_image(image)
    probabilities = get_model().predict(batch, verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    feature_relevance = assess_feature_relevance(batch)
    relevance = combine_relevance_checks(visual_relevance, feature_relevance)

    best_index = int(np.argmax(probabilities))
    label = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index])

    probability_rows = [
        {
            "key": class_key,
            "label": CLASS_LABELS[class_key],
            "probability": float(probabilities[index]),
            "percentage": round(float(probabilities[index]) * 100, 2),
        }
        for index, class_key in enumerate(CLASS_NAMES)
    ]
    probability_rows.sort(key=lambda item: item["probability"], reverse=True)

    if not relevance["is_plausible_feces"]:
        return {
            "status": "invalid_input",
            "status_message": "Sepertinya foto belum menunjukkan feses ayam. Ambil ulang foto dengan objek kotoran ayam sebagai fokus utama.",
            "predicted_class": "unknown",
            "predicted_label": "Foto tidak dikenali sebagai feses ayam",
            "confidence": 0.0,
            "confidence_percentage": 0.0,
            "probabilities": probability_rows,
            "quality": quality,
            "input_assessment": relevance,
            "recommendation": build_invalid_input_recommendation(quality["flags"], relevance["flags"]),
            "thresholds": {
                "low_confidence": LOW_CONFIDENCE_THRESHOLD,
                "high_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
            },
        }

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        status = "needs_retake"
        status_message = "Foto belum cukup meyakinkan untuk screening. Kemungkinan foto kurang jelas atau bukan foto feses ayam."
    elif confidence < MEDIUM_CONFIDENCE_THRESHOLD or quality["flags"]:
        status = "review"
        status_message = "Hasil dapat dipakai sebagai indikasi awal, tetapi perhatikan catatan kualitas foto."
    else:
        status = "accepted"
        status_message = "Foto cukup jelas dan model memiliki keyakinan yang baik."

    return {
        "status": status,
        "status_message": status_message,
        "predicted_class": label,
        "predicted_label": CLASS_LABELS[label],
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "probabilities": probability_rows,
        "quality": quality,
        "input_assessment": relevance,
        "recommendation": build_recommendation(label, confidence, quality["flags"]),
        "thresholds": {
            "low_confidence": LOW_CONFIDENCE_THRESHOLD,
            "high_confidence": MEDIUM_CONFIDENCE_THRESHOLD,
        },
    }
