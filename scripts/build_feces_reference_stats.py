import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
OUTPUT_PATH = PROJECT_ROOT / "models" / "feces_reference_stats.npz"
CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
IMAGE_SIZE = 224
DEFAULT_FEATURE_LAYER = "global_average_pooling2d_3"


def numeric_key(path: Path):
    suffix = path.stem.split(".")[-1]
    return int(suffix) if suffix.isdigit() else path.name


def select_images(class_dir: Path, limit: int) -> list[Path]:
    files = sorted(class_dir.glob("*.jpg"), key=numeric_key)
    if len(files) <= limit:
        return files
    positions = np.linspace(0, len(files) - 1, limit, dtype=int)
    return [files[index] for index in positions]


def prepare_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    left = max(0, (image.width - min(image.size)) // 2)
    top = max(0, (image.height - min(image.size)) // 2)
    right = left + min(image.size)
    bottom = top + min(image.size)
    image = image.crop((left, top, right, bottom)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def normalize(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Build ChickPoo feces feature reference stats.")
    parser.add_argument("--per-class", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-layer", default=DEFAULT_FEATURE_LAYER)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH, compile=False)
    feature_model = keras.Model(inputs=model.input, outputs=model.get_layer(args.feature_layer).output)

    paths: list[Path] = []
    labels: list[str] = []
    for class_name in CLASS_NAMES:
        selected = select_images(PROJECT_ROOT / class_name, args.per_class)
        paths.extend(selected)
        labels.extend([class_name] * len(selected))

    batches = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch = np.stack([prepare_image(path) for path in batch_paths]).astype(np.float32)
        batches.append(feature_model.predict(batch, verbose=0))

    features = normalize(np.vstack(batches).astype(np.float32))
    labels_array = np.asarray(labels)

    centroids = []
    for class_name in CLASS_NAMES:
        centroid = features[labels_array == class_name].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.vstack(centroids).astype(np.float32)

    similarities = features @ features.T
    np.fill_diagonal(similarities, -1)
    nearest_similarity = similarities.max(axis=1)
    centroid_similarity = (features @ centroids.T).max(axis=1)

    thresholds = {
        "nearest_similarity": float(max(0.56, np.quantile(nearest_similarity, 0.005) - 0.08)),
        "centroid_similarity": float(max(0.42, np.quantile(centroid_similarity, 0.005) - 0.08)),
        "strict_nearest_similarity": float(max(0.48, np.quantile(nearest_similarity, 0.001) - 0.12)),
        "strict_centroid_similarity": float(max(0.34, np.quantile(centroid_similarity, 0.001) - 0.12)),
    }

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        labels=labels_array,
        centroids=centroids,
        class_names=np.asarray(CLASS_NAMES),
        thresholds=np.asarray([thresholds[key] for key in sorted(thresholds)]),
        threshold_keys=np.asarray(sorted(thresholds)),
        feature_layer=np.asarray(args.feature_layer),
        source_files=np.asarray([str(path.relative_to(PROJECT_ROOT)) for path in paths]),
    )

    print(f"Saved {output_path}")
    print(f"Images: {len(paths)}")
    print("Thresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
