import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chickpoo.config import CLASS_CONFIG, PROJECT_ROOT, SLUG_TO_DISPLAY, SLUG_TO_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Build ChickPoo manifest and stratified splits.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def collect_images(root: Path) -> pd.DataFrame:
    records = []
    for class_item in CLASS_CONFIG:
        slug = class_item["slug"]
        folder = root / class_item["folder"]
        files = sorted(folder.rglob("*.jpg"))
        for path in files:
            records.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "label_slug": slug,
                    "label_id": SLUG_TO_ID[slug],
                    "class_name": SLUG_TO_DISPLAY[slug],
                }
            )
    return pd.DataFrame(records)


def inspect_images(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    bad_files = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Inspecting images"):
        try:
            with Image.open(row.path) as image:
                width, height = image.size
                mode = image.mode
            rows.append({**row._asdict(), "width": width, "height": height, "mode": mode})
        except Exception as exc:
            bad_files.append({"path": row.path, "error": str(exc)})
    return pd.DataFrame(rows), bad_files


def make_splits(df: pd.DataFrame, train_size: float, val_size: float, test_size: float, seed: int):
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-size + val-size + test-size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["label_id"],
    )
    return (
        train_df.assign(split="train").sort_values("relative_path"),
        val_df.assign(split="val").sort_values("relative_path"),
        test_df.assign(split="test").sort_values("relative_path"),
    )


def plot_distribution(split_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 4.8))
    sns.countplot(data=split_df, x="class_name", hue="split", order=[item["display"] for item in CLASS_CONFIG])
    plt.title("ChickPoo Dataset Distribution by Split")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = collect_images(args.root)
    inspected_df, bad_files = inspect_images(df)
    manifest_path = args.splits_dir / "manifest.csv"
    inspected_df.to_csv(manifest_path, index=False)

    train_df, val_df, test_df = make_splits(
        inspected_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.to_csv(args.splits_dir / f"{name}.csv", index=False)
    all_splits.to_csv(args.splits_dir / "all_splits.csv", index=False)

    counts = (
        all_splits.groupby(["split", "class_name"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "class_name"])
    )
    counts.to_csv(args.splits_dir / "split_counts.csv", index=False)

    summary = {
        "total_images": int(len(inspected_df)),
        "bad_images": bad_files,
        "class_counts": inspected_df["class_name"].value_counts().sort_index().to_dict(),
        "split_counts": {
            split: group.set_index("class_name")["count"].to_dict()
            for split, group in counts.groupby("split")
        },
        "image_dimensions": {
            f"{int(width)}x{int(height)}": int(count)
            for (width, height), count in inspected_df.groupby(["width", "height"]).size().sort_values(ascending=False).items()
        },
    }
    with open(args.splits_dir / "eda_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_distribution(all_splits, args.figures_dir / "dataset_split_distribution.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
