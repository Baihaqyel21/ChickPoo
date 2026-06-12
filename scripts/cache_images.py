import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chickpoo.config import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Create a resized image cache for faster CPU training.")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache_128")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimize", action="store_true")
    return parser.parse_args()


def cache_one(source: Path, destination: Path, size: int, quality: int, optimize: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=quality, optimize=optimize)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.splits_dir / "all_splits.csv")
    cached_paths = []
    jobs = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Caching images"):
        source = Path(row.path)
        destination = args.cache_dir / row.relative_path
        cached_paths.append(str(destination))
        if not destination.exists():
            jobs.append((source, destination))

    if jobs:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(cache_one, source, destination, args.size, args.quality, args.optimize)
                for source, destination in jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing missing cache"):
                future.result()

    manifest["cached_path"] = cached_paths
    manifest.to_csv(args.splits_dir / f"all_splits_cached_{args.size}.csv", index=False)
    for split in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split].copy()
        split_df.to_csv(args.splits_dir / f"{split}_cached_{args.size}.csv", index=False)
    print(f"Cached {len(manifest)} images in {args.cache_dir}")


if __name__ == "__main__":
    main()
