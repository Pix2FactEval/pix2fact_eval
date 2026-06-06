"""Export Pix2FactBenchmark from Hugging Face parquet to local CSV + images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_ID = "pix2fact/Pix2FactBenchmark"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_CSV_NAME = "Pix2Fact_1k.csv"


def safe_image_path(row: dict[str, Any], row_idx: int) -> str:
    raw_path = str(row.get("local_image_path", "") or "").strip().lstrip("/")
    if raw_path and raw_path.lower() != "nan":
        path = Path(raw_path)
        return str(path if path.suffix else path.with_suffix(".jpg"))
    row_id = str(row.get("index", row_idx)).replace("/", "_")
    return f"images/{row_id}.jpg"


def save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, format="JPEG", quality=95)
    elif suffix == ".png":
        image.save(path, format="PNG")
    else:
        image.convert("RGB").save(path, format="JPEG", quality=95)


def export_dataset(output_dir: Path, split: str, limit: int | None) -> Path:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(tqdm(dataset, desc="Exporting Pix2Fact")):
        row_dict = dict(row)
        image = row_dict.pop("image")
        rel_image_path = safe_image_path(row_dict, row_idx)
        image_path = output_dir / rel_image_path
        save_image(image, image_path)
        row_dict["local_image_path"] = rel_image_path
        rows.append(row_dict)

    csv_path = output_dir / DEFAULT_CSV_NAME
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Pix2FactBenchmark from Hugging Face parquet and export local images/CSV."
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=0, help="If > 0, export only the first N rows.")
    args = parser.parse_args()

    csv_path = export_dataset(
        output_dir=Path(args.output_dir),
        split=args.split,
        limit=args.limit if args.limit > 0 else None,
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
