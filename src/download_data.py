"""
Download images from Pix2FactBenchmark dataset using multi-threading.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark/resolve/main/"
DATA_CSV_URL = f"{BASE_URL}Pix2Fact_1k.csv"


def download_file(row_data: tuple[str, str]) -> tuple[str, bool, str | None]:
    """
    Download a single file from the given URL to the target path.

    Args:
        row_data: Tuple of (image_url, local_path)

    Returns:
        Tuple of (local_path, success, error_message)
    """
    url, local_path = row_data
    try:
        urlretrieve(url, local_path)
        return (local_path, True, None)
    except Exception as e:
        return (local_path, False, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Pix2FactBenchmark images using multi-threading."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="./data",
        help="Directory to save downloaded images (e.g. ./data)",
    )
    parser.add_argument(
        "-n",
        "--workers",
        type=int,
        default=8,
        help="Number of download threads (default: 8)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset metadata...")
    df = pd.read_csv(DATA_CSV_URL)
    df.to_csv(output_dir / "Pix2Fact_1k.csv", index=False)
    if "local_image_path" not in df.columns:
        raise ValueError(
            f"Column 'local_image_path' not found. Available columns: {list(df.columns)}"
        )

    # Build (url, local_path) for each row
    tasks = []
    for _, row in df.iterrows():
        rel_path = row["local_image_path"]
        if pd.isna(rel_path) or not str(rel_path).strip():
            continue
        rel_path = str(rel_path).strip().lstrip("/")
        url = BASE_URL + rel_path
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((url, str(local_path)))

    total = len(tasks)
    success_count = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, t): t for t in tasks}
        with tqdm(total=total, desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                local_path, ok, err = future.result()
                if ok:
                    success_count += 1
                else:
                    failed.append((local_path, err))
                pbar.update(1)

    print(f"\nDone. Success: {success_count}, Failed: {len(failed)}")
    if failed:
        print("Failed downloads:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
