import argparse
import ast
import base64
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import dotenv
import openai
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.prompt import PROMPT_TEMPLATE

dotenv.load_dotenv()

DEFAULT_INPUT_CSV = "data/Pix2Fact_1k.csv"
DEFAULT_IMAGE_DIR = "data"
DEFAULT_OUTPUT_DIR = "outputs/pix2fact_eval"
DEFAULT_QUESTION_COLUMN = "[Final]question"
DEFAULT_IMAGE_COLUMN = "local_image_path"
DEFAULT_MAX_IMAGE_MB = 10

# Never commit real keys; set OPENROUTER_API_KEY in .env (see env.example).
OPENROUTER_API_KEY_PLACEHOLDER = "<YOUR_OPENROUTER_API_KEY>"
_raw_openrouter_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_KEY = (
    _raw_openrouter_key
    if (_raw_openrouter_key and _raw_openrouter_key.strip() and _raw_openrouter_key != OPENROUTER_API_KEY_PLACEHOLDER)
    else None
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-3.1-pro-preview:online")
OPENROUTER_ENABLE_WEB_SEARCH = os.getenv("OPENROUTER_ENABLE_WEB_SEARCH", "1").lower() in {"1", "true", "yes"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = (
    openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL) if OPENROUTER_API_KEY else None
)


def crop_image_by_bbox(image_path: Path, bounding_box_raw: Any) -> Image.Image:
    """Return a PIL Image cropped exactly to the first rect in bounding_box_raw."""
    ann = ast.literal_eval(str(bounding_box_raw))
    region = ann[0]["region"]
    pts = (region["tl"], region["tr"], region["bl"], region["br"])
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    x0 = max(0, int(min(xs)))
    y0 = max(0, int(min(ys)))
    x1 = min(w, int(max(xs)))
    y1 = min(h, int(max(ys)))
    return im.crop((x0, y0, x1, y1))


def crop_image_by_bbox_expanded(
    image_path: Path,
    bounding_box_raw: Any,
    expand_ratio: float = 0.15,
) -> Image.Image:
    """Return a PIL Image cropped to the first rect in bounding_box_raw, expanded
    outward by *expand_ratio* of the original image width/height on each side.
    For example, expand_ratio=0.15 pads each edge by 15 % of the image dimension.
    """
    ann = ast.literal_eval(str(bounding_box_raw))
    region = ann[0]["region"]
    pts = (region["tl"], region["tr"], region["bl"], region["br"])
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    pad_x = int(w * expand_ratio)
    pad_y = int(h * expand_ratio)
    x0 = max(0, int(min(xs)) - pad_x)
    y0 = max(0, int(min(ys)) - pad_y)
    x1 = min(w, int(max(xs)) + pad_x)
    y1 = min(h, int(max(ys)) + pad_y)
    return im.crop((x0, y0, x1, y1))


def get_image_base64(image_path: Path, max_size_bytes: int) -> str:
    """Resize/re-encode so the base64-encoded output stays under max_size_bytes.

    Anthropic (and most APIs) measure the base64 string length, which is 4/3× the
    raw image bytes.  We therefore cap raw bytes at max_size_bytes * 3/4 so that
    the final base64 output never exceeds max_size_bytes.
    """
    # Convert the caller's base64-size limit to an equivalent raw-bytes limit.
    raw_limit = int(max_size_bytes * 3 / 4)

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    if len(image_data) <= raw_limit:
        return base64.b64encode(image_data).decode("utf-8")

    # Alias so the rest of the function uses the raw limit consistently.
    max_size_bytes = raw_limit

    img = Image.open(io.BytesIO(image_data))
    img_format = (img.format or "JPEG").upper()
    # JPEG compresses photos much better than PNG; normalize modes for JPEG output.
    if img_format != "PNG":
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img_format = "JPEG"

    output_buffer = io.BytesIO()
    quality = 85
    min_side = 100

    while True:
        output_buffer.seek(0)
        output_buffer.truncate(0)
        if img_format == "PNG":
            img.save(output_buffer, format="PNG", optimize=True)
        else:
            img.save(output_buffer, format="JPEG", quality=quality, optimize=True)

        if output_buffer.tell() <= max_size_bytes:
            return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        width, height = img.size
        new_width = int(width * 0.9)
        new_height = int(height * 0.9)
        if new_width >= min_side and new_height >= min_side:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            continue

        # Cannot shrink further at this min_side; lower JPEG quality or relax limits.
        if img_format == "JPEG" and quality > 10:
            quality -= 10
            continue

        if img_format == "PNG":
            img = img.convert("RGB")
            img_format = "JPEG"
            quality = 85
            continue

        if min_side > 32:
            min_side = 32
            continue
        if min_side > 16:
            min_side = 16
            continue

        # Last resort: hard shrink so we always respect the byte cap.
        if max(width, height) > 64:
            scale = 0.5
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
            continue

        # Absolute fallback: keep halving until under the limit.
        while output_buffer.tell() > max_size_bytes and max(img.size) > 1:
            w, h = img.size
            img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
            output_buffer.seek(0)
            output_buffer.truncate(0)
            img.save(output_buffer, format="JPEG", quality=10, optimize=True)

        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _extract_content_from_raw_text(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    payload = json.loads(raw_text[start : end + 1])
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def call_model_with_retry(base64_image: str, question: str, model_name: str, retries: int, image_path: Any = None) -> tuple[str, str]:
    full_prompt = f"{PROMPT_TEMPLATE}\nInput Question: {question}\nInput Image: \n"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]

    for attempt in range(1, retries + 1):
        try:
            if not client:
                return "error", "Missing OPENROUTER_API_KEY (set in environment to a real key; not the placeholder)."
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "max_tokens": 16000,
            }
            print("call openrouter!")
            raw_response = client.chat.completions.with_raw_response.create(**request_kwargs)
            raw_text = raw_response.text
            print("Get response:", raw_text)
            try:
                response = raw_response.parse()
                content = response.choices[0].message.content if response.choices else ""
            except Exception:
                content = _extract_content_from_raw_text(raw_text)
            return "success", content or ""
        except openai.RateLimitError as exc:
            logger.warning("Attempt %s/%s rate limit: %s", attempt, retries, exc)
            time.sleep(10)
        except openai.OpenAIError as exc:
            exc_str = str(exc)
            if "image exceeds" in exc_str or "image_url" in exc_str or "base64" in exc_str:
                logger.error(
                    "Attempt %s/%s image-size error (image_path=%s): %s",
                    attempt, retries, image_path, exc,
                )
            else:
                logger.warning("Attempt %s/%s api error: %s", attempt, retries, exc)
            time.sleep(10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attempt %s/%s unexpected error: %s", attempt, retries, exc)
            time.sleep(10)

    return "error", f"Failed after {retries} attempts"


def parse_output_json(model_output: str) -> dict[str, Any]:
    if not isinstance(model_output, str) or model_output.startswith("ERROR:"):
        return {}
    text = model_output.strip()
    candidates = [text]

    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced_matches)

    for candidate in candidates:
        c = candidate.strip()
        if not c:
            continue
        left = c.find("{")
        right = c.rfind("}")
        if left != -1 and right != -1 and right >= left:
            c = c[left : right + 1]
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            continue
    return {}


def get_by_alias(data: dict[str, Any], *aliases: str, default: Any = "") -> Any:
    for alias in aliases:
        if alias in data:
            return data[alias]
    lowered_map = {str(k).lower(): v for k, v in data.items()}
    for alias in aliases:
        key = alias.lower()
        if key in lowered_map:
            return lowered_map[key]
    return default


def _to_list(val: Any) -> list:
    """Coerce a value to a list; wraps non-list scalars so callers always get a list."""
    if isinstance(val, list):
        return val
    if not val:
        return []
    return [str(val)]


def fill_response_fields(output_row: dict[str, Any]) -> None:
    parsed = parse_output_json(str(output_row.get("model_output", "")))
    output_row["response_observation"] = str(
        get_by_alias(parsed, "Observation", "observation", default="")
    )
    output_row["response_search_plan"] = json.dumps(
        _to_list(get_by_alias(parsed, "Search Plan", "search_plan", default=[])), ensure_ascii=False
    )
    output_row["response_search_query"] = json.dumps(
        _to_list(get_by_alias(parsed, "Search Query", "search_query", default=[])), ensure_ascii=False
    )
    output_row["response_comprehensive_answer"] = str(
        get_by_alias(parsed, "Comprehensive Answer", "comprehensive_answer", default="")
    )
    output_row["response_final_answer"] = str(
        get_by_alias(parsed, "Final Answer", "final_answer", "answer", default="")
    )
    output_row["response_is_error"] = str(output_row.get("model_output", "")).startswith("ERROR:")


def normalize_image_name(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return Path(text).name


def image_name_from_cdn_url(value: Any) -> str:
    url = str(value).strip()
    if not url or url.lower() == "nan":
        return ""
    parsed = urlparse(url)
    return Path(unquote(parsed.path)).name


def resolve_image_path(image_dir: Path, row: dict[str, Any], image_column: str) -> Path:
    text = str(row.get(image_column, "")).strip().lstrip("/")
    if text and text.lower() != "nan":
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        direct = image_dir / candidate
        if direct.exists() or len(candidate.parts) > 1:
            return direct
        return image_dir / candidate.name

    image_name = image_name_from_cdn_url(row.get("cdn_url", ""))
    return image_dir / image_name


def process_row(
    row_index: int,
    row_data: dict[str, Any],
    image_dir: Path,
    question_column: str,
    image_column: str,
    model_name: str,
    retries: int,
    per_case_dir: Path,
    max_size_bytes: int,
    crop_bbox: bool = False,
    crop_bbox_expand: bool = False,
    bbox_expand_ratio: float = 0.15,
) -> dict[str, Any]:
    cache_file = per_case_dir / f"{row_index}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_row = json.load(f)
            needs_backfill = not str(cached_row.get("response_final_answer", "")).strip()
            if needs_backfill:
                fill_response_fields(cached_row)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_row, f, ensure_ascii=False, indent=2)
            cached_output = str(cached_row.get("model_output", ""))
            if not cached_output.startswith("ERROR:") and len(cached_output) > 0:
                print("Found cache and skip!", row_index, cached_output)
                return cached_row
            logger.info("Cached row %s has ERROR model_output %s; re-running inference", row_index, cached_output)
        except Exception as e:
            logger.error("cache failed, ignore this")
            
    output_row = dict(row_data)
    question = str(output_row.get(question_column, "")).strip()
    image_path = resolve_image_path(image_dir, output_row, image_column)
    image_name = image_path.name

    if not question:
        output_row["model_output"] = "ERROR: empty question"
    elif not image_name:
        output_row["model_output"] = f"ERROR: image not found: {image_path}"
    else:
        for attempt in range(1, retries + 1):
            if image_path.exists():
                break
            logger.warning(
                "Image missing (attempt %s/%s), retrying after delay: %s",
                attempt,
                retries,
                image_path,
            )
            if attempt < retries:
                time.sleep(10)

        if not image_path.exists():
            output_row["model_output"] = f"ERROR: image not found: {image_path}"
        else:
            try:
                if crop_bbox_expand:
                    bbox_raw = output_row.get("bounding_box", "")
                    cropped_img = crop_image_by_bbox_expanded(image_path, bbox_raw, expand_ratio=bbox_expand_ratio)
                    crop_save_path = per_case_dir / f"{row_index}_crop_expand.jpg"
                    cropped_img.save(crop_save_path, format="JPEG")
                    base64_image = get_image_base64(crop_save_path, max_size_bytes)
                elif crop_bbox:
                    bbox_raw = output_row.get("bounding_box", "")
                    cropped_img = crop_image_by_bbox(image_path, bbox_raw)
                    crop_save_path = per_case_dir / f"{row_index}_crop.jpg"
                    cropped_img.save(crop_save_path, format="JPEG")
                    base64_image = get_image_base64(crop_save_path, max_size_bytes)
                else:
                    base64_image = get_image_base64(image_path, max_size_bytes)
                decoded_bytes = len(base64_image) * 3 // 4
                logger.info(
                    "Sending image %s | base64 chars: %d | ~decoded bytes: %d (%.2f MB)",
                    image_path,
                    len(base64_image),
                    decoded_bytes,
                    decoded_bytes / 1024 / 1024,
                )
                status, response_text = call_model_with_retry(
                    base64_image=base64_image,
                    question=question,
                    model_name=model_name,
                    retries=retries,
                    image_path=image_path,
                )
                if status == "success":
                    output_row["model_output"] = response_text
                else:
                    output_row["model_output"] = f"ERROR: {response_text}"
            except Exception as exc:  # noqa: BLE001
                output_row["model_output"] = f"ERROR: {exc}"

    fill_response_fields(output_row)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(output_row, f, ensure_ascii=False, indent=2)

    return output_row


def build_output_csv_name(model_name: str) -> str:
    normalized = model_name.replace("/", "_").replace("-", "_")
    return f"Pix2Fact_with_response_{normalized}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pix2Fact inference on OpenRouter with per-row cache.")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Input csv path")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing images")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--start_index", type=int, default=0, help="Start row index")
    parser.add_argument(
        "--retries",
        type=int,
        default=20,
        help="Retries: wait for local image file if missing, and retry each API call",
    )
    parser.add_argument("--question_column", default=DEFAULT_QUESTION_COLUMN, help="Question column name")
    parser.add_argument("--image_column", default=DEFAULT_IMAGE_COLUMN, help="Image path column name")
    parser.add_argument(
        "--max_image_mb",
        type=int,
        default=DEFAULT_MAX_IMAGE_MB,
        help="Max encoded image size in MB before resize (default: %(default)s)",
    )
    parser.add_argument(
        "--crop_bbox",
        action="store_true",
        help="If set, crop each image exactly to its bounding_box region before sending to the model",
    )
    parser.add_argument(
        "--crop_bbox_expand",
        action="store_true",
        help="If set, crop each image to its bounding_box region expanded outward by --bbox_expand_ratio of image size",
    )
    parser.add_argument(
        "--bbox_expand_ratio",
        type=float,
        default=0.15,
        help="Expansion ratio (fraction of image width/height) used with --crop_bbox_expand (default: 0.15)",
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "Missing OPENROUTER_API_KEY: set it in .env to your real key (not the <YOUR_OPENROUTER_API_KEY> placeholder)."
        )

    input_csv = Path(args.input_csv)
    image_dir = Path(args.image_dir)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR + "_" + args.model_name
    output_dir = Path(args.output_dir)
    per_case_dir = output_dir / f"cases_{args.model_name.replace('/', '_').replace('-', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    # df = df.head(5)
    if args.question_column not in df.columns:
        raise ValueError(f"Missing question column: {args.question_column}")
    if args.image_column not in df.columns:
        raise ValueError(f"Missing image column: {args.image_column}")

    max_size_bytes = max(1, args.max_image_mb) * 1024 * 1024

    if args.start_index > 0:
        df = df.iloc[args.start_index:].copy()

    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {}
        for row_index, row in df.iterrows():
            futures[
                executor.submit(
                    process_row,
                    row_index=row_index,
                    row_data=row.to_dict(),
                    image_dir=image_dir,
                    question_column=args.question_column,
                    image_column=args.image_column,
                    model_name=args.model_name,
                    retries=max(1, args.retries),
                    per_case_dir=per_case_dir,
                    max_size_bytes=max_size_bytes,
                    crop_bbox=args.crop_bbox,
                    crop_bbox_expand=args.crop_bbox_expand,
                    bbox_expand_ratio=args.bbox_expand_ratio,
                )
            ] = row_index

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            all_results.append(future.result())

    results_df = pd.DataFrame(all_results)
    if "index" in results_df.columns:
        results_df.sort_values("index", inplace=True)

    output_csv = output_dir / build_output_csv_name(args.model_name)
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Saved output csv: %s", output_csv)


if __name__ == "__main__":
    main()
