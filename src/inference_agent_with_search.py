"""Inference using the Agent tool-calling loop (SearchTool + VisitTool + TerminateTool).

Output format matches inference_genai_with_search_v2.py: per-row JSON cache and
a final merged CSV with the same response_* columns.

Environment variables:
  OPENAI_API_KEY     — API key (default: "EMPTY")
  OPENAI_BASE_URL    — base URL of the OpenAI-compatible endpoint
  OPENAI_MODEL_NAME  — model name / deployment ID
"""

from __future__ import annotations

import argparse
import ast
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
import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from src.agents import Agent, SearchTool, VisitTool
from src.prompt import PROMPT_TEMPLATE_TOOL_CALL

dotenv.load_dotenv()

DEFAULT_INPUT_CSV = "data/Pix2Fact_1k.csv"
DEFAULT_IMAGE_DIR = "data"
DEFAULT_OUTPUT_DIR = "outputs/pix2fact_eval_agent"
DEFAULT_QUESTION_COLUMN = "question"
DEFAULT_IMAGE_COLUMN = "local_image_path"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:34573/v1")
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen3.6-27B")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

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


def normalize_image_name(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return Path(text).name


def image_name_from_cdn_url(value: Any) -> str:
    url = str(value).strip()
    if not url or url.lower() == "nan":
        return ""
    parsed_url = urlparse(url)
    return Path(unquote(parsed_url.path)).name


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


# ---------------------------------------------------------------------------
# Output parsing helpers (mirrors fill_response_fields in other inference scripts)
# ---------------------------------------------------------------------------

def _to_list(val: Any) -> list:
    """Coerce a value to a list; wraps non-list scalars so callers always get a list."""
    if isinstance(val, list):
        return val
    if not val:
        return []
    return [str(val)]


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
        if alias.lower() in lowered_map:
            return lowered_map[alias.lower()]
    return default


def fill_response_fields(output_row: dict[str, Any]) -> None:
    """Populate response_* columns from model_output JSON (terminate tool args)."""
    parsed = parse_output_json(str(output_row.get("model_output", "")))
    output_row["response_observation"] = str(
        get_by_alias(parsed, "observation", "Observation", default="")
    )
    output_row["response_search_plan"] = json.dumps(
        _to_list(get_by_alias(parsed, "search_plan", "Search Plan", default=[])),
        ensure_ascii=False,
    )
    output_row["response_search_query"] = json.dumps(
        _to_list(get_by_alias(parsed, "search_query", "Search Query", default=[])),
        ensure_ascii=False,
    )
    output_row["response_comprehensive_answer"] = str(
        get_by_alias(parsed, "comprehensive_answer", "Comprehensive Answer", default="")
    )
    output_row["response_final_answer"] = str(
        get_by_alias(parsed, "final_answer", "Final Answer", "answer", default="")
    )
    output_row["response_is_error"] = str(output_row.get("model_output", "")).startswith("ERROR:")


# ---------------------------------------------------------------------------
# Agent output extraction
# ---------------------------------------------------------------------------

def extract_terminate_args(messages: list[dict[str, Any]]) -> str:
    """Return the JSON arguments string of the last terminate tool call in messages."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") == "terminate":
                return fn.get("arguments") or ""
    return ""


def build_trajectory(
    instruction: str,
    image_path: Path,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a clean trajectory dict from agent messages.

    Groups messages into rounds: each round is one assistant tool-call batch
    paired with the subsequent tool responses.  The user/system messages and
    any large base64 image blobs are excluded to keep files readable.
    """
    rounds: list[dict[str, Any]] = []

    # Index tool responses by tool_call_id for O(1) lookup.
    tool_responses: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tool_responses[msg.get("tool_call_id", "")] = msg.get("content", "")

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue

        calls = []
        responses = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "")
            try:
                args = json.loads(raw_args)
            except Exception:  # noqa: BLE001
                args = raw_args
            calls.append({"name": name, "arguments": args})
            tc_id = tc.get("id", "")
            if tc_id in tool_responses:
                responses.append({"tool_call_id": tc_id, "content": tool_responses[tc_id]})

        rounds.append({"tool_calls": calls, "tool_responses": responses})

    return {
        "instruction": instruction,
        "image": str(image_path),
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(
    row_index: int,
    row_data: dict[str, Any],
    image_dir: Path,
    question_column: str,
    image_column: str,
    model_name: str,
    retries: int,
    per_case_dir: Path,
    enable_search: bool = True,
    disable_search: bool = False,
    crop_bbox: bool = False,
    crop_bbox_expand: bool = False,
    bbox_expand_ratio: float = 0.15,
    max_steps: int = 20,
    token_budget: int = 32_768,
    image_token_budget: int = 1_024,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    cache_file = per_case_dir / f"{row_index}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_row = json.load(f)
        needs_backfill = not str(cached_row.get("response_final_answer", "")).strip()
        if needs_backfill:
            fill_response_fields(cached_row)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_row, f, ensure_ascii=False, indent=2)
        cached_output = str(cached_row.get("model_output", ""))
        if not cached_output.startswith("ERROR:") and len(cached_output) > 0:
            return cached_row
        logger.info("Cached row %s has ERROR model_output; re-running inference", row_index)

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
                attempt, retries, image_path,
            )
            if attempt < retries:
                time.sleep(10)

        if not image_path.exists():
            output_row["model_output"] = f"ERROR: image not found: {image_path}"
        else:
            try:
                effective_image: Path = image_path
                if crop_bbox_expand:
                    bbox_raw = output_row.get("bounding_box", "")
                    cropped_img = crop_image_by_bbox_expanded(image_path, bbox_raw, expand_ratio=bbox_expand_ratio)
                    crop_save_path = per_case_dir / f"{row_index}_crop_expand.jpg"
                    cropped_img.save(crop_save_path, format="JPEG")
                    effective_image = crop_save_path
                elif crop_bbox:
                    bbox_raw = output_row.get("bounding_box", "")
                    cropped_img = crop_image_by_bbox(image_path, bbox_raw)
                    crop_save_path = per_case_dir / f"{row_index}_crop.jpg"
                    cropped_img.save(crop_save_path, format="JPEG")
                    effective_image = crop_save_path

                if disable_search:
                    tools = []
                elif enable_search:
                    tools = [SearchTool(), VisitTool()]
                else:
                    tools = [VisitTool()]
                client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
                agent = Agent(
                    client=client,
                    deployment=model_name,
                    tools=tools,
                    system_prompt=PROMPT_TEMPLATE_TOOL_CALL,
                    max_steps=max_steps,
                    token_budget=token_budget,
                    image_token_budget=image_token_budget,
                    enable_thinking=enable_thinking,
                    verbose=True,
                )

                instruction = f"Input Question: {question}\nInput Image:"
                result = agent.run(instruction=instruction, image=effective_image)

                # Save trajectory separately (never mixed into the cache JSON).
                traj = build_trajectory(instruction, effective_image, result.messages)
                traj_file = per_case_dir / f"traj_{row_index}.json"
                with open(traj_file, "w", encoding="utf-8") as f:
                    json.dump(traj, f, ensure_ascii=False, indent=2)

                # Both "success" and "fail" are normal agent outcomes; only
                # exceptions or missing terminate calls are treated as errors.
                model_output = extract_terminate_args(result.messages)
                if not model_output:
                    # Agent returned a plain-text answer without calling terminate
                    model_output = result.answer
                output_row["model_output"] = model_output
            except Exception as exc:  # noqa: BLE001
                output_row["model_output"] = f"ERROR: {exc}"

    fill_response_fields(output_row)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(output_row, f, ensure_ascii=False, indent=2)

    return output_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_output_csv_name(model_name: str) -> str:
    normalized = model_name.replace("/", "_").replace("-", "_").replace(":", "_")
    return f"Pix2Fact_with_response_{normalized}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pix2Fact inference using the Agent tool-calling loop."
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing images")
    parser.add_argument("--output_dir", default=None, help="Output directory (auto-named if omitted)")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Model name / deployment ID")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel worker threads")
    parser.add_argument("--start_index", type=int, default=0, help="Skip this many rows from the top")
    parser.add_argument("--max_rows", type=int, default=0, help="If > 0, only run this many rows.")
    parser.add_argument(
        "--retries", type=int, default=20,
        help="Retries: wait for local image file if missing, and retry each API call",
    )
    parser.add_argument("--question_column", default=DEFAULT_QUESTION_COLUMN)
    parser.add_argument("--image_column", default=DEFAULT_IMAGE_COLUMN)
    parser.add_argument("--max_steps", type=int, default=20, help="Max agent tool-call rounds per row")
    parser.add_argument("--token_budget", type=int, default=120000, help="Token budget per agent run")
    parser.add_argument("--image_token_budget", type=int, default=5000, help="Image token budget")
    parser.add_argument(
        "--no-search", action="store_true", help="Disable SearchTool (agent uses VisitTool only)"
    )
    parser.add_argument(
        "--disable-search", action="store_true",
        help="Disable all search/visit tools; agent only has the terminate tool",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable thinking mode (chat_template enable_thinking=False)"
    )
    parser.add_argument(
        "--crop_bbox", action="store_true",
        help="Crop each image exactly to its bounding_box region before sending to the agent",
    )
    parser.add_argument(
        "--crop_bbox_expand", action="store_true",
        help="Crop each image to its bounding_box region expanded outward by --bbox_expand_ratio of image size",
    )
    parser.add_argument(
        "--bbox_expand_ratio", type=float, default=0.15,
        help="Expansion ratio (fraction of image width/height) used with --crop_bbox_expand (default: 0.15)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        model_slug = args.model_name.replace("/", "_").replace("-", "_").replace(":", "_")
        args.output_dir = f"{DEFAULT_OUTPUT_DIR}_{model_slug}"

    output_dir = Path(args.output_dir)
    model_slug = args.model_name.replace("/", "_").replace("-", "_").replace(":", "_")
    per_case_dir = output_dir / f"cases_{model_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input_csv)
    image_dir = Path(args.image_dir)

    df = pd.read_csv(input_csv)
    if args.question_column not in df.columns:
        raise ValueError(f"Missing question column: {args.question_column!r}")
    if args.image_column not in df.columns:
        raise ValueError(f"Missing image column: {args.image_column!r}")

    if args.start_index > 0:
        df = df.iloc[args.start_index:].copy()
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

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
                    enable_search=not args.no_search,
                    disable_search=args.disable_search,
                    crop_bbox=args.crop_bbox,
                    crop_bbox_expand=args.crop_bbox_expand,
                    bbox_expand_ratio=args.bbox_expand_ratio,
                    max_steps=args.max_steps,
                    token_budget=args.token_budget,
                    image_token_budget=args.image_token_budget,
                    enable_thinking=not args.no_thinking,
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
