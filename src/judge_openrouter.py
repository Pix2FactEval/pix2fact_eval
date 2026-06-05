import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import openai
import pandas as pd
from tqdm import tqdm
# --- Constants and Configuration ---
import dotenv

dotenv.load_dotenv()
# --- Constants and Configuration ---

# API Configuration (Azure first, then fallback for compatibility)
JUDGE_AZURE_OPENAI_API_KEY = os.getenv("JUDGE_AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
JUDGE_AZURE_OPENAI_ENDPOINT = os.getenv("JUDGE_AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
JUDGE_MODEL_NAME = (
    os.getenv("JUDGE_AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("JUDGE_MODEL_NAME")
    or "gemini-2.5-pro"
)
JUDGE_API_VERSION = os.getenv("JUDGE_AZURE_OPENAI_API_VERSION", "2024-02-01")
JUDGE_TT_LOGID = os.getenv("JUDGE_TT_LOGID", "${your_logid}")

BASE_URL = os.getenv("BASE_URL")
AK = os.getenv("AK")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", JUDGE_MODEL_NAME)

# Never commit real keys; set OPENROUTER_API_KEY in .env (see env.example).
OPENROUTER_API_KEY_PLACEHOLDER = "<YOUR_OPENROUTER_API_KEY>"
_raw_openrouter_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_KEY = (
    _raw_openrouter_key
    if (_raw_openrouter_key and _raw_openrouter_key.strip() and _raw_openrouter_key != OPENROUTER_API_KEY_PLACEHOLDER)
    else None
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

judge_model_client = (
    openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    if OPENROUTER_API_KEY
    else None
)

ACTIVE_MODEL_NAME = JUDGE_MODEL_NAME


# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JudgeCacheStats:
    """Thread-safe counts for on-disk judge cache lookups inside call_judge_model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1


# Judge Prompt Template
JUDGE_PROMPT_TEMPLATE = """You are a strict judge. Compare a Ground Truth answer vs a Model answer.

Rules:
1) Output ONLY one token: True or False (case-sensitive, no punctuation, no space, no code fences).
2) True if and only if the Model answer semantically matches the Ground Truth with respect to meaning and exact factual content.
   - Numbers/dates/names must match.
   - Language or casing differences are acceptable if meaning is identical.
   - If Ground Truth is '[NO_DEFINITIVE_ANSWER]', output True only if Model answer is exactly '[NO_DEFINITIVE_ANSWER]'.
3) If uncertain for any reason, output False.

Ground Truth: {ground_truth}
Model Answer: {model_answer}
"""

JUDGE_PROMPT_TEMPLATE_v2 = """You are an answer consistency judge. Your task is to determine whether a model's answer is substantively correct when compared to the ground truth answer.

## Input Format

Question: <the original question>
Ground Truth: <ground truth answer>
Model Final Answer: <model's final answer>
Model Reasoning: <model's comprehensive answer / reasoning process>

## Judgment Criteria

Judge as CORRECT if:
- The answer conveys the same factual information as the ground truth, even if phrased differently
- Minor formatting differences (e.g., "12:00 PM" vs "12 pm", "1-1-6" vs "1 Chome-1-6")
- Different but equivalent representations (e.g., "11.75" vs "11 hours and 45 minutes")
- Additional detail that does not contradict the ground truth (e.g., a specific branch name when the ground truth is a brand name)
- The Final Answer is incomplete or oddly phrased, but the Reasoning clearly arrives at the correct answer

Judge as INCORRECT if:
- The factual content differs from the ground truth (wrong number, wrong name, wrong address, etc.)
- The model outputs "NO_DEFINITIVE_ANSWER" or equivalent when the ground truth has a specific answer
- The model's reasoning reveals a flawed assumption that led to a wrong answer (e.g., wrong currency conversion, incorrect exclusion logic), even if the Final Answer looks superficially close
- The answer is partially correct but missing information that materially changes the meaning

## Output Format

Respond with a JSON object:
{
  "judgment": "CORRECT" or "INCORRECT",
  "reason": "<one sentence explaining your judgment, referencing the question context or reasoning if relevant>"
}

Question: <<QUESTION>>
Ground Truth: <<GROUND_TRUTH>>
Model Final Answer: <<MODEL_ANSWER>>
Model Reasoning: <<MODEL_REASONING>>
"""

JUDGE_PROMPT_TEMPLATE_v3 = """You are a STRICT semantic judge. Your default verdict is False.
You output True only when the Model answer is, beyond any reasonable doubt, the SAME factual answer as the Ground Truth.

OUTPUT FORMAT (HARD RULE):
- Output ONLY one token: True or False
- Case-sensitive. No punctuation, whitespace, quotes, code fences, or explanation.

================================================================
CORE PRINCIPLE
================================================================
- Default to False.
- Output True ONLY if every factual element in the Ground Truth is present and unchanged in the Model answer, AND the Model answer introduces no factual element that conflicts with or extends beyond the Ground Truth.
- Surface-form differences are tolerated ONLY in the narrow, enumerated cases below (E1–E8). Anything not explicitly listed as equivalent is a mismatch.
- Any ambiguity, any uncertainty, any "probably the same" → False.

================================================================
NARROW EQUIVALENCE RULES (the ONLY tolerated differences)
================================================================

E1) Pure formatting of the SAME number:
    - Thousands separators: "1000" ≡ "1,000"
    - Trailing zeros in decimals: "73%" ≡ "73.00%"
    - Digit grouping in phone numbers (see E4)
    NOT equivalent: rounded vs exact ("73%" ≠ "73.4%"), different precision, different units.

E2) Mathematically identical values across notations:
    - "11.75 hours" ≡ "11 hours 45 minutes"
    - "6,153 million" ≡ "6,153,000,000"
    - "1.5 billion" ≡ "1,500 million"
    Values must be EXACTLY equal after conversion. Approximations are NOT equivalent.

E3) Currency / unit symbols may be omitted on ONE side ONLY when the question itself specifies the unit:
    - Question asks "in USD" → "$187 million" ≡ "187 million"
    - Question asks "in feet" → "20 feet" ≡ "20"
    - Question asks "how many properties" → "35" ≡ "35 properties"
    If the question does not lock the unit, different units are NOT equivalent.

E4) Phone numbers — equivalent iff the digit sequences match after this normalization:
    - Strip spaces, dashes, parentheses, dots, and a leading "+".
    - Strip a leading country-calling-code IF AND ONLY IF the remaining number begins with the corresponding national trunk prefix (e.g. "+44 20..." ≡ "020...", "+81 3-..." ≡ "03-...", "+65 6536 6739" ≡ "6536 6739").
    - After normalization, digit strings must be IDENTICAL. Even one different digit → False.
    - Mnemonic letter codes (e.g. "1-855-TTY-KORS") must be decoded to digits before comparison.

E5) Addresses — equivalent iff ALL of the following hold:
    - Same street number, same street name, same locality, same postal code (when either side gives one).
    - Tolerated cosmetic differences ONLY:
        * Street-type abbreviations: Street/St, Boulevard/Blvd, Avenue/Ave, Road/Rd, Drive/Dr, Lane/Ln
        * Unit notation: "B1-10" ≡ "#B1-10" ≡ "Unit B1-10"
        * Punctuation, casing, and reordering of components
        * Country/region suffix on one side only
        * Adding the building/mall NAME at the same address
    - NOT equivalent:
        * Different street name, number, or postcode → False
        * Street address vs different venue name at that address → False
        * Different branch of the same brand → False

E6) Names / entities — equivalent iff they refer to the SAME specific real-world entity:
    - Same entity in different scripts/languages: equivalent
    - Brand prefix added to a model/product name: equivalent ("Soul" ≡ "Kia Soul")
    - Neutral category descriptor that does not change the referent: equivalent ("Veuve Monsigny" ≡ "Veuve Monsigny Champagne Brut")
    - NOT equivalent:
        * Different specific product/model/variant/edition → False
        * Qualifier that points to a DIFFERENT entity → False
        * Sub-type vs broader type when the question requires the specific form → False
        * When in doubt → False

E7) Polite wrappers may be ignored on the Model side:
    - Strippable: leading "Yes,", "No,", "Sure,", "The answer is", "It is"
    - The factual core after stripping must still satisfy E1–E6.
    - This rule does NOT permit stripping factual qualifiers like "Up to", "At least", "Approximately", "Around", "More than", "Less than" — see M-LIMITS.

E8) Casing and punctuation differences alone never determine the verdict.

================================================================
EXPLICIT MISMATCHES (always False)
================================================================

M-RANGE) Range vs list vs alternation are NEVER equivalent, even when they enumerate the same items:
    - "Monday to Tuesday" (range) ≠ "Monday and Tuesday" (list)
    - "Monday through Friday" ≠ "Monday, Tuesday, Wednesday, Thursday, Friday"
    - "A or B" ≠ "A and B" ≠ "A to B"
    - "9am-5pm" ≠ "9am and 5pm"
    The connective type MUST match the Ground Truth's connective type.

M-LIMITS) Bounding qualifiers change meaning and are NOT strippable by default:
    - "20" ≠ "Up to 20" ≠ "At least 20" ≠ "More than 20" ≠ "Approximately 20" ≠ "Around 20"

    Exception (allowed equivalence) — ALL of the following must hold:
      (a) The GT is a bare value (no qualifier of its own).
      (b) The question text contains a word or phrase whose semantics MATCH the qualifier on the Model side:
          * Upper-bound cues for "Up to" / "At most" / "Maximum":
              "maximum", "max", "up to", "peak", "highest", "ceiling", "capacity", "limit",
              "range" (when asking for the range of a single-direction metric like airflow, speed, distance, output)
          * Lower-bound cues for "At least" / "Minimum":
              "minimum", "min", "at least", "starting from", "lowest", "floor"
          * Approximation cues for "Approximately" / "Around" / "About":
              "approximately", "about", "roughly", "around"
      (c) The qualifier direction on the Model side must be consistent with (b).

    Examples:
      - Q: "max airflow?", GT="20", Model="Up to 20" → True (matches "max")
      - Q: "range of airflow?", GT="20", Model="Up to 20" → True ("range" of single-direction metric)
      - Q: "what is the airflow?", GT="20", Model="Up to 20" → False (no bound cue)
      - Q: "starting price?", GT="500", Model="At least 500" → True (matches "starting")
      - Q: "what is the price?", GT="500", Model="Approximately 500" → False (no approximation cue)

    When the question's bound semantics are unclear → False.

M-TIME) Time of day requires AM/PM (or 24-hour form) to match:
    - "5:00" ≡ "5:00 AM" only if the question or context explicitly fixes the period.
    - "17:00" ≡ "5:00 PM"; "05:00" ≡ "5:00 AM".

M-NUMBER) Different numeric values are never equivalent:
    - Even one differing digit, decimal place, or sign → False.
    - Rounded vs exact → False, unless the question explicitly requested that precision.

M-EXTRA-FACT) The Model answer must not introduce extra factual claims:
    - Adding a NEUTRAL descriptor of the same referent: allowed.
    - Adding a NEW factual claim (different specs, additional features, different location/time): → False.
    - When in doubt about whether an addition is neutral → False.

M-MISSING-FACT) The Model answer must contain every factual element the GT contains:
    - If GT lists multiple items and Model gives only some → False.
    - If GT specifies a qualifier and Model omits it → False, unless the broader term clearly refers to the same specific entity in context.

M-NO-DEFINITIVE) If GT is exactly "[NO_DEFINITIVE_ANSWER]":
    - Output True only if Model answer is also exactly "[NO_DEFINITIVE_ANSWER]".

M-UNCERTAIN) Any residual uncertainty after applying all rules → False.

================================================================
INTERNAL PROCEDURE (do NOT output any of this)
================================================================

Step 1 — Normalize:
    Strip Model's polite wrappers (E7).
    Apply notation/format normalization for the relevant rule type (E1–E6).

Step 2 — Check explicit mismatches (M-*):
    If ANY M-* condition is triggered → verdict is False, skip to Step 4.

Step 3 — Confidence self-check:
    Silently assess your confidence that the two answers refer to the SAME fact:
        - HIGH      : All elements match unambiguously under E1–E8; no extra/missing facts; no plausible alternative interpretation.
        - MEDIUM    : The answers look similar but at least one of the following is true:
                      * an equivalence rule applies only "by analogy" rather than directly,
                      * the Model adds or omits a descriptor whose neutrality is not obvious,
                      * the connective/qualifier/unit alignment requires a judgment call,
                      * you would want to "give it the benefit of the doubt".
        - LOW       : Clear differences in entity, number, scope, qualifier, or connective.
    Mapping to verdict:
        - HIGH   → True
        - MEDIUM → False  (do NOT give the benefit of the doubt)
        - LOW    → False

Step 4 — Output:
    Emit exactly one token: True or False. Nothing else.
    

================================================================
Input
================================================================   
Question: <<QUESTION>>
Ground Truth: <<GROUND_TRUTH>>
Model Final Answer: <<MODEL_ANSWER>>
Model Reasoning: <<MODEL_REASONING>>
"""
# --- API Interaction ---

def judge_cache_key(
    judge_model_name: str,
    ground_truth: str,
    model_answer: str,
    *,
    prompt_version: str = "v1",
    question: Optional[str] = None,
    model_reasoning: Optional[str] = None,
) -> str:
    # v1 keeps the legacy payload so existing cache files remain valid.
    if prompt_version == "v1":
        payload_obj: Dict[str, Any] = {
            "model": judge_model_name,
            "gt": ground_truth,
            "ans": model_answer,
        }
    else:
        payload_obj = {
            "model": judge_model_name,
            "gt": ground_truth,
            "ans": model_answer,
            "pv": prompt_version,
            "q": question or "",
            # "reason": model_reasoning or "",
            # Bump when cached payload shape changes (e.g. store judge reason in cache file).
            "cks": "3",
        }
    payload = json.dumps(payload_obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_judge_cache(cache_dir: Path, key: str) -> Optional[Tuple[str, str]]:
    """Return (judged, judge_reason) if cache hit; judge_reason may be empty."""
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        val = data.get("judged")
        if val in ("True", "False"):
            reason_val = data.get("reason")
            reason_out = str(reason_val) if reason_val is not None else ""
            return str(val), reason_out
    except Exception:  # noqa: BLE001
        logger.warning("Invalid judge cache entry, ignoring: %s", path)
    return None


def write_judge_cache(cache_dir: Path, key: str, judged: str, reason: str = "") -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    tmp = cache_dir / f"{key}.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"judged": judged, "reason": reason}, f, ensure_ascii=False)
    tmp.replace(path)


def parse_v2_judgment_response(raw: str) -> Optional[Tuple[str, str]]:
    """Parse v2 JSON judgment; return (judged_token, reason) or None if invalid."""
    text = raw.strip()
    try:
        parsed = json.loads(normalize_json(text))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    judgment = parsed.get("judgment")
    reason_raw = parsed.get("reason", "")
    reason = str(reason_raw) if reason_raw is not None else ""
    if judgment == "CORRECT":
        return "True", reason
    if judgment == "INCORRECT":
        return "False", reason
    return None


def parse_bool_verdict_token(raw: str) -> Optional[str]:
    """Parse v1/v3 single-token True/False verdict. Returns 'True', 'False', or None."""
    text = (raw or "").strip()
    if text in ("True", "False"):
        return text
    # First line only if the model appends a newline
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if first_line in ("True", "False"):
            return first_line
    # First whitespace-delimited token
    parts = text.split()
    if parts and parts[0] in ("True", "False"):
        return parts[0]
    return None


def call_judge_model(
    ground_truth: Any,
    model_answer: Any,
    max_retries: int = 5,
    *,
    judge_model_name: str,
    cache_dir: Optional[Path] = None,
    cache_stats: Optional[JudgeCacheStats] = None,
    prompt_version: str = "v1",
    question: str = "",
    model_reasoning: str = "",
) -> Tuple[str, str, str]:
    """Calls the judge model. Returns (status, judged, judge_reason).

    judge_reason is non-empty only for v2 JSON judge; v1 and v3 return ''.
    """
    if not judge_model_client:
        logger.error("Judge model client not initialized.")
        return "error", "Client not initialized", ""

    ground_truth_str = str(ground_truth)
    model_answer_str = str(model_answer)

    cache_key: Optional[str] = None
    if cache_dir is not None:
        cache_key = judge_cache_key(
            judge_model_name,
            ground_truth_str,
            model_answer_str,
            prompt_version=prompt_version,
            question=question if prompt_version != "v1" else None,
            model_reasoning=model_reasoning if prompt_version != "v1" else None,
        )
        cached = read_judge_cache(cache_dir, cache_key)
        if cached is not None:
            if cache_stats is not None:
                cache_stats.record_hit()
            c_judged, c_reason = cached
            return "success", c_judged, c_reason
        if cache_stats is not None:
            cache_stats.record_miss()

    if prompt_version == "v2":
        query = (
            JUDGE_PROMPT_TEMPLATE_v2
            .replace("<<QUESTION>>", str(question))
            .replace("<<GROUND_TRUTH>>", ground_truth_str)
            .replace("<<MODEL_ANSWER>>", model_answer_str)
            .replace("<<MODEL_REASONING>>", str(model_reasoning))
        )
        max_out_tokens = 1024
    elif prompt_version == "v3":
        query = (
            JUDGE_PROMPT_TEMPLATE_v3
            .replace("<<QUESTION>>", str(question))
            .replace("<<GROUND_TRUTH>>", ground_truth_str)
            .replace("<<MODEL_ANSWER>>", model_answer_str)
            .replace("<<MODEL_REASONING>>", str(model_reasoning))
        )
        # v3 asks for a single-token True/False verdict; keep completion short.
        max_out_tokens = 16000
    else:
        query = JUDGE_PROMPT_TEMPLATE.format(ground_truth=ground_truth_str, model_answer=model_answer_str)
        max_out_tokens = 500
    messages = [{"role": "user", "content": query}]
    for attempt in range(max_retries):
        try:
            completion = judge_model_client.chat.completions.create(
                model=judge_model_name,
                messages=messages,
                max_tokens=max_out_tokens,
                temperature=0.0,
                stream=False,
            )
            if completion.choices and completion.choices[0].message.content:
                response = completion.choices[0].message.content.strip()
                if prompt_version == "v2":
                    parsed_v2 = parse_v2_judgment_response(response)
                    if parsed_v2 is not None:
                        judged, judge_reason = parsed_v2
                        if cache_dir is not None and cache_key is not None:
                            write_judge_cache(cache_dir, cache_key, judged, judge_reason)
                        return "success", judged, judge_reason
                    logger.warning(
                        "Attempt %s: Unexpected v2 response: %r. Retrying.",
                        attempt + 1,
                        response[:500] if len(response) > 500 else response,
                    )
                elif prompt_version in ("v1", "v3"):
                    verdict = parse_bool_verdict_token(response)
                    if verdict is not None:
                        if cache_dir is not None and cache_key is not None:
                            write_judge_cache(cache_dir, cache_key, verdict, "")
                        return "success", verdict, ""
                    logger.warning(
                        "Attempt %s: Unexpected %s response: %r. Retrying.",
                        attempt + 1,
                        prompt_version,
                        response[:500] if len(response) > 500 else response,
                    )
                else:
                    logger.warning(f"Attempt {attempt + 1}: Unexpected response: '{response}'. Retrying.")
            else:
                logger.warning(f"Attempt {attempt + 1}: No response from model. Retrying.")
            
            time.sleep(5)

        except openai.RateLimitError as e:
            logger.warning(f"Attempt {attempt + 1}: Rate limit error: {e}. Retrying in 10s.")
            time.sleep(10)
        except Exception as e:
            error_text = str(e)
            logger.error(
                "Attempt %s: judge call failed: %s | messages=%s",
                attempt + 1,
                error_text,
                sanitize_messages_for_log(messages),
            )
            if is_non_retryable_error(error_text):
                return "error", error_text, ""
            time.sleep(5)
            
    return "error", f"Failed after {max_retries} attempts", ""


def sanitize_messages_for_log(messages: Any) -> str:
    text = json.dumps(messages, ensure_ascii=False)
    # Mask any inline base64 image payloads if they ever appear in messages.
    text = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "data:image/<redacted>;base64,<BASE64_IMAGE_PLACEHOLDER>",
        text,
    )
    return text


def is_non_retryable_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "error code: 403" in lowered
        or "unauthorized_access_rejected" in lowered
        or '"code":"4003"' in lowered
        or "request is rejected by operations gateway" in lowered
    )

def normalize_json(json_str):
    if "```json" in json_str:
        json_str = json_str.replace("```json", "")
    if "```" in json_str:
        json_str = json_str.replace("```", "")
    
    index = json_str.rfind("{")
    if index != -1:
        json_str = json_str[index:].strip()
    index = json_str.rfind("}")
    if index != -1:
        json_str = json_str[:index+1].strip()
    return json_str


def extract_model_components(model_output: Any) -> Tuple[str, str, Optional[str]]:
    """Parse CSV 'model_output' JSON: Final Answer and optional Comprehensive Answer.

    Returns (final_answer, comprehensive_answer, error_message). error_message set => skip judge.
    """
    if not isinstance(model_output, str):
        return "Failed to answer", "", "model_output is not a string"
    if not model_output.strip():
        return "Failed to answer", "", "model_output is empty"
    if model_output.startswith("ERROR:"):
        return "Failed to answer", "", model_output

    try:
        parsed = json.loads(normalize_json(model_output))
        if isinstance(parsed, dict):
            final_val: Optional[str] = None
            if "Final Answer" in parsed:
                final_val = str(parsed.get("Final Answer"))
            else:
                for k in ("Final", "answer", "final_answer"):
                    if k in parsed:
                        final_val = str(parsed.get(k))
                        break
            if final_val is not None:
                comp = ""
                if "Comprehensive Answer" in parsed and parsed.get("Comprehensive Answer") is not None:
                    comp = str(parsed.get("Comprehensive Answer"))
                return final_val, comp, None
        return "Failed to answer", "", "parsed json but missing Final Answer"
    except Exception:
        return "Failed to answer", "", "failed to parse model_output as json"


def extract_model_answer(model_output: Any) -> Tuple[str, Optional[str]]:
    """Extract a best-effort model answer string from the CSV 'model_output' field.

    Returns (answer, error_message). If error_message is not None, extraction was best-effort.
    """
    final, _comp, err = extract_model_components(model_output)
    return final, err


def judge_one(
    index: int,
    gt: Any,
    model_output: Any,
    question: str,
    max_retries: int,
    judge_model_name: str,
    cache_dir: Optional[Path],
    prompt_version: str = "v1",
    cache_stats: Optional[JudgeCacheStats] = None,
) -> Tuple[int, str, str]:
    """Judge a single row and return (row_index, judged_token, judge_reason)."""
    model_answer, comprehensive, err = extract_model_components(model_output)
    # Skip judge API call for rows that are clearly invalid.
    if err is not None:
        return index, "False", ""
    gt_text = str(gt).strip()
    model_text = str(model_answer).strip()
    if gt_text == model_text:
        return index, "True", ""
    status, judged, judge_reason = call_judge_model(
        gt,
        model_answer,
        max_retries=max_retries,
        judge_model_name=judge_model_name,
        cache_dir=cache_dir,
        cache_stats=cache_stats,
        prompt_version=prompt_version,
        question=question,
        model_reasoning=comprehensive,
    )
    if status != "success":
        # Treat any error as incorrect for accuracy computation.
        return index, "False", ""
    return index, judged, judge_reason


def pick_ground_truth(row: pd.Series) -> Any:
    if "answer" in row and pd.notna(row.get("answer")):
        return row.get("answer")
    if "[Final]answer" in row and pd.notna(row.get("[Final]answer")):
        return row.get("[Final]answer")
    return ""


def pick_model_output(row: pd.Series) -> Any:
    # Prefer parsed field from new inference script if available.
    if "response_final_answer" in row and pd.notna(row.get("response_final_answer")):
        final_answer = str(row.get("response_final_answer")).strip()
        if final_answer:
            out: Dict[str, str] = {"Final Answer": final_answer}
            if "response_comprehensive_answer" in row and pd.notna(row.get("response_comprehensive_answer")):
                ca = str(row.get("response_comprehensive_answer")).strip()
                if ca:
                    out["Comprehensive Answer"] = ca
            return json.dumps(out, ensure_ascii=False)
    return row.get("model_output", None)


def pick_question(row: pd.Series) -> str:
    if "question" in row and pd.notna(row.get("question")):
        return str(row.get("question"))
    if "[Final]question" in row and pd.notna(row.get("[Final]question")):
        return str(row.get("[Final]question"))
    return ""


def main():
    parser = argparse.ArgumentParser(description="Judge Pix2Fact Model Outputs")
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file from infer_pix2fact.py')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--max_retries', type=int, default=50, help='Max retries for judge model calls')
    parser.add_argument('--max_rows', type=int, default=0, help='If > 0, only judge the first N rows (for smoke tests)')
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/judge_cache",
        help="Directory for judge API result cache (same gt+answer+model skips API). Empty string disables.",
    )
    parser.add_argument(
        "--prompt_version",
        choices=["v1", "v2", "v3"],
        default="v3",
        help="v1: True/False token judge (GT vs model answer only). "
        "v2: JSON judge (question + Final Answer + Comprehensive Answer). "
        "v3: strict True/False token judge with question + reasoning (same inputs as v2, separate cache).",
    )
    args = parser.parse_args()

    judge_model_name = args.model_name if args.model_name is not None else ACTIVE_MODEL_NAME
    cache_dir: Optional[Path] = Path(args.cache_dir) if args.cache_dir else None
    cache_stats: Optional[JudgeCacheStats] = JudgeCacheStats() if cache_dir else None
    logger.info(f"Loading data from {args.input_csv}")

    logger.info("Using judge model: %s, prompt_version: %s", judge_model_name, args.prompt_version)
    full_df = pd.read_csv(args.input_csv)
    if args.max_rows and args.max_rows > 0:
        full_df = full_df.head(args.max_rows).copy()

    # Pre-fill judged column to keep output aligned with input ordering
    full_df["judged"] = "False"
    # Judge JSON "reason" — use judge_reason if input already has a "reason" column
    reason_out_col = "judge_reason" if "reason" in full_df.columns else "reason"
    full_df[reason_out_col] = ""

    # Submit judging tasks
    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, row in full_df.iterrows():
            gt = pick_ground_truth(row)
            model_output = pick_model_output(row)
            question = pick_question(row)
            futures[
                executor.submit(
                    judge_one,
                    int(idx),
                    gt,
                    model_output,
                    question,
                    args.max_retries,
                    judge_model_name,
                    cache_dir,
                    args.prompt_version,
                    cache_stats,
                )
            ] = int(idx)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            row_index, judged, judge_reason = future.result()
            full_df.at[row_index, "judged"] = judged
            full_df.at[row_index, reason_out_col] = judge_reason

    # Compute accuracy
    judged_series = full_df["judged"].astype(str)
    total = int(len(judged_series))
    correct = int((judged_series == "True").sum())
    accuracy = (correct / total) if total else 0.0
    judge_model_name_to_save = judge_model_name.replace(":", "_").replace("/", "_")
    out_csv = args.input_csv.replace(".csv", f"_judged_by_{judge_model_name_to_save}.csv")
    logger.info("Writing judged CSV to %s", out_csv)
    full_df.to_csv(out_csv, index=False)

    logger.info("Final Accuracy: %.6f", accuracy)
    logger.info("Correct: %d, Total: %d", correct, total)
    if cache_stats is not None:
        ch, cm = cache_stats.hits, cache_stats.misses
        lookups = ch + cm
        if lookups > 0:
            hit_rate = ch / lookups
            logger.info(
                "Judge cache: hits=%d, misses=%d, hit_rate=%.6f (%.2f%%)",
                ch,
                cm,
                hit_rate,
                100.0 * hit_rate,
            )
        else:
            logger.info("Judge cache: no lookups (all rows skipped API judge or cache unused)")
    else:
        logger.info("Judge cache: disabled (empty --cache_dir)")

    # Keep existing behavior of writing a small summary json
    os.makedirs("data", exist_ok=True)
    with open("data/result.json", "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total}, f)


if __name__ == "__main__":
    main()

    
