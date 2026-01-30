      
import json
import openai
import pandas as pd
import argparse
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
# --- Constants and Configuration ---
import dotenv

dotenv.load_dotenv()
# --- Constants and Configuration ---

# API Configuration
BASE_URL = os.getenv("BASE_URL")
AK = os.getenv("AK")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
print(BASE_URL, AK, DEFAULT_MODEL_NAME)
# Initialize OpenAI Client
# Initialize OpenAI Client
judge_model_client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=AK,
)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# --- API Interaction ---

def call_judge_model(ground_truth: Any, model_answer: Any, max_retries: int = 5) -> Tuple[str, str]:
    """Calls the judge model to compare answers."""
    if not judge_model_client:
        logger.error("Judge model client not initialized.")
        return "error", "Client not initialized"

    ground_truth_str = str(ground_truth)
    model_answer_str = str(model_answer)

    query = JUDGE_PROMPT_TEMPLATE.format(ground_truth=ground_truth_str, model_answer=model_answer_str)
    print("Judge:\n", query)
    for attempt in range(max_retries):
        try:
            completion = judge_model_client.chat.completions.create(
                model=DEFAULT_MODEL_NAME,
                messages=[{"role": "user", "content": query}],
                max_tokens=1024,
                temperature=1.0,
            )
            if completion.choices and completion.choices[0].message.content:
                response = completion.choices[0].message.content.strip()
                if response in ["True", "False"]:
                    return "success", response
                else:
                    logger.warning(f"Attempt {attempt + 1}: Unexpected response: '{response}'. Retrying.")
            else:
                logger.warning(f"Attempt {attempt + 1}: No response from model. Retrying.")
            
            time.sleep(5)

        except openai.RateLimitError as e:
            logger.warning(f"Attempt {attempt + 1}: Rate limit error: {e}. Retrying in 10s.")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")
            time.sleep(5)
            
    return "error", f"Failed after {max_retries} attempts"

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


def extract_model_answer(model_output: Any) -> Tuple[str, Optional[str]]:
    """Extract a best-effort model answer string from the CSV 'model_output' field.

    Returns (answer, error_message). If error_message is not None, extraction was best-effort.
    """
    if not isinstance(model_output, str):
        return "Failed to answer", "model_output is not a string"

    try:
        parsed = json.loads(normalize_json(model_output))
        if isinstance(parsed, dict):
            if "Final Answer" in parsed:
                return str(parsed.get("Final Answer")), None
            # Some older runs may use different keys
            for k in ("Final", "answer", "final_answer"):
                if k in parsed:
                    return str(parsed.get(k)), "nonstandard key"
        return "Failed to answer", "parsed json but missing Final Answer"
    except Exception:
        # If the model output is not valid JSON, treat as failed.
        print("Parsed failed!!!", model_output)
        return "Failed to answer", "failed to parse model_output as json"


def judge_one(index: int, gt: Any, model_output: Any, max_retries: int) -> Tuple[int, str]:
    """Judge a single row and return (row_index, judged_token)."""
    model_answer, _err = extract_model_answer(model_output)
    status, response = call_judge_model(gt, model_answer, max_retries=max_retries)
    if status != "success":
        # Treat any error as incorrect for accuracy computation.
        return index, "False"
    return index, response

def main():
    parser = argparse.ArgumentParser(description="Judge Pix2Fact Model Outputs")
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file from infer_pix2fact.py')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--max_retries', type=int, default=5, help='Max retries for judge model calls')
    parser.add_argument('--max_rows', type=int, default=0, help='If > 0, only judge the first N rows (for smoke tests)')
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input_csv}")
    full_df = pd.read_csv(args.input_csv)
    if args.max_rows and args.max_rows > 0:
        full_df = full_df.head(args.max_rows).copy()

    # Pre-fill judged column to keep output aligned with input ordering
    full_df["judged"] = "False"

    # Submit judging tasks
    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, row in full_df.iterrows():
            gt = row.get('[Final]answer', "")
            model_output = row.get('model_output', None)
            futures[executor.submit(judge_one, int(idx), gt, model_output, args.max_retries)] = int(idx)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            row_index, judged = future.result()
            full_df.at[row_index, "judged"] = judged

    # Compute accuracy
    judged_series = full_df["judged"].astype(str)
    total = int(len(judged_series))
    correct = int((judged_series == "True").sum())
    accuracy = (correct / total) if total else 0.0

    out_csv = args.input_csv.replace(".csv", "_judged.csv")
    logger.info("Writing judged CSV to %s", out_csv)
    full_df.to_csv(out_csv, index=False)

    logger.info("Final Accuracy: %.6f", accuracy)
    logger.info("Correct: %d, Total: %d", correct, total)

    # Keep existing behavior of writing a small summary json
    os.makedirs("data", exist_ok=True)
    with open("data/result.json", "w") as f:
        json.dump({"accuray": accuracy, "correct": correct, "total": total}, f)


if __name__ == "__main__":
    main()

    
