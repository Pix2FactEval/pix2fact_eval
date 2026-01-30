      
import sys
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import io
import json
import logging
import os
from pathlib import Path
import time

import dotenv
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from PIL import Image

sys.path.insert(0, '.')
from src.prompt import PROMPT_TEMPLATE

dotenv.load_dotenv()
# --- Constants and Configuration ---

# API Configuration
BASE_URL = os.getenv("BASE_URL")
AK = os.getenv("AK")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
print(BASE_URL, AK, DEFAULT_MODEL_NAME)
# Initialize OpenAI Client
# Initialize OpenAI Client
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=AK,
)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# LIMIT to 18 MB
MAX_SIZE_BYTES = 10 * 1024 * 1024
# --- API Interaction ---

def checkQA(base64_image, question, model_name, max_retries=3, initial_max_tokens=4096):
    """Calls the API for question answering with retry logic."""
    max_tokens = initial_max_tokens
    
    for attempt in range(max_retries):
        try:
            full_prompt = f"{PROMPT_TEMPLATE}\nInput Question: {question}\nInput Image: \n"
            
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]}
            ]
        
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1.0 if model_name.startswith("gpt-5") else 0.0,
            )

            logger.info(f"API call successful, attempt {attempt + 1}/{max_retries}")
            logger.info(f"Response Message: {response}")
            logger.info(f"Response Finish Reason: {response.choices[0].finish_reason}")

            if response.choices is None or len(response.choices) == 0:
                error_msg = 'No response from API'
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
                time.sleep(10)
                continue
                
            if response.choices[0].finish_reason == 'length':
                error_msg = f'Output over max_token {max_tokens}'
                max_tokens *= 2
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}, increasing max_tokens to {max_tokens}")
                continue
                
            return 'success', response.choices[0].message.content
            
        except openai.RateLimitError as e:
            logger.warning(f'Attempt {attempt + 1}/{max_retries}: Rate limit error - {e}')
            time.sleep(10)
            
        except openai.OpenAIError as e:
            error_msg = str(e)
            if "maximum context length" in error_msg:
                logger.error(f"Context length exceeded, skipping - {error_msg}")
                return 'error', error_msg
            logger.warning(f'Attempt {attempt + 1}/{max_retries}: OpenAI API error - {e}')
            time.sleep(10)
            
        except Exception as e:
            logger.warning(f'Attempt {attempt + 1}/{max_retries}: Unexpected error - {e}')
            time.sleep(10)
    
    error_msg = f"Failed after {max_retries} attempts"
    logger.error(error_msg)
    return 'error', error_msg

# --- Data Processing ---
def get_image_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    if len(image_data) <= MAX_SIZE_BYTES:
            return base64.b64encode(image_data).decode("utf-8")
    img = Image.open(io.BytesIO(image_data))
    img_format = img.format if img.format else 'JPEG'
    output_buffer = io.BytesIO()
    quality = 85 
    while True:
        output_buffer.seek(0)
        output_buffer.truncate(0)
        if img_format.upper() == 'PNG':
            img.save(output_buffer, format=img_format)
        else:
            img.save(output_buffer, format=img_format, quality=quality)
        current_size = output_buffer.tell()
        logger.info(f"Current image size after compress: {current_size}")
        if current_size <= MAX_SIZE_BYTES:
            break
        # 如果还是太大，进行缩放
        width, height = img.size
        new_width = int(width * 0.9)
        new_height = int(height * 0.9)
        # 防止图片缩得太小
        if new_width < 100 or new_height < 100:
            break
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # 4. 编码为 Base64
    base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return base64_image

def process_data(row, model_name, output_file_name):
    """Processes a single data row."""
    if output_file_name.exists():
        with open(output_file_name, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(output_file_name)
                raise ValueError(str(output_file_name))
            if 'model_output' in data and not data["model_output"].startswith("ERROR:"):
                output_row = row.to_dict()
                output_row['model_output'] = data['model_output']
                return output_row
    
    try:

        image_path = row['local_image_path']
        question = row['[Final]question']
        
        logger.info(f"Processing (Image Path: {image_path})")
        
        base64_image = get_image_base64(os.path.join("data", image_path))
        
        if not base64_image or base64_image == 'no_image':
            logger.warning(f"Skipping due to image error")
            output_row = row.to_dict()
            output_row['model_output'] = f"ERROR: no image"
            return output_row

        logger.info(f"Image {image_path} successfully fetched and encoded.")
        
        status, response_content = checkQA(base64_image, question, model_name)
        
        if status == 'success':
            # Create a new dictionary for the output row
            output_row = row.to_dict()
            output_row['model_output'] = response_content
            logger.info(f"Successfully processed")
            return output_row
        else:
            logger.error(f"API failed for index {image_path}: {response_content}")
            # Still return the row but with an error message in the output
            output_row = row.to_dict()
            output_row['model_output'] = f"ERROR: {response_content}"
            return output_row
            
    except Exception as e:
        logger.error(f"Error processing data row (Index: {row.get('index', 'unknown')}): {e}")
        output_row = row.to_dict()
        output_row['model_output'] = f"ERROR: {str(e)}"
        return output_row

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Pix2Fact QA Inference")
    parser.add_argument('--input_csv', default="data/Pix2Fact_1k.csv", help='Path to the input CSV file')
    parser.add_argument('--output_dir', default="outputs/pix2fact_eval/", help='Directory to save the output CSV')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, help='Name of the model to use for inference')
    parser.add_argument('--max_workers', type=int, default=30, help='Maximum number of worker threads')
    parser.add_argument('--start_index', type=int, default=0, help='Row index to start processing from')
    args = parser.parse_args()

    output_filename = f"Pix2Fact_QA_cases_1k_{args.model_name.replace('-', '_')}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    output_dir = Path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.input_csv}")
    try:
        full_df = pd.read_csv(args.input_csv)
        # full_df = full_df.head(5)
        logger.info(f"Successfully loaded {len(full_df)} total rows.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    if 'index' in full_df.columns:
        df_to_process = full_df[full_df['index'] >= args.start_index].copy()
    else:
        df_to_process = full_df.iloc[args.start_index:].copy()

    logger.info(f"Filtered to {len(df_to_process)} rows to process (starting from index {args.start_index}).")
    # return
    if df_to_process.empty:
        logger.warning("No data to process.")
        return
    
    logger.info("Starting processing...")
    
    results = []
    single_outputs_folder = output_dir / f"outputs_{args.model_name.replace('-', '_')}"
    single_outputs_folder.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a future for each row
        futures = dict()
        for i, row in df_to_process.iterrows():
            output_file_name = single_outputs_folder / f"{i}.json"
            futures[executor.submit(process_data, row, args.model_name, output_file_name)] = i
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            result_row = future.result()
            row_index = futures[future]
            if result_row:
                # print("Write result", row_index)
                output_file_name = single_outputs_folder / f"{row_index}.json"
                with open(output_file_name, "w") as f:
                    json.dump(result_row, f, indent=4, ensure_ascii=False)

                results.append(result_row)
            else:
                print("Failed to process", row_index)
    
    logger.info("Processing complete!")

    if results:
        # Sort results by original index to maintain order
        results_df = pd.DataFrame(results)
        if 'index' in results_df.columns:
            results_df.sort_values(by='index', inplace=True)
            
        logger.info(f"Saving {len(results_df)} results to: {output_path}")
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("All done.")
    else:
        logger.warning("No results were generated.")

if __name__ == "__main__":
    main()

    