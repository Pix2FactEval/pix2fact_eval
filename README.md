<p align="center">
  <img src="assets/logo1_pix2fact.png" alt="Pix2Fact Logo" width="140">
</p>

<h1 align="center">Pix2Fact: When Vision Is Not Enough</h1>

<p align="center">
  <strong>Benchmarking Fine-Grained VQA with Web Verification on High-Resolution Real-World Scenes</strong>
</p>

<p align="center">
  <a href="https://fanfan7589.github.io/pix2fact/">Project Page</a> |
  <a href="https://arxiv.org/abs/2602.00593">Paper</a> |
  <a href="https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark">Dataset</a>
</p>

<p align="center">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-Pix2Fact-4B7BEC">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E">
  <img alt="Questions" src="https://img.shields.io/badge/Questions-1,000-20BF6B">
  <img alt="Scenes" src="https://img.shields.io/badge/Scenes-4K%2B_real--world-FA8231">
  <img alt="Task" src="https://img.shields.io/badge/Task-VQA_%2B_Web_Search-8854D0">
</p>

Pix2Fact is a visual question-answering benchmark of 1,000 expert-crafted questions on 4K+ real-world scenes. Each answer requires both fine-grained visual grounding and open-web knowledge search.

## Dataset

The dataset is hosted at [pix2fact/Pix2FactBenchmark](https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark). Hugging Face serves it as parquet with one `default` subset and a `train` split of 1,000 rows. The main fields are:

| Column | Type | Description |
| --- | --- | --- |
| `image` | image | The benchmark image decoded by `datasets`. |
| `question` | string | The visual + web verification question. |
| `answer` | string | The ground-truth final answer. |
| `index` | string | Dataset case id. |
| `local_image_path` | string | Suggested local filename/path for the image. |
| `search_query` | string | Reference search query or entity hint. |
| `bounding_box` | string | Region annotation used by crop settings. |
| `evidence_1/2/3` | string | Human evidence notes. |
| `evidence_url_1/2/3` | string | Supporting URLs. |
| `category` | string | Scene category. |
| `image_resolution` | string | Original image resolution. |

The benchmark covers eight everyday scene categories:

| Category | Questions |
| --- | ---: |
| Street scene with people | 193 |
| Storefronts & facades | 144 |
| Retail & commercial interior | 129 |
| Traffic & infrastructure | 125 |
| Markets & outdoor vendors | 111 |
| Public & cultural interior | 105 |
| Landmarks & attractions | 103 |
| Cityscape & aerial | 90 |

### Download from HF parquet

The recommended path is to use the Hugging Face `datasets` parquet backend and export a local CSV plus images:

```bash
uv run python -m src.download_hf_dataset --output_dir data
```

This writes:

```text
data/Pix2Fact_1k.csv
data/<local_image_path files>
```

For a quick local smoke test, export only one row:

```bash
uv run python -m src.download_hf_dataset --output_dir data --limit 1
```

### Iterate over samples

You can also read the parquet dataset directly:

```python
from datasets import load_dataset

ds = load_dataset("pix2fact/Pix2FactBenchmark", split="train")
sample = ds[0]

print(sample["question"])
print(sample["answer"])
print(sample["local_image_path"])
sample["image"].save("sample.jpg")

for row in ds:
    question = row["question"]
    image = row["image"]
    answer = row["answer"]
    # call your model here
```

## Install

We use `uv` to manage the environment:

```bash
uv venv
uv sync
```

Copy `env.example` to `.env` and fill in the provider keys you need.

## Evaluation Settings

The leaderboard evaluates each model under four conditions:

| Setting | Image | Web search |
| --- | --- | --- |
| C1 | original | no |
| C2 | original | yes |
| C3 | crop | no |
| C4 | crop | yes |

The project page reports Gemini-3.1-Pro as the best listed model at 51.7% in C4, while human expert PhD annotators with web access are approximately 100%.

## OpenRouter Inference

### One-sample model call

After exporting one or more samples, run one case through an OpenRouter-compatible model:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --model_name x-ai/grok-4.20:online \
  --output_dir outputs/pix2fact_eval_smoke \
  --max_workers 1 \
  --max_rows 1 \
  --retries 3
```

This produces a CSV like:

```text
outputs/pix2fact_eval_smoke/Pix2Fact_with_response_x_ai_grok_4.20_online.csv
```

### Batch with search

Run all rows with search:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --max_workers 4 \
  --model_name x-ai/grok-4.20 \
  --output_dir outputs/pix2fact_eval_grok \
  --retries 3
```

### Batch without search

Use a non-online model name:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --max_workers 4 \
  --model_name x-ai/grok-4.20 \
  --output_dir outputs/pix2fact_eval_grok_no_search \
  --retries 3
```

### Batch with cropped images

Use `--crop_bbox` when the CSV includes `bounding_box`:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --max_workers 4 \
  --model_name x-ai/grok-4.20:online \
  --output_dir outputs/pix2fact_eval_grok \
  --retries 3 \
  --crop_bbox
```

### Judge the output

Judge a model output CSV:

```bash
uv run python -m src.judge_openrouter \
  --input_csv outputs/pix2fact_eval_grok/Pix2Fact_with_response_x_ai_grok_4.20_online.csv \
  --max_workers 4 \
  --prompt_version v3 \
  --model_name openai/gpt-5.4
```

For a smoke test, add `--max_rows 1`.

## Agent Inference

To run open-weight models, deploy an OpenAI-compatible endpoint first, for example with `vllm`, then run:

```bash
uv run python -m src.inference_agent_with_search \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --model_name Qwen/Qwen3.6-27B \
  --output_dir outputs/pix2fact_eval_agent \
  --max_workers 8 \
  --max_steps 10
```

The agent path can use `SearchTool`, `VisitTool`, and `TerminateTool`. Configure `MODELHUB_SEARCH_URL`, `MODELHUB_SEARCH_API_KEY`, and `JINA_READER_API_KEY` in `.env` when enabling search/visit tools.

## End-to-end smoke test

The smallest connected flow is:

```bash
uv run python -m src.download_hf_dataset --output_dir data --limit 1

uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --model_name x-ai/grok-4.20:online \
  --output_dir outputs/pix2fact_eval_smoke \
  --max_workers 1 \
  --max_rows 1 \
  --retries 3

uv run python -m src.judge_openrouter \
  --input_csv outputs/pix2fact_eval_smoke/Pix2Fact_with_response_x_ai_grok_4.20_online.csv \
  --max_workers 1 \
  --max_rows 1 \
  --prompt_version v3 \
  --model_name openai/gpt-5.4
```
