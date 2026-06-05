# Pix2Fact

Code artifact for **Pix2Fact: When Vision Is Not Enough — Benchmarking Fine-Grained VQA with Web Verification on High-Resolution Real-World Scenes**.

Project page: https://fanfan7589.github.io/pix2fact/

Paper: https://arxiv.org/abs/2602.00593

Pix2Fact is a visual question-answering benchmark of 1,000 expert-crafted questions on 4K+ real-world scenes. Each answer requires both fine-grained visual grounding and open-web knowledge search.

## Dataset

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

Prepare the CSV and images under `data/`. The default scripts expect:

```text
data/Pix2Fact_1k.csv
data/images/
```

If your image paths in the CSV are relative paths, pass `--image_dir data`. If your images are flat files, pass the directory that contains those files.

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

Run with search:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --max_workers 4 \
  --model_name x-ai/grok-4.20:online \
  --output_dir outputs/pix2fact_eval_grok \
  --retries 3
```

Run without search by using a non-online model:

```bash
uv run python -m src.inference_openrouter_with_search_v2 \
  --input_csv data/Pix2Fact_1k.csv \
  --image_dir data \
  --max_workers 4 \
  --model_name x-ai/grok-4.20 \
  --output_dir outputs/pix2fact_eval_grok \
  --retries 3
```

Run with cropped images when the CSV includes a `bounding_box` column:

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

## Judge

The inference scripts write a CSV. Judge it with:

```bash
uv run python -m src.judge_openrouter \
  --input_csv /path/to/your/csv \
  --max_workers 4 \
  --prompt_version v3 \
  --model_name openai/gpt-5.4
```
