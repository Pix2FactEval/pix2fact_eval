<p align="center">
  <img src="assets/logo1_pix2fact.png" alt="Pix2Fact Logo" width="120">
</p>

<p align="center">
        &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark">Hugging Face Dataset</a>&nbsp</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="">Paper</a>&nbsp&nbsp
<br>


A visual QA benchmark evaluating expert-level perception and knowledge-intensive multi-hop reasoning—SOTA VLMs reach only 24% accuracy vs. 56% human performance.

一个评测专家级视觉感知与知识密集型多跳推理的视觉问答基准；当前最优 VLM 准确率仅 24%，人类达 56%。

> [**From Pixels to Facts (Pix2Fact): Benchmarking Multi-Hop Reasoning for Fine-Grained Visual Fact Checking**]()<br>
> [Yifan Jiang](), [Cong Zhang](), [Bofei Zhang](), [Yifan Yang](), [Bingzhang Wang](), [Yew Soon Ong]()
<p align="center">
  <img src="assets/teaser.png" alt="Pix2Fact Logo" width="100%">
</p>

---

## 🚀 Getting Started

### 🔍 Install
We use `uv` to manage this environment. To use uv, checkout [here](https://docs.astral.sh/uv/getting-started/installation/). After you install uv, simply do:
```bash
git clone https://github.com/Pix2FactEval/pix2fact_eval.git
cd pix2fact_eval

uv venv
uv sync
```
This will install a virtual env.
### Get Dataset
You can use this script to do the download, this script will download image files and csv file to a local folder:
```bash
uv run src/downlaod_data.py data
```
Also, it's okay to follow HF to manually download this data. The HF dataset is [here](https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark/tree/main). We also have [a csv file](https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark/resolve/main/Pix2Fact_1k.csv) contains all items in benchmark.

### 🚗 Run Inference and Judge
In our experiment, we mainly use `openai` compatible format to call api. To use the api, you should configure `.env` based on your model provider. Checkout `.env.example` for an example. To run inferce, execute this:
```bash
uv run src/inference_openai.py 
```
Taking `gpt-5-2025-08-07` as an example, exeute this script with default parameters will finally gives use a csv file `outputs/pix2fact_eval/Pix2Fact_QA_cases_1k_gpt_5_2025_08_07.csv`. Then, you can use judge script:
```bash
uv run src/judge.py --input_csv outputs/pix2fact_eval/Pix2Fact_QA_cases_1k_gpt_5_2025_08_07.csv
```
We recommend use `gpt-4o-2024-11-20` to for judge script.

### 📓 Notes for non-OpenAI models
#### Gemini
We use `openai` package to test gemini-2.5 and gemini-3. To use their search api, we called the `chat` api with the following code snippet:
```python
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=max_tokens,
    temperature=0,
    top_p=0.9,
    tools=[
        {
            "type": "google_search"
        }
    ]
)
```
`google_search` is inside tool that gemini official api will provide. You don't need to worry about how the tool is executed and how to handle the response.

#### Doubao
We use volcano engine to run doubao and doubao use `responses` api to handle search engine:
```python
# endpoint https://ark.cn-beijing.volces.com/api/v3
response = client.responses.create(
    model=model_name,
    input=messages,
    temperature=0,
    tools = [
        {"type": "web_search", "max_keyword": 10}
    ],
    max_output_tokens=16000
)
```
Please check their offical documentation for parameter of web search.

---

## Citation

If you use Pix2Fact in your research, please cite:

```bibtex
@article{jiang2025pix2fact,
  title={From Pixels to Facts (Pix2Fact): Benchmarking Multi-Hop Reasoning for Fine-Grained Visual Fact Checking},
  author={Jiang, Yifan and Zhang, Cong and Zhang, Bofei and Yang, Yifan and Wang, Bingzhang and Ong, Yew Soon},
  journal={},
  year={2025}
}
```

---

## License

MIT

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Pix2FactEval/pix2fact_eval&type=Date)](https://www.star-history.com/#Pix2FactEval/pix2fact_eval&Date)