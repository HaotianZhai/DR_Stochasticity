# Deep Diagnosis: Diagnosing and Mitigating Stochasticity in Deep Research Agents

This repository provides the code for paper: Evaluating Stochasticity in Deep Research Agents. It contains three main components:

1. **Data Generator** — A deep research agent pipeline (based on [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)) that generates research trajectories from benchmark questions.
2. **Evaluation Pipeline** — A multi-stage pipeline that extracts claims, decomposes them into atomic facts, clusters findings semantically, and computes stochasticity metrics.
3. **Mitigation Methods** — Techniques for reducing stochasticity including query ensemble, structured output, and consistency voting.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Model Hosting](#model-hosting)
- [Dataset Preparation](#dataset-preparation)
- [Data Generation](#data-generation)
  - [Environment Setup](#environment-setup)
  - [DeepResearch Pipeline](#deepresearch-pipeline)
  - [Key Adjustments to the Original DeepResearch Framework](#key-adjustments-to-the-original-deepresearch-framework)
- [Evaluation Pipeline](#evaluation-pipeline)
  - [Full Stochasticity Pipeline](#full-stochasticity-pipeline)
  - [Individual Steps](#individual-steps)
  - [Tunable Hyperparameters](#tunable-hyperparameters)
- [Mitigation Methods](#mitigation-methods)
  - [Setup](#mitigation-setup)
  - [Running Mitigation Experiments](#running-mitigation-experiments)
  - [Mitigation Hyperparameters](#mitigation-hyperparameters)
- [Citation](#citation)

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) for local model hosting (recommended: 2x GPUs for running two models concurrently)
- API keys:
  - [You.com API](https://you.com/api) — used as the search engine for data generation
  - [Together AI API](https://www.together.ai/) — used for cloud model inference in mitigation experiments and embeddings

```bash
# Clone the repository
git clone https://github.com/<your-org>/deep-diagnosis.git
cd deep-diagnosis

# Install dependencies (for evaluation & mitigation)
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your actual API keys
```

---

## Model Hosting

We locally host models using [SGLang](https://github.com/sgl-project/sglang) with **deterministic inference** enabled. This is critical for our temperature ablation study — even with `temperature=0`, standard LLM inference can produce different outputs due to dynamic batching and varying reduction orders in GPU kernels. SGLang's deterministic inference mode uses batch-invariant operators to guarantee consistent outputs across runs.

See the [SGLang deterministic inference documentation](https://github.com/sgl-project/sgl-project.github.io/blob/main/_sources/advanced_features/deterministic_inference.md) for details.

### Launch the model servers

**Qwen3-4B (GPU 0, port 30000):**

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --attention-backend fa3 \
  --enable-deterministic-inference \
  --port 30000 \
  --host 0.0.0.0
```

**Qwen3-30B-A3B (GPU 1, port 30001):**

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --attention-backend fa3 \
  --enable-deterministic-inference \
  --port 30001 \
  --host 0.0.0.0
```

The `--enable-deterministic-inference` flag combined with `--attention-backend fa3` (FlashAttention 3) ensures fully deterministic outputs, supporting CUDA graphs, chunked prefill, and radix cache. This lets us isolate temperature as the sole source of randomness in our ablation studies.

---

## Dataset Preparation

### WebWalkerQA (for data generation & evaluation)

We use the [WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA) dataset (680 questions in the `main` split) as our primary benchmark for studying stochasticity in deep research agents.

```bash
python scripts/download_webwalkerqa.py --num-instances xx --output data/webwalkerqa.json
```

This downloads the dataset from HuggingFace and converts it to the `[{"question": "...", "answer": "..."}]` format expected by the pipelines.

### DeepSearchQA (for mitigation experiments)

We use the [DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) dataset (900 questions) from Google DeepMind for evaluating mitigation methods. This is a more challenging benchmark with multi-step information-seeking tasks.

```bash
python scripts/download_deepsearchqa.py --num-instances xx --output data/deepsearchqa.json
```

---

## Data Generation

### Environment Setup

The data generator is built on top of the [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) codebase and should be run in a **separate conda environment** (`react_infer_env`) following their setup instructions:

```bash
# Create the environment (Python 3.10 recommended)
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env

# Install DeepResearch dependencies
pip install -r https://raw.githubusercontent.com/Alibaba-NLP/DeepResearch/main/requirements.txt

# Also install our additional dependencies
pip install youdotcom python-dotenv
```

> **Note:** The evaluation and mitigation pipelines use the main `requirements.txt` at the repo root and do **not** require this environment. Only the data generator under `data_generator/deepresearch/` needs `react_infer_env`.

### DeepResearch Pipeline

This pipeline uses a ReAct agent with modular temperature control over three stages: summarization, reasoning, and query.

```bash
conda activate react_infer_env
cd data_generator/deepresearch

python run_multi_react_modular.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output ./results \
  --dataset ../../data/webwalkerqa.json \
  --temperature 0.0 \
  --temp-summarization 0.0 \
  --temp-reasoning 0.0 \
  --temp-query 0.0 \
  --roll_out_count 10 \
  --seed "1,3,5,7,9,11,13,15,17,19" \
  --max_workers 5
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name/path on the inference server | — |
| `--output` | Directory to save trajectory outputs | — |
| `--dataset` | Path to the input dataset JSON file | — |
| `--temperature` | Global sampling temperature applied to all modules (overridden by module-specific flags) | `0.0` |
| `--temp-summarization` | Temperature for the summarization module | inherits `--temperature` |
| `--temp-reasoning` | Temperature for the reasoning module | inherits `--temperature` |
| `--temp-query` | Temperature for the query generation module | inherits `--temperature` |
| `--seed` | Comma-separated list of seeds, one per rollout (e.g., `"1,3,5,7,9"`) | `None` |
| `--seed-summarization` | Base seed for the summarization module | `42` |
| `--seed-reasoning` | Base seed for the reasoning module | `42` |
| `--seed-query` | Base seed for the query module | `42` |
| `--roll_out_count` | Number of independent rollouts per question | `3` |
| `--max_workers` | Maximum parallel workers for concurrent inference | `20` |
| `--top_p` | Top-p (nucleus) sampling parameter | `0.95` |
| `--presence_penalty` | Presence penalty for generation | `1.1` |
| `--total_splits` | Split dataset into N chunks (for distributed runs) | `1` |
| `--worker_split` | Which chunk to process (1-indexed) | `1` |

### Key Adjustments to the Original DeepResearch Framework

We made the following changes to the original [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) codebase:

1. **Modular temperature control.** The original framework applies a single global temperature to all LLM calls. We refactored the ReAct agent into three independently configurable modules — *Summarization*, *Reasoning*, and *Query* — each with its own temperature and seed settings. This allows us to study how stochasticity in each module propagates through the pipeline.

2. **You.com search integration.** We replaced the original Serper/Jina-based search with the [You.com API](https://you.com/api) for web search, providing a unified search backend across both pipelines.

3. **Together AI API backend.** We added support for calling models via the [Together AI API](https://www.together.ai/) (OpenAI-compatible endpoint), enabling cloud-hosted inference for larger models (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507-tput`).

4. **Trajectory logging.** We added structured trajectory logging that records every step of the agent's execution (tool calls, model responses, intermediate findings) as JSON files, enabling downstream evaluation.

5. **Multi-rollout orchestration.** We added `run_multi_react_modular.py` to run multiple independent rollouts per question with different seeds in parallel, facilitating large-scale stochasticity measurement.

6. **Mitigation strategies.** We implemented three mitigation techniques within the modular agent:
   - **Query Ensemble**: Generates multiple search queries per turn and aggregates results.
   - **Structured Output**: Forces the agent to produce structured JSON responses for intermediate steps, reducing formatting variance.
   - **Consistency Voting**: Runs multiple reasoning passes and selects the most consistent output via majority voting.

---

## Evaluation Pipeline

The evaluation pipeline measures stochasticity across three dimensions:

- **Answer-level stochasticity** — Are the final answers semantically equivalent across runs? (LLM-as-judge comparison)
- **Finding-level stochasticity** — How much do the atomic findings differ across runs? (Cosine distance between canonical finding sets)
- **Citation-level stochasticity** — How consistent is the evidence sourcing? (Set-level citation overlap)

Additionally, the pipeline computes **accuracy** against ground truth answers.

### Full Stochasticity Pipeline

The easiest way to run the full evaluation is via the shell script:

```bash
cd evaluation/scripts

./run_full_stochasticity_pipeline.sh \
  --input-dir /path/to/trajectories \
  --output-dir ./eval_results \
  --threshold 0.94 \
  --max-workers 10 \
  --llm-base-url http://localhost:30000/v1 \
  --llm-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --accuracy \
  --reference /path/to/dataset.json
```

This runs four steps automatically:

1. **Extract QA Answers** — Extracts the final answer from each trajectory using an LLM.
2. **Extract Claims & Atomic Facts** — Decomposes research reports into atomic factual claims.
3. **Cluster Atomic Findings** — Embeds findings (via Together API's `intfloat/multilingual-e5-large-instruct`) and clusters semantically equivalent ones using cosine similarity + LLM verification.
4. **Calculate Stochasticity Metrics** — Computes answer/finding/citation-level stochasticity scores.

### Converting DeepResearch JSONL Output

The DeepResearch pipeline outputs JSONL files. Convert them to the trajectory format before evaluation:

```bash
python evaluation/scripts/extract_deepresearch_reports.py \
  --input-dir /path/to/deepresearch/output \
  --output-dir /path/to/trajectories
```

### Individual Steps

You can also run each step independently:

```bash
# Step 1: Extract QA answers
python evaluation/claim_extraction/extract_claims.py \
  --input-dir /path/to/trajectories \
  --output-dir ./answers \
  --mode qa_answers \
  --llm-base-url http://localhost:30000/v1 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Step 2a: Extract claims
python evaluation/claim_extraction/extract_claims.py \
  --input-dir /path/to/trajectories \
  --output-dir ./claims \
  --mode claims

# Step 2b: Decompose into atomic facts
python evaluation/claim_extraction/extract_claims.py \
  --input-dir ./claims \
  --output-dir ./atomic_facts \
  --mode atomic_facts

# Step 3: Cluster findings
python evaluation/atomic_findings/atomic_findings_pipeline.py \
  ./atomic_facts/atomic_facts_*.json \
  --threshold 0.94 \
  --output ./clustered/clustered_findings.json \
  --llm-base-url http://localhost:30000/v1 \
  --llm-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --embedding-model intfloat/multilingual-e5-large-instruct \
  --max-workers 10

# Step 4: Calculate metrics
python evaluation/atomic_findings/calculate_stochasticity.py \
  ./clustered/clustered_findings.json \
  ./answers \
  --llm-base-url http://localhost:30000/v1 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max-workers 10 \
  --accuracy --reference /path/to/dataset.json \
  --output ./metrics/stochasticity_metrics.json
```

### Tunable Hyperparameters

#### Data Generation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temperature` (per module) | Sampling temperature for each agent module | `0.0` |
| `seed` | Random seed passed to the model server | `None` |
| `rollout_count` | Number of independent runs per question | `5` |
| `max_workers` | Parallel workers for data generation | `1`–`10` |

#### Evaluation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--threshold` | Cosine similarity threshold for clustering atomic findings | `0.94` |
| `--max-workers` | Parallel workers for LLM verification calls | `10` |
| `--llm-base-url` | Endpoint for the LLM used in evaluation | `http://localhost:30000/v1` |
| `--embedding-model` | Model for computing finding embeddings | `intfloat/multilingual-e5-large-instruct` |
| `--no-llm` | Disable LLM semantic comparison (fall back to exact match) | `False` |

---

## Mitigation Methods

Mitigation experiments use the **Together AI API** for cloud-hosted inference (no local GPU required) and the **DeepSearchQA** dataset.

### Mitigation Setup

```bash
# Set your Together AI API key
export TOGETHER_API_KEY="your-together-api-key"

# Download the DeepSearchQA dataset
python scripts/download_deepsearchqa.py --num-instances 25 --output data/deepsearchqa.json
```

### Running Mitigation Experiments

```bash
cd mitigation/scripts

# Baseline (no mitigation, temperature=0.7)
./run_mitigation_experiment.sh baseline \
  --temp 0.7 \
  --rollouts 5 \
  --seeds "1,24,32,444,50239" \
  --dataset ../../data/deepsearchqa.json

# Query ensemble mitigation
./run_mitigation_experiment.sh query_ensemble \
  --temp 0.7 \
  --use-ensemble \
  --rollouts 5 \
  --seeds "1,24,32,444,50239" \
  --dataset ../../data/deepsearchqa.json

# Structured output mitigation
./run_mitigation_experiment.sh structured_output \
  --temp 0.7 \
  --use-structure \
  --rollouts 5 \
  --seeds "1,24,32,444,50239" \
  --dataset ../../data/deepsearchqa.json

# Consistency voting mitigation
./run_mitigation_experiment.sh consistency_voting \
  --temp 0.7 \
  --use-consistency \
  --rollouts 5 \
  --seeds "1,24,32,444,50239" \
  --dataset ../../data/deepsearchqa.json

# Per-module temperature ablation
./run_mitigation_experiment.sh low_query_temp \
  --temp 0.7 \
  --temp-query 0.0 \
  --rollouts 5 \
  --seeds "1,24,32,444,50239" \
  --dataset ../../data/deepsearchqa.json
```

Each experiment runs inference, evaluates stochasticity, and saves a summary JSON to `ablation_outputs/iterative_mitigation/_results/`.

### Mitigation Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--temp` | Global temperature (applied to all modules unless overridden) | `0.7` |
| `--temp-summarization` | Summarization module temperature | inherits `--temp` |
| `--temp-reasoning` | Reasoning module temperature | inherits `--temp` |
| `--temp-query` | Query module temperature | inherits `--temp` |
| `--use-ensemble` | Enable query ensemble (multiple queries per turn) | off |
| `--use-structure` | Enable structured JSON output for intermediate steps | off |
| `--use-consistency` | Enable consistency voting across reasoning passes | off |
| `--rollouts` | Number of independent rollouts per question | `5` |
| `--seeds` | Comma-separated seed list for rollouts | `1,24,32,444,50239` |
| `--max-workers` | Parallel workers for inference and evaluation | `10` |
| `--model` | Model name on Together AI | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt                      # Dependencies for evaluation & mitigation
├── .env.example
├── .gitignore
├── scripts/
│   ├── download_webwalkerqa.py           # Download WebWalkerQA dataset
│   └── download_deepsearchqa.py          # Download DeepSearchQA dataset
├── data/                                 # Downloaded datasets (gitignored)
├── data_generator/
│   └── deepresearch/                     # Alibaba DeepResearch-based pipeline
│       ├── react_agent_modular.py        # Modular ReAct agent (3-module)
│       ├── run_multi_react_modular.py    # Multi-rollout orchestrator
│       ├── prompt.py                     # System prompts
│       └── tool_search.py               # You.com search tool
├── evaluation/
│   ├── claim_extraction/
│   │   └── extract_claims.py             # Claim/answer/atomic fact extraction
│   ├── atomic_findings/
│   │   ├── atomic_findings_pipeline.py   # Embedding + semantic clustering
│   │   └── calculate_stochasticity.py    # Stochasticity metrics
│   └── scripts/
│       ├── run_full_stochasticity_pipeline.sh   # End-to-end eval pipeline
│       └── extract_deepresearch_reports.py      # JSONL → trajectory converter
└── mitigation/
    ├── inference/
    │   ├── react_agent_modular_new.py    # Agent with mitigation strategies
    │   ├── run_multi_react_modular_new.py
    │   ├── prompt.py                     # Shared prompts
    │   └── tool_search.py               # Shared search tool
    └── scripts/
        └── run_mitigation_experiment.sh  # Single experiment runner
```

---

## Citation

```bibtex
@article{deep_diagnosis_2025,
  title={Deep Diagnosis: Diagnosing and Mitigating Stochasticity in Deep Research Agents},
  year={2025}
}
```

---

## Acknowledgments

This work builds upon:
- [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) by Alibaba NLP / Tongyi Lab
- [SGLang](https://github.com/sgl-project/sglang) for deterministic inference
- [WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA) dataset
- [DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) dataset by Google DeepMind
