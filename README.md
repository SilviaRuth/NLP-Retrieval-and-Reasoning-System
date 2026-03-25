# NLP Text Classification and Retrieval Reasoning

This repository upgrades the original CITS4012 group notebook into a modular NLP system that is easier to extend, evaluate, and present as a production-style project.

It keeps the original Natural Language Inference (NLI) framing and adds:

- Transformer fine-tuning with Hugging Face BERT
- Retrieval with SentenceTransformers + FAISS, with a TF-IDF fallback
- Evaluation metrics for classification and retrieval
- Error analysis and robustness testing
- Experiment tracking to JSON and CSV
- An optional FastAPI inference layer

---

## Project Goals

* Build a **retrieval + reasoning pipeline** similar to modern AI systems (e.g., Copilot, Bing)
* Demonstrate **evaluation depth and research thinking**

---

## System Architecture

### 1. Retrieval Layer

* Vector embeddings using SentenceTransformers
* FAISS index for similarity search
* Top-K document retrieval

### 2. Reasoning Layer (NLI)

* Fine-tuned BERT/RoBERTa model
* Classifies entailment / contradiction / neutral
* Used for ranking and validation

### 3. Evaluation Layer

* Accuracy, Precision, Recall, F1
* Recall@K, MRR (for retrieval)
* Error categorization

### 4. Robustness Testing

* Noise injection (typos)
* Paraphrasing
* Word order perturbation

---

## Key Features

* 🔍 Retrieval + reasoning pipeline
* 🧠 Fine-tuned Transformer models
* 📉 Error analysis and failure categorization
* ⚙️ Robustness evaluation under input perturbations
* 📈 Experimental comparison across configurations

---

## Experiments

* BERT vs LSTM vs CNN comparison
* Attention ablation study
* Embedding dimension tuning
* Retrieval performance (Recall@K, MRR)
* Prompt / input variation impact

---

## Project Layout

```text
project/
|-- api/
|-- data/
|-- evaluation/
|-- experiments/
|-- models/
|-- retrieval/
|-- utils/
|-- main.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Supported Data Format

The loader supports both:

1. A list of row objects:

```json
[
  {"premise": "...", "hypothesis": "...", "label": "entailment"}
]
```

2. The original notebook-friendly columnar JSON format:

```json
{
  "premise": {"0": "...", "1": "..."},
  "hypothesis": {"0": "...", "1": "..."},
  "label": {"0": "entails", "1": "neutral"}
}
```

Labels such as `entails` and `contradicts` are normalized automatically.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a BERT NLI model:

```bash
python train.py --train-path data/train.json --val-path data/validation.json --test-path data/test.json --output-dir outputs/bert_nli
```

Run retrieval + reasoning inference:

```bash
python main.py --corpus-path data/train.json --query "A soccer game with multiple males playing." --checkpoint-dir outputs/bert_nli --top-k 5
```

Run a baseline instead of BERT:

```bash
python train.py --model-type lstm --train-path data/train.json --val-path data/validation.json --output-dir outputs/lstm_baseline
```

Start the optional API:

```bash
uvicorn api.app:app --reload
```

## What Changed from the Notebook

The original notebook bundled everything into one Colab workflow. This refactor separates concerns:

- `data/`: loading, vocabulary building, and datasets
- `models/`: BERT fine-tuning and baseline models
- `retrieval/`: dense retrieval with FAISS and fallback indexing
- `evaluation/`: metrics, error analysis, and robustness checks
- `experiments/`: result logging and run summaries

That makes the project easier to test, explain, and extend for ablations or demos.

## Reproducibility

Training sets random seeds across Python, NumPy, and PyTorch. Each run writes:

- `metrics.json`
- `error_analysis.json`
- `robustness.json`
- `run_summary.json`
- `experiments/results/summary.csv`

## Notes

- BERT and SentenceTransformers checkpoints may need to be downloaded the first time you run them.
- In offline environments, the retrieval system can still fall back to TF-IDF if dense retrieval packages are unavailable.
- The original notebook can remain as the coursework artifact; this repository is the engineered version for GitHub and portfolio use.
