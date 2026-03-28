# Benchmark Plan

This repository now ships the benchmark ledger files and entry points needed to turn the project into a reproducible portfolio system.

## Core benchmark suite

1. Classification: BERT vs BiLSTM vs TextCNN on the same train/validation/test split.
2. Retrieval: dense retrieval against TF-IDF fallback.
3. Pipeline: reranking off vs reranking on.
4. Robustness: typo, paraphrase, and word-order perturbations.
5. Serving: CLI/API latency and end-to-end response time.

## Artifact map

- `results/classification_results.csv`: headline model comparison table.
- `results/baselines.csv`: compact benchmark export for README tables.
- `results/retrieval_metrics.csv`: retrieval benchmark rows emitted by `python -m evaluation.retrieval_benchmark`.
- `results/retrieval_results.csv`: curated retrieval table for README/docs.
- `results/robustness_results.csv`: full perturbation-by-model table.
- `results/robustness_summary.csv`: compressed robustness summary for reporting.

## Recommended run order

```bash
bash scripts/run_baselines.sh
bash scripts/run_retrieval_eval.sh
python main.py --corpus-path data/sample_nli.json --query "Several men are playing soccer outdoors."
python -m unittest discover -s tests
```

## Reporting guidance

- Keep all benchmark rows tied to a concrete artifact directory in `outputs/`.
- Report accuracy and macro F1 for classification, not just one headline number.
- Report Recall@k, MRR, and average latency for retrieval.
- When reranking is enabled, preserve both the normalized retrieval score and the entailment score so score composition is auditable.
- Update the README tables only after the CSV ledgers are refreshed.
