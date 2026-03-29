# Benchmark Plan

This repository now treats benchmarking as a controlled quality-improvement workflow rather than a one-number leaderboard.

## Core benchmark suite

1. Classification: BERT vs BiLSTM vs TextCNN on the same train/validation/test split.
2. Retrieval: dense retrieval and reranking latency/quality trade-offs.
3. Robustness: typo, paraphrase, and word-order perturbations.
4. Hard set: curated failure buckets for long-context, negation, and numeric/date reasoning.
5. Run promotion: compare candidate runs against a baseline using rule-based gates.

## Artifact map

- `results/classification_results.csv`: headline model comparison table.
- `results/baselines.csv`: compact benchmark export for README tables.
- `results/retrieval_metrics.csv`: current retrieval benchmark snapshot.
- `experiments/results/retrieval_history.csv`: append-only retrieval benchmark history.
- `results/robustness_summary.csv`: compressed robustness summary for reporting.
- `data/eval/hard_set.json`: curated hard-set evaluation file.
- `experiments/results/summary.csv`: append-only training run history with config and metric summaries.
- `outputs/<run>/hard_set_metrics.json`: hard-set evaluation for a specific run.
- `outputs/<run>/best_run_summary.json`: best epoch, losses, optimizer-step info, and augmentation summary.

## Recommended run order

```bash
python scripts/build_hard_set.py
python scripts/generate_targeted_nli_data.py --negation 500 --numeric 300 --temporal 300 --long_reasoning 400 --output-path data/generated/targeted_nli_run3.json
python train.py --config-path configs/bert_run3.json
python scripts/eval_hard_set.py --checkpoint outputs/bert_run3
python scripts/compare_runs.py --base outputs/bert_nli --candidate outputs/bert_run3
python scripts/summarize_run.py --run outputs/bert_run3
python -m evaluation.retrieval_benchmark --corpus-path data/test.json --checkpoint-dir outputs/bert_run3 --output-json results/retrieval_metrics.json --output-csv results/retrieval_metrics.csv
```

## Promotion policy

A candidate run is only considered improved if:

- validation macro F1 improves,
- hard-set macro F1 improves or does not regress materially,
- typo robustness does not collapse,
- shuffle robustness does not collapse,
- and the run writes reproducible artifacts.

The comparison logic is implemented in `scripts/compare_runs.py` and should be used instead of manually comparing isolated metrics.

## Reporting guidance

- Keep all benchmark rows tied to a concrete artifact directory in `outputs/`.
- Report accuracy and macro F1 for classification, not just one headline number.
- Treat `results/retrieval_metrics.csv` as the clean current snapshot and `experiments/results/retrieval_history.csv` as history.
- Use the hard set to track whether the model is improving on the failure buckets that dominate error analysis.
- Update README tables only after the benchmark ledgers and run summaries are refreshed.
