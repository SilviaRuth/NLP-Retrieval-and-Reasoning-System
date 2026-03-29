#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/sample_nli.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-results/retrieval_metrics.json}"
OUTPUT_CSV="${OUTPUT_CSV:-results/retrieval_metrics.csv}"
HISTORY_CSV="${HISTORY_CSV:-experiments/results/retrieval_history.csv}"
RUN_NAME="${RUN_NAME:-}"

ARGS=(--corpus-path "$CORPUS_PATH" --output-json "$OUTPUT_JSON" --output-csv "$OUTPUT_CSV" --history-csv "$HISTORY_CSV")
if [[ -n "$CHECKPOINT_DIR" ]]; then
  ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [[ -n "$RUN_NAME" ]]; then
  ARGS+=(--run-name "$RUN_NAME")
fi

python -m evaluation.retrieval_benchmark "${ARGS[@]}"
