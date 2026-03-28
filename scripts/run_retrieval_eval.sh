#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/sample_nli.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-results/retrieval_metrics.json}"
OUTPUT_CSV="${OUTPUT_CSV:-results/retrieval_metrics.csv}"

ARGS=(--corpus-path "$CORPUS_PATH" --output-json "$OUTPUT_JSON" --output-csv "$OUTPUT_CSV")
if [[ -n "$CHECKPOINT_DIR" ]]; then
  ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

python -m evaluation.retrieval_benchmark "${ARGS[@]}"
