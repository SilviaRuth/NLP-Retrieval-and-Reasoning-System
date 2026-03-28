#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/sample_nli.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
QUERY="${QUERY:-Several men are playing soccer outdoors.}"

if [[ -f data/train.json && -f data/validation.json ]]; then
  bash scripts/run_baselines.sh
fi

bash scripts/run_retrieval_eval.sh

ARGS=(--corpus-path "$CORPUS_PATH" --query "$QUERY" --output-path results/example_inference.json)
if [[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]]; then
  ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

python main.py "${ARGS[@]}"
