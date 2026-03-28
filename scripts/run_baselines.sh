#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${TRAIN_PATH:-data/train.json}"
VAL_PATH="${VAL_PATH:-data/validation.json}"
TEST_PATH="${TEST_PATH:-data/test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

python train.py --model-type bert --train-path "$TRAIN_PATH" --val-path "$VAL_PATH" --test-path "$TEST_PATH" --output-dir "$OUTPUT_ROOT/bert_nli"
python train.py --model-type bilstm --train-path "$TRAIN_PATH" --val-path "$VAL_PATH" --test-path "$TEST_PATH" --output-dir "$OUTPUT_ROOT/bilstm_baseline"
python train.py --model-type cnn --train-path "$TRAIN_PATH" --val-path "$VAL_PATH" --test-path "$TEST_PATH" --output-dir "$OUTPUT_ROOT/textcnn_baseline"
