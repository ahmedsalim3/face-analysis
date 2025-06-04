#!/bin/bash

# Select best image from folder
INPUT=${1:-"input/"}
OUTPUT=${2:-"output/best"}
DEVICE=${3:-"cpu"}
DETECTOR=${4:-"retinaface"}

mkdir -p "$OUTPUT"

python scripts/best_selecter.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --detector "$DETECTOR"
