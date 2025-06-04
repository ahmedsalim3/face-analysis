#!/bin/bash

# Single image face analysis
INPUT=${1:-"input/test_1.png"}
OUTPUT=${2:-"output/single"}
DEVICE=${3:-"cpu"}
DETECTOR=${4:-"retinaface"}

mkdir -p "$OUTPUT"

python scripts/image_analysis.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --detector "$DETECTOR" \
    --save-annotated \
    --display

echo "Results saved to: $OUTPUT"