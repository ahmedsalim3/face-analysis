#!/bin/bash

# Analyze all images in a folder
INPUT=${1:-"input/"}
OUTPUT=${2:-"output/folder"}
DEVICE=${3:-"cpu"}
DETECTOR=${4:-"retinaface"}

mkdir -p "$OUTPUT"

python scripts/folder_analysis.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --detector "$DETECTOR" \
    --save-annotated

echo "Results saved to: $OUTPUT"
echo "Check $OUTPUT/annotated/ for processed images"
echo "Check $OUTPUT/results.json for analysis data"