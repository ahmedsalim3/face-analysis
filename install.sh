#!/bin/bash

echo "Downloading Shape Predictor..."
bash scripts/shape_predictor.sh

echo "Downloading L2CSNet Weights..."
bash scripts/l2csnet_weights.sh

echo "Installing dependencies..."
pip install .

echo "Verifying installation..."
python -c "import face_analysis; print(face_analysis.__version__)"

echo "Installation complete!"
