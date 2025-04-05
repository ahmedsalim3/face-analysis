# This script is used to download the weights of the L2CSNet_gaze360 model
# found at: https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s
# Repo: https://github.com/Ahmednull/L2CS-Net.git
#
#!/bin/bash

pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s"

mv Gaze360/L2CSNet_gaze360.pkl models/L2CSNet_gaze360.pkl
rm -rf Gaze360
echo "Download completed!"
