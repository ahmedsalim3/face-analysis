#!/bin/bash
# This script downloads the Closed Eyes In The Wild (CEW) dataset 
# from Google Drive and extracts them to the 'data' directory.
# Ref: https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

pwd=$(pwd -P)
cd "$pwd"
mkdir -p data
cd data

# Download images
echo "Downloading Eye images in size of 24x24..."
wget --no-check-certificate \
    "https://drive.usercontent.google.com/u/0/uc?id=1Z5hZZnkN4VycK-mOOzUX1Zuwty8BEePy&export=download" \
    -O dataset_B_Eye_Images.rar

echo "Downloading Facial images in original resolution..."
wget --no-check-certificate \
    "https://drive.usercontent.google.com/u/0/uc?id=12DB4kwdeikxyQcK4gA7hL0QbzF6iOAZ2&export=download" \
    -O dataset_B_FacialImages_highResolution.rar

echo "Downloading Facial images in original resolution..."
wget --no-check-certificate \
    "https://drive.usercontent.google.com/u/0/uc?id=1FcRw251WxGKwT6Dt9ncP-1TbW6h445W5&export=download" \
    -O dataset_B_Facial_Images.rar

# Extract files
echo "Extracting downloaded files..."
unrar x dataset_B_Eye_Images.rar
unrar x dataset_B_FacialImages_highResolution.rar
unrar x dataset_B_Facial_Images.rar

# Clean up
echo "Cleaning up..."
rm -f dataset_B_Eye_Images.rar
rm -f dataset_B_FacialImages_highResolution.rar
rm -f dataset_B_Facial_Images.rar

echo "Download and extraction complete."