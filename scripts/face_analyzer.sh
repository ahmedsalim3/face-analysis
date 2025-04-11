# usage: run_face_analysis.py [-h] [--images IMAGES [IMAGES ...]] [--output OUTPUT] [--device {cuda,cpu}] [--save-annotated-images]

# Run combined face analysis pipeline

# options:
#   -h, --help            show this help message and exit
#   --images IMAGES [IMAGES ...]
#                         List of image paths to analyze
#   --output OUTPUT       Output JSON file path
#   --device {cuda,cpu}   Device to run inference on
#   --save-annotated-images
#                         Save all the annotated images
# python -m face_analysis.run_face_analysis --images input/test_1.png --output output --device "cuda" --save-annotated-images
# python -m face_analysis.run_face_analysis --images input/test_1.png --output output
python -m face_analysis.run_face_analysis --images input/test_1.png input/test_2.jpg input/test_3.jpg input/test_4.png --output output --save-annotated-images