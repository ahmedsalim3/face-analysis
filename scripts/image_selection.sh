# usage: image_selection.py [-h] --input INPUT [--output OUTPUT] [--device {cuda,cpu}] [--save-annotated-images]

# Select best images based on face analysis

# options:
#   -h, --help            show this help message and exit
#   --input INPUT         Input folder containing images
#   --output OUTPUT       Output folder for results
#   --device {cuda,cpu}   Device to run inference on
#   --save-annotated-images
#                         Save all the annotated images
# python -m face_analysis.image_selection --input input/ --output output
python -m face_analysis.image_selection --input input/couples --output output/image_selection/couples --save-annotated-images
python -m face_analysis.image_selection --input input/kids --output output/image_selection/kids --save-annotated-images