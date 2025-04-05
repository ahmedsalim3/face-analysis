# Face Analysis

A package for analyzing faces in images to detect eye state, gaze direction, and facial expressions

## How to install:

Follow these steps:

1. Create a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```sh
bash install.sh
```

## How to run the script:

- Place your images in the `input/` folder.
- Run the script with these options:

```sh
python -m face_analysis.image_selection --input input/couples --output output/image_selection/couples --save-annotated-images
```

### Using specific image:

```sh
python -m face_analysis.run_face_analysis --images input/test_1.png --output output
```

### Or multiple images:

```sh
python -m face_analysis.run_face_analysis --images input/test_1.png input/test_2.jpg --output output --save-annotated-images
```
_setting `--device cuda` will use the GPU for processing, otherwise it will use the `cpu`._

~~### Demo App:~~

~~The project includes a graphical demo application for interactively analyzing images:~~

**Let me know, if you need it**

~~The demo app provides:~~

- ~~Side-by-side view of original and annotated images~~
- ~~Option to select any image from your file system~~
- ~~Ability to download the analysis results as a JSON file~~
- ~~Visual representation of face features, eye state, and gaze direction~~

## Output

The scripts generates two JSON files in the results directory:

1. `face_analysis_results.json` - Contains detailed analysis of all faces
2. `image_selection.json` - Contains simplified results with only:
   - Whether eyes are open
   - Whether subject is looking at camera
   - Smiling score (0-1)

Annotated images are saved to the `output/annotated/` directory, check them [here](output/annotated).

## Quick Start Example

To quickly test the face analysis pipeline refer to this [notebook](./notebooks/notebook.ipynb):

## Repo Structure

```sh
project_root/
├── data/               
├── input/               
│   ├── test_1.jpg
│   └── ... (8 test images)
├── models/          
│   ├── L2CSNet_gaze360.pkl
│   ├── eye_state_classifier.h5
│   └── shape_predictor_68_face_landmarks.dat
├── output/             
│   ├── annotated/ 
│   │   ├── test_1_emotions.png
│   │   ├── test_1_eyes.png
│   │   └── ...
│   ├── image_selection.json
│   └── test_3.jpg
├── scripts/               
│   ├── datasets/
│   ├── face_analyzer.sh
│   └── ...
├── src/                    
│   └── face_analysis/  
│       ├── emotions/      
│       ├── eyes/          
│       ├── gazes/  
│       ├── config.py
│       ├── image_selection.py
│       └── run_face_analysis.py
├── LICENSE.txt
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt
└── results.json
```