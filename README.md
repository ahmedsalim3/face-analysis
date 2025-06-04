# Face Analysis

A package for analyzing faces in images to detect eye state, gaze direction, and facial expressions.

|   |   |
|-------------|--------------|
| ![demo](./data/demo.png) |
|   |   |

## Features

- **Gaze Detection**: Determine gaze direction using ResNet models
- **Eye State Classification**: Detect whether eyes are open or closed
- **Emotion Recognition**: Identify facial expressions and emotions


## How to install:

1. Create a virtual environment and install dependencies:

```sh
make install
```

This will:
- Install all dependencies using `uv`
- Set up pre-commit hooks for code quality

2. Activate the environment:

```sh
source .venv/bin/activate
```

## Usage Examples

### Command Line Interface

The package provides several command-line tools for analyzing faces in images:

1. Analyze a single image:
```sh
bash scripts/run_analysis.sh single input/test_1.png
```

2. Analyze all images in a folder:
```sh
bash scripts/run_analysis.sh folder input/
```

3. Select the best image from a folder:
```sh
bash scripts/run_analysis.sh best input/
```

Output will be saved to `output/<command>/` directory.

### Python API

#### Gaze Detection

```python
from face_analysis.gazes import Pipeline as GazesPipeline
from face_analysis.gazes import render as GazesRender

gaze_pipeline = GazesPipeline(
    arch='ResNet50',  # Options: "ResNet18", "ResNet34", "ResNet101", "ResNet152"
    detector="retinaface",  # Options: "mtcnn"
    device="cuda",  # or "cpu"
)

img_in = cv2.imread("input/test_1.png")
results = gaze_pipeline.step(img_in)
img_out = GazesRender(img_in, results)
```

|   |   |
|-------------|--------------|
| ![input](./input/test_2.jpg) | ![output](./output/annotated/test_2_gazes.png) |
|   |   |

#### Eye State Detection

```python
from face_analysis.eyes import Pipeline as EyesPipeline
from face_analysis.eyes import render as eyes_render

eye_pipeline = EyesPipeline(
    detector="retinaface", # or "dlib"
    device="cpu", # or "cuda"
)

img_in = cv2.imread(img_path)
results = eye_pipeline.step(img_in)
img_out = eyes_render(img_in, results)
```

|   |   |
|-------------|--------------|
| ![input](./input/test_2.jpg) | ![output](./output/annotated/test_2_eyes.png) |
|   |   |

#### Emotion Detection

```python
from face_analysis.emotions import Pipeline as EmotionsPipeline
from face_analysis.emotions import render as emotions_render

emotion_pipeline = EmotionsPipeline(
    detector= "retinaface", # or "mtcnn", or "cascade"
    device= "cpu",
)

img_in = cv2.imread(img_path)
results = emotion_pipeline.step(img_in)
img_out = emotions_render(img_in, results)
```

|   |   |
|-------------|--------------|
| ![input](./input/test_2.jpg) | ![output](./output/annotated/test_2_emotions.png) |
|   |   |


## Repo Structure

```sh
project_root/
├── data/       
├── input/
├── output/
├── scripts/
├── face_analysis/                 
│
├── LICENSE.txt
├── pyproject.toml
├── README.md
├── requirements.txt
├── uv.lock
└── Makefile
```