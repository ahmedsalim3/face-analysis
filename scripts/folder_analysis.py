"""Analyze all images in a folder"""

import json
import argparse
from pathlib import Path
from image_analysis import analyze_image


def process_folder(input_folder, output_folder, device='cpu', detector='retinaface', save_annotated=True):
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    input_path = Path(input_folder)
    for ext in image_extensions:
        image_paths.extend(list(input_path.glob(f"*{ext}")))
        image_paths.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in {input_folder}")
        return {}
    
    print(f"Processing {len(image_paths)} images...")
    
    all_results = {}
    
    for img_path in image_paths:
        print(f"Processing {img_path.name}")
        results = analyze_image(str(img_path), output_folder, device, detector, save_annotated, display=False)
        all_results[img_path.name] = results
    
    # Save combined results
    with open(Path(output_folder) / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Processed {len(all_results)} images")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--detector', default='retinaface')
    parser.add_argument('--save-annotated', action='store_true')
    
    args = parser.parse_args()
    process_folder(args.input, args.output, args.device, args.detector, args.save_annotated)
