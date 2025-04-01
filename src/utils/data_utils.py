import os
import json
import logging


def simplified_results(img_name, results):
    """
    Create a simplified version of the results including only:
    1. If subject's eyes are open
    2. If subject is looking at the camera
    3. Subject's smiling score
    """
    simplified = {img_name: {}}

    if "error" not in results and "faces" in results:
        faces_data = results["faces"]
        for face_id, face_data in faces_data.items():

            eyes_open = (
                face_data["left_eye"]["state"] == "open"
                and face_data["right_eye"]["state"] == "open"
            )

            looking_at_camera = face_data["is_looking"]
            smiling_score = face_data["emotion_data"]["emotions"]["scores"]["happy"]

            simplified[img_name][face_id] = {
                "eyes_open": eyes_open,
                "looking_at_camera": looking_at_camera,
                "smiling_score": smiling_score,
            }

    return simplified


def save_to_json(data, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}

            if isinstance(data, dict):
                existing_data.update(data)
                data = existing_data

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False)

        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving to JSON: {e}")
