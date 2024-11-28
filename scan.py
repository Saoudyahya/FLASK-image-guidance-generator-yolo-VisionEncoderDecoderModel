from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor
)
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from collections import defaultdict

app = Flask(__name__)

class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        detected_objects = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            label = self.model.names[int(cls)]
            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
            })
        return detected_objects

class ImageCaptioning:
    def __init__(self, model_name):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_caption(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption


from collections import Counter
from typing import List, Dict
import numpy as np

class UserGuidance:
    @staticmethod
    def generate_guidance(detected_objects: List[Dict], caption: str = "", image_width: int = 640) -> str:
        if not detected_objects:
            return "No objects detected."

        position_groups = {"left": [], "front": [], "right": []}
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            label = obj["label"]
            center_x = (x1 + x2) / 2
            
            # First-person perspective positioning
            # Swapping left and right to match camera view
            if center_x < image_width / 3:
                position_groups["right"].append(label)
            elif center_x > 2 * image_width / 3:
                position_groups["left"].append(label)
            else:
                position_groups["front"].append(label)

        # Convert the guidance to a string directly
        guidance_parts = []
        for position in ["left", "front", "right"]:
            if position_groups[position]:
                counter = Counter(position_groups[position])
                descriptions = []
                for obj, count in counter.most_common():
                    if count > 1:
                        descriptions.append(f"{count} {obj}s")
                    else:
                        descriptions.append(f"a {obj}")

                if descriptions:
                    # Improved joining of descriptions
                    if len(descriptions) > 1:
                        objects_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
                    else:
                        objects_text = descriptions[0]
                    
                    guidance_parts.append(f"To your {position}, there is {objects_text}.")

        return " ".join(guidance_parts) if guidance_parts else "No distinct objects detected in specific positions."

    @staticmethod
    def merge_guidance(all_guidance: List[str], captions: List[str]) -> str:
        # If captions exist, prepend them to the first guidance
        if captions and all_guidance:
            return f"Scene overview: {' '.join(captions)} {all_guidance[0]}"
        elif captions:
            return f"Scene overview: {' '.join(captions)}"
        elif all_guidance:
            return all_guidance[0]
        else:
            return "No scene description available."

class MainApp:
    def __init__(self, yolo_path, caption_model_name):
        self.detector = ObjectDetection(yolo_path)
        self.captioner = ImageCaptioning(caption_model_name)
        self.guidance_generator = UserGuidance()

    def process_single_image(self, image_path):
        detected_objects = self.detector.detect_objects(image_path)
        image = cv2.imread(image_path)
        image_caption = self.captioner.generate_caption(image)
        guidance = self.guidance_generator.generate_guidance(detected_objects)
        
        return {
            "caption": image_caption,
            "detected_objects": detected_objects,
            "guidance": guidance
        }

# Initialize the MainApp with model paths
main_app = MainApp(
    yolo_path="yolov9c.pt",
    caption_model_name="nlpconnect/vit-gpt2-image-captioning"
)

@app.route('/process_images', methods=['POST'])
def process_images():
    if 'images' not in request.files:
        return 'No images provided', 400

    files = request.files.getlist('images')
    if not files:
        return 'No selected files', 400

    results = []

    for file in files:
        if file:
            # Read the image directly from the file object
            img = file.read()

            # Save the image temporarily to process
            img_path = 'temp_image.jpg'
            with open(img_path, 'wb') as temp_file:
                temp_file.write(img)

            try:
                result = main_app.process_single_image(img_path)
                result_dict = {
                    'caption': result['caption'],
                    'detected_objects': result['detected_objects'],
                    'guidance': result['guidance']
                }
                results.append(result_dict)

            except Exception as e:
                return f'Error processing image: {str(e)}', 500

            finally:
                # Clean up temporary image file
                if os.path.exists(img_path):
                    os.remove(img_path)

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
