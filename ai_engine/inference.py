from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

class MathPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to ../weight relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "weight")
        
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True).to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_path_or_obj):
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert("RGB")
        else:
            image = image_path_or_obj.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

if __name__ == "__main__":
    # Test block
    try:
        predictor = MathPredictor()
        print("Predictor initialized.")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
