import os
import sys

# Ensure texo package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoConfig, AutoModel
try:
    from texo.model.hgnet2 import HGNetv2, HGNetv2Config
    # Register the custom model
    AutoConfig.register("my_hgnetv2", HGNetv2Config)
    AutoModel.register(HGNetv2Config, HGNetv2)
except ImportError:
    print("Warning: Could not import texo package. Custom model 'my_hgnetv2' might fail to load.")

from PIL import Image
import torch

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

    def process_image(self, image, size=(384, 384)):
        """
        Resize image while maintaining aspect ratio, then pad to target size.
        """
        image = image.convert("RGB")
        w, h = image.size
        target_w, target_h = size
        
        # Calculate new size maintaining aspect ratio
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        # Create a new image with white background
        new_image = Image.new("RGB", size, (255, 255, 255))
        # Paste the resized image in the center
        new_image.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        
        return new_image

    
    def predict(self, image_path_or_obj, threshold=0.0):
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else:
            image = image_path_or_obj
            
        # Preprocess image
        processed_image = self.process_image(image)

        # Get pixel values
        pixel_values = self.processor(images=processed_image, return_tensors="pt").pixel_values.to(self.device)

        # Generate with improved parameters
        # We set num_return_sequences = num_beams to get all candidates
        num_beams = 4
        outputs = self.model.generate(
            pixel_values,
            num_beams=num_beams,
            num_return_sequences=num_beams, # Return all beams
            max_length=512,
            early_stopping=True,
            no_repeat_ngram_size=0,
            return_dict_in_generate=True,
            output_scores=True,
            return_legacy_cache=False,
            # length_penalty=1.0, # Default
        )
        
        generated_sequences = outputs.sequences
        sequences_scores = outputs.sequences_scores
        
        candidates = []
        
        # Batch decode all sequences
        decoded_texts = self.processor.batch_decode(generated_sequences, skip_special_tokens=True)
        
        for i, text in enumerate(decoded_texts):
            # Calculate confidence
            score = sequences_scores[i].item()
            confidence = torch.exp(torch.tensor(score)).item()
            
            if confidence >= threshold:
                candidates.append({
                    "text": text,
                    "confidence": confidence
                })
        
        # Sort by confidence descending
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return candidates

if __name__ == "__main__":
    # Test block
    try:
        predictor = MathPredictor()
        print("Predictor initialized.")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
