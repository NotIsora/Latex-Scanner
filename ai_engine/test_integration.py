import sys
import os
from PIL import Image, ImageDraw

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ai_engine.inference import MathPredictor

def create_dummy_image():
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), "x + y = 2", fill=(0,0,0))
    return img

def test_inference():
    print("Initializing MathPredictor...")
    try:
        predictor = MathPredictor()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    print("Running prediction on dummy image...")
    try:
        img = create_dummy_image()
        result = predictor.predict(img)
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"FAILED to predict: {e}")

if __name__ == "__main__":
    test_inference()
