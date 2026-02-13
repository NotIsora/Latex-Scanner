import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoConfig, AutoModel
import sys
import os

# Add current directory to path so we can import texo
sys.path.append(os.getcwd())

try:
    from texo.model.hgnet2 import HGNetv2, HGNetv2Config
    print("Successfully imported HGNetv2 from texo package.")
    
    # Register the custom model
    AutoConfig.register("my_hgnetv2", HGNetv2Config)
    AutoModel.register(HGNetv2Config, HGNetv2)
    print("Registered custom model 'my_hgnetv2'.")

    weights_dir = "./weight"
    print(f"Loading model from {weights_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    print("Tokenizer loaded successfully.")
    
    model = VisionEncoderDecoderModel.from_pretrained(weights_dir)
    print("Model loaded successfully.")
    
    print("Model config:", model.config)
    print("Verification complete!")
    
except ImportError as e:
    print(f"Failed to import texo: {e}")
    print("Make sure the 'texo' folder is in the current directory.")
    exit(1)
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
