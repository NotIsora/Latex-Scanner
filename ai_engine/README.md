# AI Engine Module

This directory contains the core AI logic for the Handwritten Math OCR system.

## Structure

- `data/`: Store dataset (CROHME, etc.)
- `models/`: Store model checkpoints (`.pth`, `.onnx`)
- `notebooks/`: Jupyter notebooks for EDA and experiments.

## Model Architecture (Planned)

### Vision Encoder
- **Architecture**: Vision Transformer (ViT) or ResNet.
- **Input**: Grayscale/Binary images of math formulas.
- **Output**: Feature vectors.

### Text Decoder
- **Architecture**: Transformer Decoder (GPT-2 style or RoBERTa).
- **Input**: Feature vectors from Encoder.
- **Output**: LaTeX token sequence.

## Training Strategy
1. **Pre-training**: Train on synthetic data (rendered LaTeX from arxiv).
2. **Fine-tuning**: Fine-tune on CROHME dataset with heavy augmentation.
