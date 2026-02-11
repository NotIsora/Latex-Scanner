# ✨ Latex Scanner

**Latex Scanner** is an AI-powered application that recognizes handwritten mathematical formulas and automatically solves them. The system combines a **Vision Transformer (ViT)** for image recognition with symbolic mathematics libraries to provide detailed solutions.

---

## 🚀 Features

- **Handwritten Math OCR**: Accurately recognizes handwritten math formulas using a **Vision Encoder-Decoder** model finetuned on the **CROHME** dataset.
- **Smart Solver**: Automatically solves the recognized formulas (integration in progress).
- **High Performance**: Achieved **~10.8% Character Error Rate (CER)** on validation set.
- **Robustness**: Trained with advanced data augmentation (Rotation, Gaussian Noise, Elastic Transform) to handle various handwriting styles.

## 🛠 Tech Stack

- **Model Architecture**: Vision Encoder-Decoder (TrOCR style)
- **Backbone**: Vision Transformer (ViT) / RoBERTa
- **Dataset**: `Neeze/CROHME-full`
- **Training**: PyTorch, HuggingFace Transformers, Albumentations (for augmentation), BitsAndBytes (8-bit optimization)
- **Backend / Frontend**: Python, Streamlit

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/NotIsora/Latex-Scanner.git
    cd Latex-Scanner
    ```

2.  **Install dependencies**:
    ```bash
    # It is recommended to use a virtual environment
    python -m venv venv
    .\venv\Scripts\activate
    
    # Install requirements
    pip install -r backend/requirements.txt
    pip install -r frontend/requirements.txt
    pip install -r ai_engine/requirements.txt
    ```

## 🎮 Usage

To start the application, simply run the launcher script:

```bash
run_app.bat
```

This script will automatically:
1.  Install necessary dependencies.
2.  Start the **FastAPI Backend** server.
3.  Launch the **Streamlit Frontend** interface.

Once started, the application will be accessible at `http://localhost:8501`.

## 📂 Project Structure

```
Latex Scanner/
├── ai_engine/          # AI Model training and inference logic
├── backend/            # FastAPI backend server
├── frontend/           # Streamlit web interface
├── weight/             # Pre-trained model weights
├── run_app.bat         # All-in-one launcher script
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
