# ✨ Latex Scanner

**Latex Scanner** is an AI-powered application that recognizes handwritten mathematical formulas and automatically solves them. The system utilizes a **Universal Model for Image-to-Formula Recognition (UniMER)** architecture, combining a robust **HGNetv2** backbone with a transformer decoder.

---

## 🚀 Features

- **Universal Math OCR**: Fine-tuned on the massive **UniMER** dataset, capable of recognizing complex handwritten formulas with high accuracy in real-world scenarios.
- **Advanced Architecture**: Uses a **Vision Encoder-Decoder** with a custom **HGNetv2** backbone (optimized for feature extraction) and an **mBART** decoder.
- **Smart Preprocessing**: Automatically resizes images while preserving aspect ratio (padding to square) to prevent distortion and improve accuracy on long formulas.
- **Beam Search Decoding**: Uses beam search (num_beams=4) to explore multiple possible character sequences, ensuring higher accuracy than standard greedy decoding.
- **High Performance**: Robust against noise, various handwriting styles, and complex formula structures thanks to the diverse training data.

## 🛠 Tech Stack

- **Model Architecture**: Vision Encoder-Decoder (UniMERNet variant)
- **Backbone**: **HGNetv2** (High Performance GPU Network v2)
- **Decoder**: mBART
- **Datasets**: 
    - **Training Data**: **UniMER Dataset** (Universal Mathematical Expression Recognition)
- **Training**: 
    - **Optimization**: Layer-wise Learning Rate Decay (LLDR), AdamW
    - **Frameworks**: PyTorch, HuggingFace Transformers, bitsandbytes
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
    pip install -r requirements.txt
    ```

## 🎮 Usage

To start the application, simply run the launcher script:

```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```

Or use the provided batch script if available:
```bash
run_app.bat
```

Once started, the application will be accessible at `http://localhost:8501`.

## 📂 Project Structure

```
Latex Scanner/
├── ai_engine/          # AI Model logic (inference, preprocessing)
├── weight/             # Pre-trained model weights
├── streamlit_app.py    # Main application entry point
├── run_app.bat         # Launcher script for Windows
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
