# ✨ Latex Scanner

**Latex Scanner** is an AI-powered application that recognizes handwritten mathematical formulas and automatically solves them. The system combines a **Vision Transformer (ViT)** for image recognition with symbolic mathematics libraries to provide detailed solutions.

---

## 🚀 Features

- **Handwritten Math OCR**: Accurately recognizes handwritten math formulas and converts them to LaTeX.
- **Smart Solver**: Automatically solves the recognized formulas and provides step-by-step solutions (integration in progress).
- **User-Friendly Interface**: Clean and modern Web UI built with Streamlit.
- **High Performance**: Optimized for low latency inference.

## 🛠 Tech Stack

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Vision Backbone**: Vision Transformer (ViT) / ResNet-101
- **Decoder**: GPT-2 / RoBERTa
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Math Engine**: SymPy

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
