import streamlit as st
from PIL import Image
import os
import sys

# Ensure texo package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_engine.inference import MathPredictor

# --- CONFIGURE PAGE ---
st.set_page_config(page_title="Latex Scanner", page_icon="📝", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .main .block-container {padding-top: 2rem;}
    h1 {text-align: center; color: #4F8BF9;}
    .stButton>button {width: 100%; border-radius: 10px; height: 3em;}
    .reportview-container .main .block-container{ max-width: 900px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📝 Latex Scanner")
st.markdown("### Handwritten Math Text Recognition & Solver")
st.info("Upload an image of a handwritten math formula to get the LaTeX code and solution.")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_source = st.radio("Model Source", ["Local / Uploaded", "HuggingFace Hub"])
    
    if model_source == "HuggingFace Hub":
        hf_repo_id = st.text_input("HF Repo ID", value="microsoft/trocr-base-handwritten")
        st.warning("Note: Custom preprocessing improvements are only available with local models using ai_engine.")
    else:
        st.markdown("**Note:** Local weights must be in `weight/` folder.")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_predictor(source_type, input_path):
    try:
        if source_type == "Local / Uploaded":
            # Use our improved MathPredictor
            predictor = MathPredictor()
            return predictor, None
        else:
            # Fallback for HF models (basic loading)
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor = TrOCRProcessor.from_pretrained(input_path)
            model = VisionEncoderDecoderModel.from_pretrained(input_path).to(device)
            return (processor, model), None
    except Exception as e:
        return None, str(e)

# --- LOAD MODEL ---
predictor = None
hf_components = None
error = None

if model_source == "Local / Uploaded":
    predictor, error = load_predictor("Local / Uploaded", None)
else:
    if hf_repo_id:
        hf_components, error = load_predictor("HuggingFace Hub", hf_repo_id)
    else:
        error = "Please enter a HuggingFace Repo ID."

if error:
    st.error(f"❌ Error loading model: {error}")
    st.warning("⚠️ If running on Streamlit Cloud, default local weights are missing (too large for GitHub). Please upload your model to Hugging Face Hub and select that option.")

# --- MAIN UI ---
col1, col2 = st.columns([1, 1])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Input Image")
        st.image(image, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.subheader("✅ Result")
        
        if (predictor) or (hf_components):
            with st.spinner("🧠 Scanning & Recognizing..."):
                try:
                    generated_text = ""
                    
                    if predictor:
                        # Use improved inference
                        generated_text = predictor.predict(image)
                    elif hf_components:
                        # Use basic inference for HF models
                        processor, model = hf_components
                        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device)
                        generated_ids = model.generate(pixel_values)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    st.success("Recognition Complete!")
                    st.code(generated_text, language="latex")
                    st.markdown(f"**Rendered:**")
                    st.latex(generated_text)
                    
                    # Store history
                    if "history" not in st.session_state:
                         st.session_state.history = []
                    if generated_text not in st.session_state.history:
                         st.session_state.history.append(generated_text)
                         
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
             st.warning("Model not loaded.")

# --- HISTORY ---
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.markdown("### 📜 Recent Scans")
    for item in reversed(st.session_state.history[-5:]):
         st.latex(item)
