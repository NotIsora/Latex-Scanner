import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

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
    else:
        st.markdown("**Note:** Local weights must be in `weight/` folder.")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model(source_type, input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if source_type == "Local / Uploaded":
            model_path = "weight"
            if not os.path.exists(os.path.join(model_path, "config.json")):
                 return None, None, f"Model not found in {model_path}. Please running locally or switch to HuggingFace."
        else:
            model_path = input_path
            
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        return processor, model, None
    except Exception as e:
        return None, None, str(e)

# --- LOAD MODEL ---
if model_source == "Local / Uploaded":
    processor, model, error = load_model("Local / Uploaded", None)
else:
    if hf_repo_id:
        processor, model, error = load_model("HuggingFace Hub", hf_repo_id)
    else:
        processor, model, error = None, None, "Please enter a HuggingFace Repo ID."

if error:
    st.error(f"❌ Error loading model: {error}")
    st.warning("⚠️ If running on Streamlit Cloud, default local weights are missing (too large for GitHub). Please upload your model to Hugging Face Hub and select that option, or use 'microsoft/trocr-base-handwritten' for testing.")

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
        
        if processor and model:
            with st.spinner("🧠 Scanning & Recognizing..."):
                try:
                    # Inference
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
