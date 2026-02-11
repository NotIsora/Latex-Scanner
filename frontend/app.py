import streamlit as st
import requests
from PIL import Image
import io

# Cấu hình trang
st.set_page_config(
    page_title="Latex Scanner",
    page_icon="✨",
    layout="wide" # Wide layout for split view
)

# Custom CSS để ẩn Deploy button và tùy chỉnh Dark Mode
st.markdown("""
    <style>
    /* Ẩn nút Deploy và Menu */
    .stDeployButton {display:none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tùy chỉnh vùng upload */
    .stFileUploader {
        border-radius: 10px;
    }
    
    /* Tiêu đề căn giữa */
    h1 {
        text-align: center; 
        margin-bottom: 30px;
        color: #FFD700; /* Vàng kim loại */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ Latex Scanner")

# Layout chia đôi: Trái (Cài đặt/Input) - Phải (Kết quả)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload & Settings")
    
    # 1. Slider Độ tin cậy
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.8,
        help="Only show results if confidence is above this threshold."
    )
    
    # 2. Upload Ảnh
    uploaded_file = st.file_uploader("Upload a math formula image...", type=['png', 'jpg', 'jpeg'])
    
    # 3. Nút Giải toán
    if uploaded_file is not None:
        if st.button("🚀 Scan & Solve", use_container_width=True, type="primary"):
            with st.spinner('Analyzing...'):
                try:
                    # Gọi API Backend
                    files = {"file": uploaded_file.getvalue()}
                    try:
                        response = requests.post("http://localhost:8000/predict", files=files)
                    except requests.exceptions.ConnectionError:
                         st.error("⚠️ Cannot connect to Backend server!")
                         st.info("💡 Please check if the Backend window (cmd) is running. If not, please run `run_app.bat` and wait for 'Uvicorn running on...'")
                         if st.button("🔄 Retry Connection"):
                             st.rerun()
                         st.stop()
                    
                    if response.status_code == 200:
                        st.session_state['result'] = response.json()
                        st.success("Done!")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Unknown error: {str(e)}")

with col2:
    st.subheader("👁️ Preview & Result")
    
    if uploaded_file is not None:
        # Show image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Hiển thị kết quả nếu có trong session_state
        if 'result' in st.session_state:
            result = st.session_state['result']
            conf = result.get("confidence", 1.0)
            
            # Kiểm tra ngưỡng tin cậy (Mocking logic for now as API returns mock)
            # Trong thực tế API nên trả về confidence thật
            if conf >= confidence_threshold:
                st.markdown("### 📝 LaTeX:")
                st.code(result.get("latex", ""), language="latex")
                st.latex(result.get("latex", ""))
                
                solution = result.get("solution", "")
                if solution and "not implemented" not in solution:
                     st.markdown("### 💡 Solution:")
                     st.latex(solution)
            else:
                st.warning(f"Confidence ({conf}) is lower than threshold ({confidence_threshold}). Please try a clearer image.")

    else:
        st.info("👈 Please upload an image on the left.")

