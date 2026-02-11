from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image
import sys
import os

# Add ai_engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ai_engine.inference import MathPredictor

app = FastAPI(
    title="Latex Scanner API",
    description="Backend for Latex Scanner",
    version="0.1.0"
)

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8501", # Streamlit default port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model globally
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = MathPredictor()
        print("MathPredictor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize MathPredictor: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to Antigravity Math Solver API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_math(file: UploadFile = File(...)):
    if predictor is None:
        return {"error": "Model not initialized"}

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Model inference
    try:
        latex_result = predictor.predict(image)
        # TODO: Implement Solver here
        solution_result = "Solver not implemented yet" 
        
        return {
            "filename": file.filename,
            "latex": latex_result,
            "solution": solution_result,
            "confidence": 0.95 # Mock confidence for now
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
