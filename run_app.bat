@echo off
echo ===================================================
echo             LATEX SCANNER - LAUNCHER
echo ===================================================

echo [1/3] Installing Dependencies...
echo ---------------------------------------------------
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
rem ai_engine requirements are likely covered but let's be safe if it exists
if exist ai_engine\requirements.txt pip install -r ai_engine\requirements.txt

echo.
echo [2/3] Starting Backend (FastAPI)...
echo ---------------------------------------------------
start "Latex Scanner Backend" cmd /k "cd backend/app && uvicorn main:app --reload --host 0.0.0.0 --port 8000 > ../../backend.log 2>&1"

echo.
echo.
echo Waiting for backend to initialize (15s)...
echo Please wait until you see "Uvicorn running on http://0.0.0.0:8000" in the backend window.
timeout /t 15 /nobreak >nul

echo.
echo [3/3] Starting Frontend (Streamlit)...
echo ---------------------------------------------------
cd frontend
streamlit run app.py
cd ..

echo.
echoDone!
pause
