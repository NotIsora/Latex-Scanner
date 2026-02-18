@echo off
echo ===================================================
echo             LATEX SCANNER - LAUNCHER
echo ===================================================

echo [1/3] Installing Dependencies...
echo ---------------------------------------------------
pip install -r requirements.txt

echo.
echo.
echo [2/3] Starting Backend (Skipped for Standalone Mode)...
echo ---------------------------------------------------
echo Backend launch skipped to prevent double model loading (OOM). 
echo using standalone streamlit_app.py which loads model directly.

echo.
echo.
echo [3/3] Starting Frontend (Streamlit)...
echo ---------------------------------------------------
streamlit run streamlit_app.py

echo.
echo Done!
pause
