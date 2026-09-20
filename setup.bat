@echo off
chcp 65001 >nul
cd /d %~dp0
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete.
echo Edit .env and set OPENROUTER_API_KEY and UPSTAGE_API_KEY.
echo Then run start.bat or: python chat.py
echo.
pause
