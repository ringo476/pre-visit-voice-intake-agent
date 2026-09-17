@echo off
cd /d "%~dp0..\backend"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8080
