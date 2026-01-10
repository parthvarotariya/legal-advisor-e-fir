@echo off
echo ============================================================
echo Starting Legal Advisor e-FIR System
echo ============================================================
echo.

echo [1/2] Starting ML API Server...
start "ML API Server" cmd /k "cd ml && python api_server.py"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend Development Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo Both servers are starting!
echo ============================================================
echo.
echo ML API Server will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:5173
echo.
echo Press any key to close this window (servers will keep running)
pause >nul
