# Legal Advisor e-FIR System - Quick Start

## 🚀 Easy Start (Recommended)

Just double-click the `start-servers.bat` file in the project root. It will automatically start both servers.

## 📋 Manual Start

If you prefer to start servers manually:

### 1. Start ML API Server
```bash
cd ml
python api_server.py
```
Wait for "✅ Model loaded successfully" message

### 2. Start Frontend (in a new terminal)
```bash
cd frontend
npm run dev
```

## 🌐 Access the Application

- **Frontend**: http://localhost:5173
- **ML API**: http://localhost:5000

## ⚠️ Important Notes

- Make sure Python and Node.js are installed
- Both servers must be running to use the Legal Advice feature
- Keep both terminal windows open while using the application
