import subprocess
import sys
import time
import os

def main():
    print("Starting SIH-26 AMrit Application...")
    
    print("-> Starting FastAPI Backend on port 8000...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
    )
    
    time.sleep(3)
    
    print("-> Starting Streamlit Frontend on port 8510...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8510", "--server.headless", "true"]
    )
    
    try:
        print("\n✅ Application is running!")
        print("🌍 Frontend: http://localhost:8510")
        print("⚙️  Backend API: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both servers.")
        
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("Gracefully stopped.")

if __name__ == "__main__":
    main()
