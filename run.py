import io
import os
import subprocess
import sys
import time
import urllib.request

BACKEND_HOST = os.environ.get("AMRIT_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("AMRIT_BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("AMRIT_FRONTEND_PORT", "8510"))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def wait_for_backend(url: str, process: subprocess.Popen, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def stop(*processes):
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    # Flush each status line immediately, even when output is piped to a log file
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    # Child processes log via stdout; force UTF-8 so non-ASCII text never crashes
    # them when output is redirected or the console uses a legacy code page.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    print("Starting SIH-26 AMrit Application...")

    print(f"-> Starting FastAPI Backend on port {BACKEND_PORT}...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
        cwd=PROJECT_DIR,
        env=child_env,
    )

    backend_url = f"http://127.0.0.1:{BACKEND_PORT}"
    print("-> Waiting for models to load...")
    if not wait_for_backend(f"{backend_url}/api/v1/health", backend_process):
        print("ERROR: Backend failed to start. See the log output above.")
        stop(backend_process)
        sys.exit(1)

    print(f"-> Starting Streamlit Frontend on port {FRONTEND_PORT}...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", str(FRONTEND_PORT), "--server.headless", "true"],
        cwd=PROJECT_DIR,
        env={**child_env, "AMRIT_API_URL": backend_url},
    )

    try:
        print("\nApplication is running!")
        print(f"Frontend:     http://localhost:{FRONTEND_PORT}")
        print(f"Backend API:  {backend_url}/docs")
        print("\nPress Ctrl+C to stop both servers.")

        while backend_process.poll() is None and frontend_process.poll() is None:
            time.sleep(1)
        print("\nA server exited unexpectedly; shutting down.")
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        stop(backend_process, frontend_process)
        print("Gracefully stopped.")


if __name__ == "__main__":
    main()
