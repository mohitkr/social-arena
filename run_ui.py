"""
Social Arena Web UI
Usage: .venv/bin/python3 run_ui.py
Then open: http://localhost:8000
"""
import sys
import os
import subprocess

def main():
    try:
        import uvicorn
    except ImportError:
        print("Installing missing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "fastapi", "uvicorn", "openai", "google-generativeai", "-q"])

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n{'='*50}")
    print(f"  Social Arena — Web UI")
    print(f"  http://localhost:{port}")
    print(f"{'='*50}\n")

    uvicorn.run(
        "social_arena.ui.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",
    )

if __name__ == "__main__":
    main()
