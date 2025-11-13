"""Launch the Streamlit web application."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch Streamlit app."""
    app_path = Path(__file__).parent.parent / "app.py"

    if not app_path.exists():
        print(f"❌ Error: app.py not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=true",
    ]

    print("=" * 70)
    print(" 🚀 Starting Faculty Matchmaker Web Interface")
    print("=" * 70)
    print()
    print(f"📍 App will open at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

