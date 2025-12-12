#!/usr/bin/env python3
"""
Simple script to run the Music Store Support Bot server.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("🎵 RESONANCE - Music Store Support Bot")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:8000")
    print()
    print("Features:")
    print("  • Beautiful custom UI with stunning aesthetics")
    print("  • Full LangGraph agent functionality")
    print("  • Human-in-the-loop approval flows")
    print("  • Account verification via SMS")
    print("  • Music catalog browsing & purchases")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    print()
    
    # Run uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    main()
