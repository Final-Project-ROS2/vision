#!/home/group11/vision_venv/bin/python3
"""
Benchmark Dashboard Entry Point
Ensures execution within vision_venv
"""

import sys
import os

# Ensure we're using the venv
venv_path = "/home/group11/vision_venv"
if not sys.prefix.startswith(venv_path):
    print(f"WARNING: Not running in vision_venv!")
    print(f"Current Python: {sys.executable}")
    print(f"Expected: {venv_path}/bin/python3")
    print()

# Import and run the main function
from vision.benchmark_dashboard import main

if __name__ == '__main__':
    main()
