#!/bin/bash

# Wrapper script that ensures virtual environment is activated
cd ~/final_project_ws
source vision_venv/bin/activate

# Run the debug script
cd ~/final_project_ws/src/vision
./testsh/debug_clip_find_object.sh
