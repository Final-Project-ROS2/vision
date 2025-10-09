@echo off
REM Build script for the vision ROS2 package

echo Building vision package...

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Check Python syntax
echo Checking Python syntax...
python -m py_compile vision\show_rgb_image_node.py
if %errorlevel% neq 0 (
    echo Error: Python syntax check failed
    exit /b 1
)

REM Run setup.py in develop mode
echo Installing package in development mode...
pip install -e .

echo.
echo =====================================
echo Vision package setup completed!
echo =====================================
echo.
echo To use this package in ROS2:
echo 1. Make sure ROS2 is sourced in your environment
echo 2. Copy this package to your ROS2 workspace src folder
echo 3. Run: colcon build --packages-select vision
echo 4. Source your workspace: source install/setup.bash
echo 5. Run the node: ros2 run vision show_rgb_image
echo.