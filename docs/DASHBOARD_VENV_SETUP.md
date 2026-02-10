# 🎯 Benchmark Dashboard - vision_venv Setup

## Quick Start with vision_venv

### Option 1: Using the venv launcher script (Recommended)

```bash
# Start dashboard with vision_venv
./src/vision/testsh/start_benchmark_venv.sh
```

This script will:
- ✓ Check for vision_venv existence
- ✓ Activate the virtual environment
- ✓ Verify Python version and packages
- ✓ Launch the dashboard
- ✓ Auto-open browser at http://localhost:8080

### Option 2: Manual activation

```bash
# Activate vision_venv
source /home/group11/vision_venv/bin/activate

# Run dashboard
./src/vision/testsh/start_benchmark_dashboard.sh
```

### Option 3: Direct Python execution

```bash
# Activate venv and run directly
source /home/group11/vision_venv/bin/activate
cd /home/group11/final_project_ws/src/vision
/home/group11/vision_venv/bin/python3 vision/benchmark_dashboard.py
```

## Setup Instructions

### 1. Ensure vision_venv is configured

```bash
# If vision_venv doesn't exist, create it
python3 -m venv /home/group11/vision_venv

# Activate it
source /home/group11/vision_venv/bin/activate

# Install requirements
cd /home/group11/final_project_ws/src/vision
pip install -r requirements.txt

# Install ROS2 Python packages
pip install opencv-python numpy
```

### 2. Source ROS2 environment

```bash
source /opt/ros/humble/setup.bash  # or your ROS2 distro
source ~/final_project_ws/install/setup.bash
```

### 3. Start the dashboard

```bash
./src/vision/testsh/start_benchmark_venv.sh
```

## Verification

Check that dashboard is using vision_venv:

```bash
# After starting dashboard, check in the console output:
# You should see:
#   ✓ Using Python: /home/group11/vision_venv/bin/python3
#   ✓ Python version: Python 3.10.x
#   ✓ rclpy installed
#   ✓ opencv-python installed
```

## Troubleshooting

### vision_venv not found

```bash
# Create the virtual environment
python3 -m venv /home/group11/vision_venv

# Activate and install requirements
source /home/group11/vision_venv/bin/activate
cd /home/group11/final_project_ws/src/vision
pip install -r requirements.txt
```

### ROS2 packages not available in venv

The dashboard needs both venv packages AND ROS2 packages. Make sure:

1. ROS2 is sourced BEFORE activating venv
2. Or set PYTHONPATH to include ROS2:

```bash
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
source /home/group11/vision_venv/bin/activate
```

### Dashboard won't start

Check Python interpreter:
```bash
source /home/group11/vision_venv/bin/activate
which python3
# Should show: /home/group11/vision_venv/bin/python3

python3 -c "import rclpy; print('ROS2 OK')"
python3 -c "import cv2; print('OpenCV OK')"
```

## Files Modified for venv

1. **`vision/benchmark_dashboard.py`**
   - Shebang changed to: `#!/home/group11/vision_venv/bin/python3`

2. **`testsh/start_benchmark_dashboard.sh`**
   - Added venv activation

3. **`testsh/start_benchmark_venv.sh`** (NEW)
   - Dedicated venv launcher with validation

4. **`scripts/benchmark_dashboard_venv.py`** (NEW)
   - Entry point that ensures venv usage

## Complete Workflow

```bash
# Terminal 1: Source ROS2 and activate venv
source /opt/ros/humble/setup.bash
source ~/final_project_ws/install/setup.bash
source /home/group11/vision_venv/bin/activate

# Start all services
ros2 run vision pixel_to_real_service &
ros2 run vision simple_sam_detector &
ros2 run vision clip_classifier &
ros2 run vision graspnet_detector &
ros2 run vision scene_understanding &

# Terminal 2: Start dashboard with venv
cd ~/final_project_ws
./src/vision/testsh/start_benchmark_venv.sh

# Terminal 3: Generate test data
source /home/group11/vision_venv/bin/activate
./src/vision/testsh/test_benchmark_dashboard.sh
```

## Environment Variables

The dashboard works best with these environment variables set:

```bash
# In your ~/.bashrc or session
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
export PATH=/home/group11/vision_venv/bin:$PATH
```

## Alternative: System-wide Installation

If you prefer not to use venv, you can install packages system-wide:

```bash
# Install to system Python
pip3 install opencv-python numpy --user

# Run without venv activation
ros2 run vision benchmark_dashboard
```

But **vision_venv is recommended** for:
- ✓ Isolated dependencies
- ✓ Consistent Python versions
- ✓ Avoiding system conflicts
- ✓ Reproducible environment

---

**Recommended**: Use `./testsh/start_benchmark_venv.sh` for guaranteed venv usage!
