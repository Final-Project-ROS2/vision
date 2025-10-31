#!/bin/bash

# ROS2 SAM Vision Pipeline Installation Script
# Automated setup for Ubuntu 22.04 + ROS2 Humble

set -e  # Exit on any error

echo "🚀 ROS2 SAM Vision Pipeline Installation"
echo "=========================================="

# Check if running on Ubuntu
if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu" /etc/os-release; then
    echo "❌ This script is designed for Ubuntu. Please install manually."
    exit 1
fi

# Check Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
if [[ "$UBUNTU_VERSION" != "22.04" ]]; then
    echo "⚠️ Warning: This script is tested on Ubuntu 22.04. Current version: $UBUNTU_VERSION"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📋 System Information:"
echo "   OS: $(lsb_release -ds)"
echo "   Kernel: $(uname -r)"
echo "   Architecture: $(uname -m)"
echo

# Update system
echo "🔄 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install ROS2 Humble if not present
if ! command -v ros2 &> /dev/null; then
    echo "📦 Installing ROS2 Humble..."
    
    # Add ROS2 repository
    sudo apt install software-properties-common -y
    sudo add-apt-repository universe -y
    sudo apt update
    
    # Install curl if not present
    sudo apt install curl -y
    
    # Add ROS2 GPG key
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    
    # Add repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    # Install ROS2
    sudo apt update
    sudo apt install ros-humble-desktop -y
    
    # Install development tools
    sudo apt install ros-dev-tools -y
    
    echo "✅ ROS2 Humble installed!"
else
    echo "✅ ROS2 already installed: $(ros2 --version)"
fi

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-scipy \
    python3-sklearn \
    python3-pandas \
    python3-pil \
    python3-yaml \
    git \
    wget \
    curl

# Install ROS2 packages
echo "📦 Installing ROS2 packages..."
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-geometry \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-plugins \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-rviz2 \
    ros-humble-image-view \
    ros-humble-usb-cam \
    ros-humble-v4l2-camera

# Create workspace if it doesn't exist
WORKSPACE_DIR="$HOME/ros2_ws"
if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo "📁 Creating ROS2 workspace at $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR/src"
fi

cd "$WORKSPACE_DIR"

# Install Python dependencies for vision pipeline
echo "🐍 Installing Python dependencies..."
pip3 install --user \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    opencv-python>=4.8.0 \
    pillow>=10.0.0 \
    tqdm>=4.66.0 \
    jsonschema>=4.19.0

# Optional: Install CUDA support (if NVIDIA GPU detected)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected. Installing CUDA support..."
    pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected. Using CPU-only PyTorch."
fi

# Clone or copy vision package (assuming we're already in the package)
VISION_PKG_PATH="$WORKSPACE_DIR/src/vision"
if [[ ! -d "$VISION_PKG_PATH" ]]; then
    echo "📁 Setting up vision package..."
    mkdir -p "$VISION_PKG_PATH"
    
    # Copy current directory contents to workspace
    CURRENT_DIR=$(dirname $(readlink -f $0))
    if [[ -f "$CURRENT_DIR/../package.xml" ]]; then
        cp -r "$CURRENT_DIR/.." "$VISION_PKG_PATH"
        echo "✅ Vision package copied to workspace"
    else
        echo "⚠️ Please manually copy the vision package to $VISION_PKG_PATH"
    fi
fi

# Build the workspace
echo "🔨 Building ROS2 workspace..."
cd "$WORKSPACE_DIR"
colcon build --packages-select vision

# Source the workspace
echo "📝 Setting up environment..."
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source $WORKSPACE_DIR/install/setup.bash" >> ~/.bashrc

# Create desktop launcher (optional)
DESKTOP_FILE="$HOME/Desktop/DINO_Vision_Pipeline.desktop"
if [[ -d "$HOME/Desktop" ]]; then
    echo "🖥️ Creating desktop launcher..."
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DINO Vision Pipeline
Comment=Launch ROS2 DINO Vision Pipeline
Exec=gnome-terminal -- bash -c "source /opt/ros/humble/setup.bash && source $WORKSPACE_DIR/install/setup.bash && ros2 launch vision dino_pipeline.launch.py; exec bash"
Icon=camera-web
Terminal=false
Categories=Development;Science;
EOF
    chmod +x "$DESKTOP_FILE"
    echo "✅ Desktop launcher created"
fi

# Download sample models (optional)
echo "📥 Downloading sample models..."
MODEL_DIR="$VISION_PKG_PATH/Final-proj"
if [[ -d "$MODEL_DIR" ]] && [[ ! -f "$MODEL_DIR/sam_vit_b_01ec64.pth" ]]; then
    echo "   Downloading SAM model (this may take a while)..."
    cd "$MODEL_DIR"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth || echo "⚠️ SAM model download failed (optional)"
fi

echo
echo "🎉 Installation Complete!"
echo "========================"
echo
echo "📋 Installation Summary:"
echo "   ✅ ROS2 Humble installed"
echo "   ✅ System dependencies installed"
echo "   ✅ Python packages installed"
echo "   ✅ Vision package built"
echo "   ✅ Environment configured"
echo
echo "🚀 Quick Start:"
echo "   1. Open new terminal (to load environment)"
echo "   2. Run: ros2 launch vision dino_pipeline.launch.py"
echo "   3. Or use desktop launcher if created"
echo
echo "📚 Documentation:"
echo "   - README: $VISION_PKG_PATH/README.md"
echo "   - Config: $VISION_PKG_PATH/config/"
echo "   - Examples: $VISION_PKG_PATH/Final-proj/data/test_images/"
echo
echo "🔧 Testing:"
echo "   ros2 run vision vision_demo"
echo
echo "⚠️ Note: Close and reopen terminal to load new environment"
echo

# Test installation
echo "🧪 Testing installation..."
source "$WORKSPACE_DIR/install/setup.bash"

if ros2 pkg list | grep -q "^vision$"; then
    echo "✅ Vision package successfully installed and detected"
else
    echo "❌ Vision package not detected. Check build output above."
    exit 1
fi

echo "🎯 Ready to run SAM Vision Pipeline!"