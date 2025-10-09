#!/usr/bin/env python3
"""
Simple test script to verify the vision node works
Run this in your ROS2 environment to test the node
"""

import sys
import os

# Add the vision package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vision'))

try:
    from vision.show_rgb_image_node import main
    print("✓ Successfully imported show_rgb_image_node")
    
    print("Starting RGB Image Viewer node...")
    print("Press Ctrl+C to stop")
    main()
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you have ROS2 sourced and required packages installed:")
    print("  sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-srvs ros-humble-cv-bridge")
    
except Exception as e:
    print(f"✗ Error: {e}")