#!/usr/bin/env python3
"""
Quick test script for the refactored vision pipeline

Tests that all components can be imported and initialized without errors.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all components can be imported"""
    print("Testing imports...")
    
    try:
        # Test utils import
        from vision_refactor.utils.common import VisionNodeBase, OpenCVWindow
        print("✓ Utils imported successfully")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        # Test core imports (without initializing ROS)
        import vision_refactor.core.sam_detector
        import vision_refactor.core.clip_classifier  
        import vision_refactor.core.grasp_detector
        import vision_refactor.core.scene_understanding
        print("✓ Core modules imported successfully")
    except Exception as e:
        print(f"✗ Core imports failed: {e}")
        print(traceback.format_exc())
        return False
    
    return True

def test_dependencies():
    """Test optional dependencies"""
    print("\nTesting dependencies...")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
    except ImportError:
        print("✗ NumPy not available") 
        return False
    
    # Test optional CLIP
    try:
        import torch
        from transformers import CLIPModel
        print(f"✓ CLIP (PyTorch {torch.__version__}) available")
    except ImportError:
        print("⚠ CLIP not available (optional)")
    
    # Test custom interfaces
    try:
        from custom_interfaces.msg import SAMDetections
        from custom_interfaces.srv import DetectObjects
        print("✓ Custom interfaces available")
    except ImportError:
        print("⚠ Custom interfaces not available (optional)")
    
    return True

def main():
    """Run tests"""
    print("Vision Refactor - Quick Test")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test dependencies  
    if not test_dependencies():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Refactored vision pipeline ready.")
        print("\nUsage:")
        print("  python3 -m vision_refactor.launcher sam")
        print("  python3 -m vision_refactor.launcher all")
    else:
        print("✗ Some tests failed. Check dependencies.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())