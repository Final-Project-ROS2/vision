"""
MiniGrasp - Simplified Grasp Detection Module

A lightweight single-shot grasp detection system for RealSense depth cameras.
Captures one depth image and returns the best grasp pose.

Usage:
    from minigrasp import simple_grasp_detector
    
    # Quick detection
    best_grasp = simple_grasp_detector.detect_best_grasp(visualize=True)
    
    # Advanced usage
    detector = simple_grasp_detector.SimpleGraspDetector()
    best_grasp = detector.detect_best_grasp()

Configuration:
    from minigrasp import config
    
    # Customize workspace
    config.WORKSPACE_BOUNDS = {'x_min': -0.2, 'x_max': 0.2, ...}
    
    # Use presets
    config.get_table_50cm_config()
"""

__version__ = "1.0.0"
__author__ = "GraspNet Team"

from . import config
from . import simple_grasp_detector

__all__ = ['simple_grasp_detector', 'config']
