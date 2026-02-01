#!/usr/bin/env python3
"""
Main launcher for the refactored vision pipeline

This script can launch individual components or the full pipeline.
"""

import sys
import argparse
import rclpy
from rclpy.executors import MultiThreadedExecutor

def launch_sam_detector():
    """Launch SAM detector"""
    from .core.sam_detector import SAMDetector
    return SAMDetector()

def launch_clip_classifier():
    """Launch CLIP classifier"""
    from .core.clip_classifier import CLIPClassifier
    return CLIPClassifier()

def launch_grasp_detector():
    """Launch Grasp detector"""
    from .core.grasp_detector import GraspNetDetector
    return GraspNetDetector()

def launch_scene_understanding():
    """Launch Scene understanding"""
    from .core.scene_understanding import SceneUnderstanding
    return SceneUnderstanding()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Vision Pipeline Launcher')
    parser.add_argument('component', choices=['sam', 'clip', 'grasp', 'scene', 'all'],
                       help='Component to launch')
    parser.add_argument('--real-hardware', action='store_true',
                       help='Use real hardware camera topics')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        nodes = []
        
        # Create requested nodes
        if args.component == 'sam' or args.component == 'all':
            nodes.append(launch_sam_detector())
            
        if args.component == 'clip' or args.component == 'all':
            nodes.append(launch_clip_classifier())
            
        if args.component == 'grasp' or args.component == 'all':
            nodes.append(launch_grasp_detector())
            
        if args.component == 'scene' or args.component == 'all':
            nodes.append(launch_scene_understanding())
        
        if not nodes:
            print("No valid component specified")
            return
        
        # Set hardware parameter if specified
        if args.real_hardware:
            for node in nodes:
                node.set_parameters([rclpy.Parameter('real_hardware', rclpy.Parameter.Type.BOOL, True)])
        
        # Run executor
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        
        print(f"Starting {args.component} component(s)...")
        print("Press Ctrl+C to shutdown")
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            executor.shutdown()
            for node in nodes:
                node.destroy_node()
    
    except Exception as e:
        print(f"Failed to start vision pipeline: {e}")
    
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()