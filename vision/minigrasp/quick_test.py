#!/usr/bin/env python3
"""
MiniGrasp Quick Test - Minimal setup to test your camera
Run this first to verify everything works!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minigrasp import simple_grasp_detector

def main():
    print("=" * 70)
    print("MiniGrasp Quick Test")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Connect to your RealSense camera")
    print("  2. Capture one depth image")
    print("  3. Remove table/floor background (RANSAC)")
    print("  4. Show point cloud statistics")
    print("  5. Detect the best grasp")
    print("  6. Visualize the result")
    print("\n⚠ NOTE: Approach angle filter is DISABLED for easier testing")
    print("        Enable it later in simple_grasp_detector.py if needed")
    print("\n" + "=" * 70)
    
    input("\nPress ENTER to start...")
    
    # Run detection with visualization
    best_grasp = simple_grasp_detector.detect_best_grasp(visualize=True)
    
    print("\n" + "=" * 70)
    
    if best_grasp:
        print("SUCCESS! \u2713")
        print("=" * 70)
        print(f"\nBest Grasp Found:")
        print(f"  Position (x,y,z): [{best_grasp['position'][0]:.3f}, "
              f"{best_grasp['position'][1]:.3f}, {best_grasp['position'][2]:.3f}] m")
        print(f"  Confidence: {best_grasp['score']:.1%}")
        print(f"  Gripper width: {best_grasp['width']*100:.1f} cm")
        print(f"\nYou can now use this grasp pose to control your robot!")
        
    else:
        print("NO GRASP FOUND \u2717")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  1. Check the point cloud statistics printed above")
        print("  2. If 'Too few points after filtering':")
        print("     - Edit minigrasp/config.py")
        print("     - Adjust WORKSPACE_BOUNDS to match the point cloud range")
        print("  3. If 'No point cloud data':")
        print("     - Check camera connection: realsense-viewer")
        print("     - Verify camera permissions")
        print("  4. Place an object in front of the camera and try again")
        
    print("=" * 70)

if __name__ == "__main__":
    main()
