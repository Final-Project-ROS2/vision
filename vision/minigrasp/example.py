#!/usr/bin/env python3
"""
MiniGrasp Example Usage
Demonstrates how to use the simple grasp detector
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minigrasp import simple_grasp_detector
from minigrasp import config


def example_1_basic_usage():
    """Example 1: Basic usage - detect and print best grasp"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Detect best grasp with visualization
    best_grasp = simple_grasp_detector.detect_best_grasp(visualize=True)
    
    if best_grasp:
        print("\n✓ Successfully detected grasp!")
        print(f"  Position (x, y, z): {best_grasp['position']}")
        print(f"  Confidence score: {best_grasp['score']:.2%}")
        print(f"  Gripper width: {best_grasp['width']*100:.1f} cm")
    else:
        print("\n✗ Failed to detect grasp")


def example_2_custom_config():
    """Example 2: Using custom configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    # Create detector with custom config
    detector = simple_grasp_detector.SimpleGraspDetector()
    
    # Modify configuration for specific setup
    config.WORKSPACE_BOUNDS = {
        'x_min': -0.2, 'x_max': 0.2,
        'y_min': -0.2, 'y_max': 0.2,
        'z_min': 0.25, 'z_max': 0.5,
    }
    config.NUM_GRASP_CANDIDATES = 100  # Generate more candidates
    config.MIN_GRASP_SCORE = 0.7       # Higher quality threshold
    
    print("Custom settings:")
    print(f"  Workspace: {config.WORKSPACE_BOUNDS}")
    print(f"  Candidates: {config.NUM_GRASP_CANDIDATES}")
    print(f"  Min score: {config.MIN_GRASP_SCORE}")
    
    best_grasp = detector.detect_best_grasp(visualize=False)
    
    if best_grasp:
        print(f"\n✓ Best grasp score: {best_grasp['score']:.3f}")
    else:
        print("\n✗ No grasp found")


def example_3_preset_configs():
    """Example 3: Using preset configurations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Preset Configurations")
    print("=" * 70)
    
    # Use table at 50cm preset
    print("\n[Using 50cm table preset]")
    config.get_table_50cm_config()
    
    detector = simple_grasp_detector.SimpleGraspDetector()
    best_grasp = detector.detect_best_grasp(visualize=False)
    
    if best_grasp:
        print(f"✓ Grasp detected at position: {best_grasp['position']}")


def example_4_extract_pose():
    """Example 4: Extract full 6D pose for robot control"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Extracting Full Grasp Information for Robot Control")
    print("=" * 70)
    
    best_grasp = simple_grasp_detector.detect_best_grasp(visualize=False)
    
    if best_grasp:
        print("\n📋 Complete Grasp Information:")
        print("-" * 40)
        
        # Position
        pos = best_grasp['position']
        print(f"\n1. POSITION (meters):")
        print(f"   x = {pos[0]:+.4f}")
        print(f"   y = {pos[1]:+.4f}")
        print(f"   z = {pos[2]:+.4f}")
        
        # Rotation matrix
        rot = best_grasp['rotation']
        print(f"\n2. ROTATION MATRIX (3x3):")
        for i, row in enumerate(rot):
            print(f"   [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
        
        # Gripper parameters
        print(f"\n3. GRIPPER PARAMETERS:")
        print(f"   Required width: {best_grasp['width']*100:.1f} cm")
        print(f"   Width ratio: {best_grasp['width_ratio']:.1%} of max opening")
        print(f"   Estimated force: {best_grasp['estimated_force']:.1f} N")
        print(f"   Grasp speed: {best_grasp['grasp_speed']:.2f}")
        
        # Approach vector (how gripper moves toward object)
        approach = best_grasp['approach_vector']
        print(f"\n4. APPROACH VECTOR:")
        print(f"   [{approach[0]:+.4f}, {approach[1]:+.4f}, {approach[2]:+.4f}]")
        print(f"   (Direction to move gripper toward object)")
        
        # Closing vector (direction fingers close)
        closing = best_grasp['closing_vector']
        print(f"\n5. CLOSING VECTOR:")
        print(f"   [{closing[0]:+.4f}, {closing[1]:+.4f}, {closing[2]:+.4f}]")
        print(f"   (Direction fingers close to grasp)")
        
        # Robot control parameters
        print(f"\n6. ROBOT CONTROL HINTS:")
        print(f"   Pre-grasp offset: {best_grasp['pre_grasp_offset']*100:.1f} cm")
        print(f"   (Move back along approach before closing)")
        print(f"   Confidence: {best_grasp['score']:.1%}")
        
        print("\n" + "-" * 40)
        print("💡 Example Robot Control Sequence:")
        print("-" * 40)
        
        # Calculate pre-grasp position
        pre_grasp = [
            pos[0] - approach[0] * best_grasp['pre_grasp_offset'],
            pos[1] - approach[1] * best_grasp['pre_grasp_offset'],
            pos[2] - approach[2] * best_grasp['pre_grasp_offset']
        ]
        
        print(f"\n1. Open gripper to {best_grasp['width']*100:.1f} cm")
        print(f"2. Move to pre-grasp: [{pre_grasp[0]:+.3f}, {pre_grasp[1]:+.3f}, {pre_grasp[2]:+.3f}]")
        print(f"3. Move to grasp: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
        print(f"4. Close gripper with {best_grasp['estimated_force']:.0f}N force")
        print(f"5. Lift object")
        print("-" * 40)


def example_5_error_handling():
    """Example 5: Robust error handling"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Error Handling")
    print("=" * 70)
    
    try:
        # Attempt detection
        best_grasp = simple_grasp_detector.detect_best_grasp(visualize=False)
        
        if best_grasp is None:
            print("\n⚠ No grasp detected. Possible reasons:")
            print("  1. No objects in workspace")
            print("  2. Workspace bounds too restrictive")
            print("  3. Lighting conditions poor")
            print("  4. Camera not connected")
            print("\n💡 Try adjusting workspace bounds or check camera connection")
        else:
            print(f"\n✓ Grasp found with score: {best_grasp['score']:.2%}")
            
            # Check grasp quality
            if best_grasp['score'] < 0.7:
                print("⚠ Warning: Low confidence grasp")
                print("  Consider capturing again or adjusting position")
            else:
                print("✓ High quality grasp!")
                
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("Check that camera is connected and dependencies are installed")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("MiniGrasp Examples")
    print("=" * 70)
    print("\nThese examples demonstrate how to use MiniGrasp for")
    print("simple grasp detection with RealSense depth cameras.")
    print("\n" + "=" * 70)
    
    # Ask user which example to run
    print("\nAvailable examples:")
    print("  1 - Basic usage (with visualization)")
    print("  2 - Custom configuration")
    print("  3 - Preset configurations")
    print("  4 - Extract full 6D pose")
    print("  5 - Error handling")
    print("  0 - Run all examples")
    
    choice = input("\nSelect example (0-5): ").strip()
    
    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_custom_config()
    elif choice == "3":
        example_3_preset_configs()
    elif choice == "4":
        example_4_extract_pose()
    elif choice == "5":
        example_5_error_handling()
    elif choice == "0":
        example_1_basic_usage()
        example_2_custom_config()
        example_3_preset_configs()
        example_4_extract_pose()
        example_5_error_handling()
    else:
        print("\nInvalid choice. Running basic example...")
        example_1_basic_usage()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
