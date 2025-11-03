#!/usr/bin/env python3
"""
Quick Test Script for Vision Pipeline Services
Tests all services with the RGB image from Final-proj/src/arrange.jpg

Usage:
    python quick_service_test.py
"""

import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print("-" * 70)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"[SUCCESS] {description}")
        return True
    else:
        print(result.stderr)
        print(f"[FAILED] {description}")
        return False

def check_image_exists():
    """Check if test image exists"""
    image_path = Path("Final-proj/src/arrange.jpg")
    if image_path.exists():
        print(f"✓ Test image found: {image_path}")
        return str(image_path)
    else:
        print(f"✗ Test image not found: {image_path}")
        return None

def main():
    print("\n" + "="*70)
    print("SAM Vision Pipeline - Service Test")
    print("="*70)
    
    # Check test image
    image_path = check_image_exists()
    if not image_path:
        print("\nPlease ensure Final-proj/src/arrange.jpg exists")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Test Sequence")
    print("="*70)
    print("\n1. Start the vision pipeline in one terminal:")
    print("   ros2 run vision sam_vision_pipeline")
    print("\n2. Start the image publisher in another terminal:")
    print(f"   ros2 run vision test_pipeline_images --ros-args -p image_path:={image_path}")
    print("\n3. Wait 3 seconds for initialization")
    print("\n4. Run the following service tests:\n")
    
    # Service test commands
    tests = [
        ("ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger",
         "1. Reset Pipeline"),
        
        ("ros2 service call /vision/detect_objects std_srvs/srv/Trigger",
         "2. Detect Objects (SAM)"),
        
        ("ros2 service call /vision/classify_objects std_srvs/srv/Trigger",
         "3. Classify Objects (CLIP)"),
        
        ("ros2 service call /vision/get_positions std_srvs/srv/Trigger",
         "4. Get Object Positions"),
        
        ("ros2 service call /vision/generate_grasps std_srvs/srv/Trigger",
         "5. Generate Grasp Poses"),
        
        ("ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger",
         "6. Build Scene Graph"),
    ]
    
    # Print all test commands
    for cmd, desc in tests:
        print(f"\n{desc}:")
        print(f"  {cmd}")
    
    print("\n" + "="*70)
    print("\nOr use the automated test:")
    print("  ros2 run vision test_services")
    print("\n" + "="*70)
    
    # Ask if user wants to run tests now
    print("\nDo you want to run the automated tests now? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nRunning automated tests...")
        print("(Make sure the vision pipeline and image publisher are running!)")
        time.sleep(2)
        
        results = {}
        for cmd, desc in tests:
            success = run_command(cmd, desc)
            results[desc] = success
            time.sleep(1)
        
        # Print summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        passed = sum(1 for s in results.values() if s)
        total = len(results)
        
        for test, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {test}")
        
        print(f"\nResults: {passed}/{total} tests passed")
        print("="*70)

if __name__ == "__main__":
    main()
