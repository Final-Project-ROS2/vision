#!/usr/bin/env python3
"""
Camera Diagnostic - Check your camera setup before running grasp detection
This script shows you the point cloud statistics to help configure workspace bounds
"""

import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import sys

def test_camera():
    """Test camera and show point cloud statistics"""
    print("=" * 70)
    print("MiniGrasp Camera Diagnostic")
    print("=" * 70)
    
    try:
        # Initialize camera
        print("\n[1/4] Initializing RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print(f"  ✓ Camera initialized")
        print(f"  ✓ Depth scale: {depth_scale:.4f}")
        
        # Warm up
        print("\n[2/4] Warming up camera (capturing 30 frames)...")
        for i in range(30):
            pipeline.wait_for_frames()
            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/30")
        print("  ✓ Camera ready")
        
        # Capture frame
        print("\n[3/4] Capturing test frame...")
        align = rs.align(rs.stream.color)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        
        if not depth_frame:
            print("  ✗ Failed to capture depth frame")
            pipeline.stop()
            return
        
        print("  ✓ Frame captured")
        
        # Convert to numpy and show depth statistics
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_meters = depth_image.astype(np.float32) * depth_scale
        
        valid_depths = depth_meters[depth_meters > 0]
        
        print(f"\n  Depth Image Statistics:")
        print(f"    Size: {depth_image.shape}")
        print(f"    Valid pixels: {len(valid_depths)}/{depth_image.size} ({100*len(valid_depths)/depth_image.size:.1f}%)")
        print(f"    Depth range: {valid_depths.min():.3f} - {valid_depths.max():.3f} m")
        print(f"    Mean depth: {valid_depths.mean():.3f} m")
        
        # Create point cloud
        print("\n[4/4] Creating point cloud...")
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Filter depth
        MIN_DEPTH = 0.1
        MAX_DEPTH = 2.0
        depth_meters[(depth_meters < MIN_DEPTH) | (depth_meters > MAX_DEPTH)] = 0
        
        # Create Open3D point cloud
        o3d_depth = o3d.geometry.Image((depth_meters * 1000).astype(np.uint16))
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_intrinsics.width,
            depth_intrinsics.height,
            depth_intrinsics.fx,
            depth_intrinsics.fy,
            depth_intrinsics.ppx,
            depth_intrinsics.ppy
        )
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            o3d_intrinsics,
            depth_scale=1000.0,
            depth_trunc=MAX_DEPTH
        )
        
        points = np.asarray(pcd.points)
        
        print(f"  ✓ Point cloud created: {len(points)} points")
        
        if len(points) == 0:
            print("\n  ✗ ERROR: Point cloud is empty!")
            print("  Possible issues:")
            print("    - No object in front of camera")
            print("    - Object too close (< 10cm)")
            print("    - Object too far (> 2m)")
            print("    - Poor lighting")
            pipeline.stop()
            return
        
        # Show point cloud statistics
        print(f"\n  Point Cloud Statistics (camera coordinates):")
        print(f"    X range: [{points[:, 0].min():+.3f}, {points[:, 0].max():+.3f}] m")
        print(f"    Y range: [{points[:, 1].min():+.3f}, {points[:, 1].max():+.3f}] m")
        print(f"    Z range: [{points[:, 2].min():+.3f}, {points[:, 2].max():+.3f}] m")
        print(f"\n  Object center (approximate): [{points.mean(axis=0)[0]:+.3f}, {points.mean(axis=0)[1]:+.3f}, {points.mean(axis=0)[2]:+.3f}]")
        
        # Suggest workspace bounds
        print(f"\n  💡 RECOMMENDED WORKSPACE_BOUNDS for minigrasp/config.py:")
        print(f"  {{")
        margin = 0.1  # 10cm margin
        print(f"      'x_min': {points[:, 0].min() - margin:.2f},")
        print(f"      'x_max': {points[:, 0].max() + margin:.2f},")
        print(f"      'y_min': {points[:, 1].min() - margin:.2f},")
        print(f"      'y_max': {points[:, 1].max() + margin:.2f},")
        print(f"      'z_min': {max(0.1, points[:, 2].min() - margin):.2f},")
        print(f"      'z_max': {points[:, 2].max() + margin:.2f},")
        print(f"  }}")
        
        # Visualize
        print("\n  Opening 3D visualization...")
        print("  (Close the window when done to continue)")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name="Point Cloud - Check if object is visible",
            width=1280,
            height=720
        )
        
        pipeline.stop()
        
        print("\n" + "=" * 70)
        print("✓ DIAGNOSTIC COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. If you saw your object in the visualization, you're ready!")
        print("  2. Copy the recommended WORKSPACE_BOUNDS to minigrasp/config.py")
        print("  3. Run: python quick_test.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nCommon issues:")
        print("  - Camera not connected: Check USB connection")
        print("  - Permission denied: Add user to video group")
        print("  - No device found: Run 'realsense-viewer' to verify camera")

if __name__ == "__main__":
    test_camera()
