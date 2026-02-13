#!/usr/bin/env python3
"""
Coordinate System Visualization Helper

This script helps visualize the relationship between:
- Camera coordinate frame
- Robot base coordinate frame
- Working area boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_coordinate_systems():
    """Create a 3D visualization of the coordinate systems."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera position in base frame
    cam_pos = np.array([0.0, -0.5442, 0.6711])
    
    # Base frame origin
    base_pos = np.array([0.0, 0.0, 0.0])
    
    # Floor
    floor_z = -0.805
    
    # Working area corners (on table at z=0)
    working_area_corners = np.array([
        [-0.5, -0.7, 0],  # Back-left
        [0.5, -0.7, 0],   # Back-right
        [0.5, 0.0, 0],    # Front-right
        [-0.5, 0.0, 0],   # Front-left
        [-0.5, -0.7, 0],  # Close the loop
    ])
    
    # Plot base frame origin
    ax.scatter(*base_pos, c='green', marker='o', s=200, label='Base/Table Origin')
    
    # Plot camera position
    ax.scatter(*cam_pos, c='blue', marker='^', s=200, label='Camera Position')
    
    # Draw base frame axes
    axis_length = 0.3
    ax.quiver(*base_pos, axis_length, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2, label='Base X (forward)')
    ax.quiver(*base_pos, 0, axis_length, 0, color='green', arrow_length_ratio=0.3, linewidth=2, label='Base Y (left)')
    ax.quiver(*base_pos, 0, 0, axis_length, color='blue', arrow_length_ratio=0.3, linewidth=2, label='Base Z (up)')
    
    # Draw camera frame axes (simplified - assuming looking down)
    ax.quiver(*cam_pos, axis_length*0.5, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.6)
    ax.quiver(*cam_pos, 0, 0, -axis_length*0.5, color='blue', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.6, label='Camera Z (down)')
    
    # Draw working area boundary
    ax.plot(working_area_corners[:, 0], 
            working_area_corners[:, 1], 
            working_area_corners[:, 2], 
            'r--', linewidth=2, label='Working Area')
    
    # Fill working area
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [working_area_corners[:-1]]
    poly = Poly3DCollection(verts, alpha=0.2, facecolor='yellow', edgecolor='red')
    ax.add_collection3d(poly)
    
    # Draw floor reference
    floor_corners = np.array([
        [-0.6, -0.8, floor_z],
        [0.6, -0.8, floor_z],
        [0.6, 0.1, floor_z],
        [-0.6, 0.1, floor_z],
        [-0.6, -0.8, floor_z],
    ])
    ax.plot(floor_corners[:, 0], 
            floor_corners[:, 1], 
            floor_corners[:, 2], 
            'k:', linewidth=1, alpha=0.5, label='Floor')
    
    # Draw line from camera to table center
    table_center = np.array([0.0, -0.35, 0.0])
    ax.plot([cam_pos[0], table_center[0]], 
            [cam_pos[1], table_center[1]], 
            [cam_pos[2], table_center[2]], 
            'b--', linewidth=1, alpha=0.5, label='Camera View')
    
    # Labels and formatting
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('Pixel-to-Real-World Coordinate System\nCamera at (0, -0.5442, 0.6711)', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.9, 0.2])
    ax.set_zlim([-0.9, 0.8])
    
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig, ax


def print_transformation_info():
    """Print detailed information about the coordinate transformation."""
    print("\n" + "="*70)
    print("COORDINATE SYSTEM INFORMATION")
    print("="*70)
    
    print("\n📷 Camera Position (in base frame):")
    print(f"   (0.0, -0.5442, 0.6711) meters")
    
    print("\n🎯 Base/Table Origin:")
    print(f"   (0.0, 0.0, 0.0) meters")
    
    print("\n🏢 Floor:")
    print(f"   z = -0.805 meters")
    
    print("\n📐 Working Area (x, y ranges):")
    print(f"   X: [-0.50, +0.50] meters (1.0m wide)")
    print(f"   Y: [-0.70,  0.00] meters (0.7m deep)")
    print(f"   Z:  ~0.00 meters (table plane)")
    
    print("\n🖼️  Image Resolution:")
    print(f"   640 x 480 pixels")
    print(f"   u ∈ [0, 640], v ∈ [0, 480]")
    
    print("\n🔄 Coordinate Frame Transformations:")
    print("\n   Camera Frame (RealSense):")
    print("   • X: right")
    print("   • Y: down")
    print("   • Z: forward (away from camera)")
    
    print("\n   Base Frame (Robot):")
    print("   • X: forward")
    print("   • Y: left")
    print("   • Z: up")
    
    print("\n   Transformation Equations:")
    print("   • x_base = cam_x_base + x_cam")
    print("   • y_base = cam_y_base - z_cam")
    print("   • z_base = cam_z_base - y_cam")
    
    print("\n📏 Expected Distances:")
    cam_to_table = 0.6711
    print(f"   Camera to table surface: ~{cam_to_table:.4f} meters")
    print(f"   Camera to floor: ~{0.6711 + 0.805:.4f} meters")
    
    print("\n" + "="*70)


def calculate_pixel_examples():
    """Calculate example transformations for common pixel locations."""
    print("\n" + "="*70)
    print("EXAMPLE PIXEL TRANSFORMATIONS")
    print("="*70)
    
    # Camera intrinsics (typical values for RealSense at 640x480)
    fx = 600.0  # approximate
    fy = 600.0
    ppx = 320.0
    ppy = 240.0
    
    cam_pos = np.array([0.0, -0.5442, 0.6711])
    depth = 0.67  # approximate depth to table
    
    pixels = [
        (320, 240, "Center"),
        (0, 0, "Top-left"),
        (640, 480, "Bottom-right"),
        (160, 240, "Left-center"),
        (480, 240, "Right-center"),
    ]
    
    print("\nAssuming:")
    print(f"  - Depth to table: {depth}m")
    print(f"  - Approximate intrinsics: fx={fx}, fy={fy}, ppx={ppx}, ppy={ppy}")
    print("\nPixel → Approximate Base Coordinates:")
    print("-" * 70)
    
    for u, v, desc in pixels:
        # Deproject (simplified)
        x_cam = (u - ppx) * depth / fx
        y_cam = (v - ppy) * depth / fy
        z_cam = depth
        
        # Transform to base
        x_base = cam_pos[0] + x_cam
        y_base = cam_pos[1] - z_cam
        z_base = cam_pos[2] - y_cam
        
        in_range = (-0.5 <= x_base <= 0.5) and (-0.7 <= y_base <= 0.0)
        status = "✓" if in_range else "✗"
        
        print(f"{status} ({u:3d}, {v:3d}) {desc:15s} → ({x_base:+.3f}, {y_base:+.3f}, {z_base:+.3f})")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Print transformation information
    print_transformation_info()
    
    # Calculate example transformations
    calculate_pixel_examples()
    
    # Create visualization
    print("\n📊 Generating 3D visualization...")
    fig, ax = visualize_coordinate_systems()
    
    # Save figure
    output_file = "/tmp/pixel_to_real_coordinate_system.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    
    # Show plot
    print("\n💡 Showing visualization (close window to exit)...")
    plt.show()
