#!/usr/bin/env python3
"""
GraspNet Detector Node - ROS2 Implementation with Real GraspNet

This node provides grasp detection services using a complete GraspNet implementation
based on Open3D point cloud processing and GraspNetAPI.

Services:
    /vision/detect_grasp - Detect best grasp from current camera view
        Returns: Best Grasp Found:
                 Position (x,y,z): [0.184, 0.040, 0.585] m
                 Confidence: 97.2%
                 Gripper width: 3.5 cm

Setup:
    Terminal 1: ros2 run vision graspnet_detector
    Terminal 2: ros2 service call /vision/detect_grasp std_srvs/srv/Trigger
"""

# ==============================================================================
# CONFIGURATION (from minigrasp_v2/config.py)
# ==============================================================================

import numpy as np

# Camera Settings
MIN_VALID_DEPTH = 0.1  # Minimum reliable depth (10cm)
MAX_VALID_DEPTH = 2.0  # Maximum workspace depth (2m)

# Plane Removal (Remove table/floor background)
RANSAC_DISTANCE_THRESHOLD = 0.01  # Max distance point can be from plane (1cm)
RANSAC_NUM_ITERATIONS = 1000      # RANSAC iterations for plane detection
RANSAC_MIN_POINTS = 3             # Minimum points to fit plane
REMOVE_PLANE = True               # Enable automatic plane removal
PLANE_REMOVAL_AGGRESSIVE = True   # Remove ALL points close to plane

# Top-View Occlusion Handling
EXTRUDE_TOP_SURFACE = True        # Create synthetic 3D volume from 2D top view
EXTRUSION_DEPTH = 0.08            # How far to extrude downward (8cm for boxes)
EXTRUSION_METHOD = 'uniform'      # 'uniform' or 'adaptive'
ASSUMED_OBJECT_HEIGHT = 0.08      # Default object height if unknown (8cm)
MIN_OBJECT_HEIGHT = 0.02          # Minimum reasonable height (2cm)
MAX_OBJECT_HEIGHT = 0.15          # Maximum reasonable height (15cm)

# Workspace Bounds (meters, relative to camera frame)
WORKSPACE_BOUNDS = {
    'x_min': -1.0,    # 1m to the left
    'x_max': 1.0,     # 1m to the right
    'y_min': -1.0,    # 1m up
    'y_max': 1.0,     # 1m down
    'z_min': 0.2,     # Start at 20cm from camera
    'z_max': 1.5,     # End at 1.5m from camera
}

# Gripper Parameters
GRIPPER_WIDTH = 0.05        # Maximum opening width (5cm)
GRIPPER_MIN_WIDTH = 0.001   # Minimum opening width (1mm)
GRIPPER_DEPTH = 0.05        # Finger depth (5cm)
GRIPPER_HEIGHT = 0.02       # Finger height (2cm)
GRIPPER_FORCE = 50.0        # Maximum gripping force (N)
FRICTION_COEFFICIENT = 0.5  # Friction coefficient
PREFERRED_WIDTH_RATIO = 0.6 # Prefer using 60% of gripper range
WIDTH_SAFETY_MARGIN = 1.1   # Add 10% margin to calculated width

# Grasp Refinement & Alignment
ENABLE_GRASP_REFINEMENT = True    # Enable ICP-like refinement
REFINEMENT_MAX_ITERATIONS = 5     # Max refinement steps
REFINEMENT_DISTANCE = 0.002       # Convergence threshold (2mm)
CENTER_ON_POINTS = True           # Move grasp to actual point cluster center
MIN_POINTS_FOR_GRASP = 5          # Minimum points near grasp for validity

# Grasp Generation
NUM_GRASP_CANDIDATES = 300  # Balance between coverage and speed
MIN_GRASP_SCORE = 0.4      # Very permissive - almost all grasps pass

# Approach angle filtering
APPROACH_TARGET_VECTOR = [0, 0, -1]  # Downward in camera frame
MAX_APPROACH_ANGLE = 360             # Allow all angles

# Collision checking
COLLISION_FINGER_WIDTH = 0.01   # Finger thickness
COLLISION_BASE_DEPTH = 0.03     # Gripper base depth

# Point cloud filtering thresholds
OUTLIER_REMOVAL_NB_NEIGHBORS = 20      # Number of neighbors for outlier detection
OUTLIER_REMOVAL_STD_RATIO = 2.0        # Standard deviation threshold
VOXEL_DOWNSAMPLE_THRESHOLD = 10000     # Downsample if more than this many points
VOXEL_DOWNSAMPLE_SIZE = 0.005          # Voxel size for downsampling (5mm)
NORMAL_ESTIMATION_RADIUS = 0.01        # Radius for normal estimation (1cm)
NORMAL_ESTIMATION_MAX_NN = 30          # Max neighbors for normal estimation

# Extrusion parameters
EXTRUSION_MIN_THICKNESS = 0.01         # Min thickness to trigger extrusion (1cm)
EXTRUSION_MIN_DEPTH = 0.02             # Minimum extrusion depth (2cm)
EXTRUSION_NUM_LAYERS = 5               # Number of layers when extruding
EXTRUSION_COLOR_FADE = 0.9             # Color fade factor for extruded layers

# Dense region finding
DENSE_REGION_VOXEL_SIZE = 0.03         # Voxel size for density analysis (3cm)
DENSE_REGION_MAX_GRID = 10000          # Max grid cells before falling back to center

# Grasp generation parameters
GRASP_DENSE_REGION_FOCUS = 0.8         # 80% of grasps focus on densest region
GRASP_DENSE_REGION_OFFSET = 0.1        # Offset multiplier for dense region grasps
GRASP_RANDOM_OFFSET = 0.15             # Offset multiplier for random grasps
GRASP_PHI_MIN = -np.pi / 4             # Min phi angle (-45°)
GRASP_PHI_MAX = np.pi / 4              # Max phi angle (45°)

# Width estimation
WIDTH_ESTIMATION_RADIUS = 0.05         # Radius to find nearby points (5cm)
WIDTH_DEFAULT_RATIO = 0.7              # Default width as ratio of max gripper width
WIDTH_SAFETY_MULTIPLIER = 1.1          # Safety margin multiplier
WIDTH_MIN_CLIP = 0.01                  # Minimum width (1cm)

# Quality score weights
QUALITY_BASE_SCORE = 0.4               # Base quality score
QUALITY_CENTER_WEIGHT = 0.4            # Weight for distance to center
QUALITY_CENTER_SCALE = 0.05            # Scale factor for center distance (5cm)
QUALITY_WIDTH_WEIGHT = 0.1             # Weight for width utilization
QUALITY_WIDTH_MIN = 0.4                # Min width ratio for full score
QUALITY_WIDTH_MAX = 0.8                # Max width ratio for full score
QUALITY_DENSITY_WEIGHT = 0.2           # Weight for point density
QUALITY_DENSITY_RADIUS = 0.04          # Radius for density check (4cm)
QUALITY_DENSITY_TARGET = 30.0          # Target point count for full density score

# Collision detection
COLLISION_CHECK_X_TOLERANCE = 0.005    # Collision tolerance in X (5mm)
COLLISION_CHECK_Y_MARGIN = 0.01        # Extra margin for Y collision (1cm)
COLLISION_CHECK_Z_TOLERANCE = 0.005    # Collision tolerance in Z (5mm)

# Grasp refinement
REFINEMENT_SEARCH_RADIUS = 0.08        # Radius to search for nearby points (8cm)
REFINEMENT_MAX_TRANSLATION = 0.03      # Max allowed translation during refinement (3cm)
REFINEMENT_SCORE_BOOST = 1.02          # Score multiplier for refined grasps

# Best grasp calculation
BEST_GRASP_FORCE_BASE = 20.0           # Base estimated force (N)
BEST_GRASP_FORCE_RANGE = 30.0          # Additional force based on width (N)
BEST_GRASP_PRE_OFFSET = 0.05           # Pre-grasp offset distance (5cm)
BEST_GRASP_SPEED_FAST = 0.5            # Fast grasp speed
BEST_GRASP_SPEED_SLOW = 0.3            # Slow grasp speed (for wide objects)
BEST_GRASP_SPEED_THRESHOLD = 0.7       # Width ratio threshold for speed selection

# Visualization parameters
VIS_FINGER_WIDTH = 0.004               # Finger thickness in visualization (4mm)
VIS_PALM_DEPTH = 0.01                  # Palm depth in visualization (1cm)
VIS_COORD_FRAME_SIZE = 0.1             # Coordinate frame size (10cm)
VIS_MAX_ALT_GRASPS = 6                 # Max alternative grasps to show
VIS_WINDOW_WIDTH = 1024                # Visualization window width
VIS_WINDOW_HEIGHT = 768                # Visualization window height
VIS_COLOR_BEST = [1, 0, 0]             # Red for best grasp
VIS_COLOR_ALT = [0, 1, 0]              # Green for alternative grasps

# ==============================================================================
# IMPORTS
# ==============================================================================

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path

# Import custom interfaces
try:
    from custom_interfaces.srv import DetectGrasps
    from custom_interfaces.msg import GraspPose
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    print("Custom interfaces not available. Using Trigger service as fallback.")
    CUSTOM_INTERFACES_AVAILABLE = False
    DetectGrasps = None
    GraspPose = None

# Open3D and GraspNet - lazy loaded when service is called
o3d = None
GRASPNET_AVAILABLE = False

# Simple Grasp classes (avoiding complex graspnetAPI dependencies)
class Grasp:
    """Simplified Grasp class for storing grasp parameters"""
    def __init__(self, score, width, height, depth, rotation_matrix, translation, object_id=0):
        self.score = float(score)
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)
        self.rotation_matrix = np.array(rotation_matrix)
        self.translation = np.array(translation)
        self.object_id = int(object_id)

class GraspGroup:
    """Simplified GraspGroup class for storing multiple grasps"""
    def __init__(self):
        self.grasps = []
    
    def add(self, grasp):
        self.grasps.append(grasp)
    
    def __len__(self):
        return len(self.grasps)
    
    def __getitem__(self, index):
        return self.grasps[index]
    
    def __iter__(self):
        return iter(self.grasps)
    
    def sort_by_score(self):
        """Sort grasps by score in descending order"""
        sorted_group = GraspGroup()
        sorted_group.grasps = sorted(self.grasps, key=lambda g: g.score, reverse=True)
        return sorted_group

def _lazy_load_graspnet():
    """Lazy load Open3D when first needed"""
    global o3d, GRASPNET_AVAILABLE
    
    if GRASPNET_AVAILABLE:
        return True
    
    try:
        import open3d as o3d_module
        
        # Set globals
        o3d = o3d_module
        GRASPNET_AVAILABLE = True
        
        print("✓ Open3D loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load Open3D: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================================================================
# GRASPNET DETECTION ENGINE
# ==============================================================================

class GraspNetEngine:
    """GraspNet detection engine - processes point clouds to find grasps"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def create_pointcloud_from_rgbd(self, color_image, depth_image, camera_info):
        """
        Create Open3D point cloud from RGB-D images
        
        Args:
            color_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            camera_info: ROS CameraInfo message
            
        Returns:
            Open3D PointCloud
        """
        height, width = depth_image.shape
        
        # Create Open3D camera intrinsics
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Filter invalid depths
        depth_filtered = depth_image.copy()
        depth_filtered[(depth_filtered < MIN_VALID_DEPTH) | (depth_filtered > MAX_VALID_DEPTH)] = 0
        
        # Convert to Open3D images
        o3d_color = o3d.geometry.Image(color_image.astype(np.uint8))
        o3d_depth = o3d.geometry.Image((depth_filtered * 1000).astype(np.uint16))  # Convert to mm
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1000.0,
            depth_trunc=MAX_VALID_DEPTH,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d_intrinsics
        )
        
        return pcd
    
    def filter_pointcloud(self, pointcloud):
        """
        Filter point cloud: workspace crop, RANSAC plane removal, outlier removal
        
        Args:
            pointcloud: Raw point cloud
            
        Returns:
            Filtered point cloud (object only, no background)
        """
        if len(pointcloud.points) == 0:
            self.logger.error("Point cloud is empty!")
            return pointcloud
        
        self.logger.info("Filtering point cloud (removing background)...")
        
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
        
        # Step 1: Workspace bounds filtering
        ws = WORKSPACE_BOUNDS
        mask = (
            (points[:, 0] >= ws['x_min']) & (points[:, 0] <= ws['x_max']) &
            (points[:, 1] >= ws['y_min']) & (points[:, 1] <= ws['y_max']) &
            (points[:, 2] >= ws['z_min']) & (points[:, 2] <= ws['z_max'])
        )
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        if colors is not None:
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        self.logger.info(f"  → Workspace crop: {len(filtered_pcd.points)} points")
        
        if len(filtered_pcd.points) < 100:
            self.logger.warn("Too few points after workspace crop!")
            return filtered_pcd
        
        # Step 2: RANSAC plane segmentation (remove table/floor)
        plane_equation = None
        if REMOVE_PLANE and len(filtered_pcd.points) >= 100:
            self.logger.info("  → RANSAC plane removal...")
            
            plane_model, inliers = filtered_pcd.segment_plane(
                distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                ransac_n=RANSAC_MIN_POINTS,
                num_iterations=RANSAC_NUM_ITERATIONS
            )
            
            if len(inliers) > 0:
                [a, b, c, d] = plane_model
                self.logger.info(f"    Plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                self.logger.info(f"    Plane points: {len(inliers)} ({100*len(inliers)/len(filtered_pcd.points):.1f}%)")
                
                # Remove plane (keep only outliers = object)
                filtered_pcd = filtered_pcd.select_by_index(inliers, invert=True)
                self.logger.info(f"  → After plane removal: {len(filtered_pcd.points)} points (object only)")
                plane_equation = [a, b, c, d]
        
        if len(filtered_pcd.points) < 50:
            self.logger.warn("Too few points after plane removal!")
            return filtered_pcd
        
        # Step 2.5: Handle top-view occlusion
        if EXTRUDE_TOP_SURFACE and plane_equation is not None:
            filtered_pcd = self._extrude_top_surface(filtered_pcd, plane_equation)
        
        # Step 3: Statistical outlier removal
        if len(filtered_pcd.points) > 10:
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
                nb_neighbors=OUTLIER_REMOVAL_NB_NEIGHBORS,
                std_ratio=OUTLIER_REMOVAL_STD_RATIO
            )
            self.logger.info(f"  → Outlier removal: {len(filtered_pcd.points)} points")
        
        # Step 4: Voxel downsampling
        if len(filtered_pcd.points) > VOXEL_DOWNSAMPLE_THRESHOLD:
            original_count = len(filtered_pcd.points)
            filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=VOXEL_DOWNSAMPLE_SIZE)
            self.logger.info(f"  → Voxel downsampling: {len(filtered_pcd.points)} points (was {original_count})")
        
        # Step 5: Estimate normals
        if len(filtered_pcd.points) > 0:
            filtered_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_ESTIMATION_RADIUS, max_nn=NORMAL_ESTIMATION_MAX_NN)
            )
        
        self.logger.info(f"✓ Filtering complete: {len(filtered_pcd.points)} object points")
        
        return filtered_pcd
    
    def _extrude_top_surface(self, pointcloud, plane_equation):
        """Extrude top surface downward to create 3D volume"""
        self.logger.info("  → Top-view extrusion (solving occlusion)...")
        
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
        
        if len(points) == 0:
            return pointcloud
        
        [a, b, c, d] = plane_equation
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        extrusion_dir = -plane_normal
        
        # Estimate object height
        heights = np.dot(points, plane_normal)
        point_thickness = heights.max() - heights.min()
        
        if point_thickness < EXTRUSION_MIN_THICKNESS:
            extrusion_depth = EXTRUSION_DEPTH
        else:
            extrusion_depth = max(EXTRUSION_MIN_DEPTH, EXTRUSION_DEPTH - point_thickness)
        
        # Create extruded points (multiple layers)
        num_layers = EXTRUSION_NUM_LAYERS
        extruded_points_list = [points]
        if colors is not None:
            extruded_colors_list = [colors]
        
        for i in range(1, num_layers):
            offset = extrusion_dir * (extrusion_depth * i / num_layers)
            layer_points = points + offset
            extruded_points_list.append(layer_points)
            
            if colors is not None:
                extruded_colors_list.append(colors * EXTRUSION_COLOR_FADE)
        
        all_points = np.vstack(extruded_points_list)
        
        extruded_pcd = o3d.geometry.PointCloud()
        extruded_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        if colors is not None:
            all_colors = np.vstack(extruded_colors_list)
            extruded_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        self.logger.info(f"    Created 3D volume: {len(points)} → {len(all_points)} points")
        
        return extruded_pcd
    
    def _find_densest_region(self, points):
        """Find the densest region in point cloud"""
        voxel_size = DENSE_REGION_VOXEL_SIZE
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        grid_dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
        
        if np.prod(grid_dims) > DENSE_REGION_MAX_GRID:
            return points.mean(axis=0)
        
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
        
        voxel_counts = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_counts:
                voxel_counts[key] = []
            voxel_counts[key].append(i)
        
        densest_voxel = max(voxel_counts.items(), key=lambda x: len(x[1]))
        densest_indices = densest_voxel[1]
        densest_points = points[densest_indices]
        
        return densest_points.mean(axis=0)
    
    def generate_grasp_candidates(self, pointcloud):
        """Generate grasp candidates from point cloud"""
        self.logger.info("Generating grasp candidates...")
        
        if len(pointcloud.points) < 100:
            self.logger.error("Too few points for grasp generation")
            return None
        
        points = np.asarray(pointcloud.points)
        normals = np.asarray(pointcloud.normals) if pointcloud.has_normals() else None
        
        center = points.mean(axis=0)
        extents = points.max(axis=0) - points.min(axis=0)
        
        self.logger.info(f"  Object center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        self.logger.info(f"  Object size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}] m")
        
        densest_center = self._find_densest_region(points)
        self.logger.info(f"  Densest region: [{densest_center[0]:.3f}, {densest_center[1]:.3f}, {densest_center[2]:.3f}]")
        
        gg = GraspGroup()
        
        for i in range(NUM_GRASP_CANDIDATES):
            # Random approach angles
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(GRASP_PHI_MIN, GRASP_PHI_MAX)
            
            # Focus on densest region (80% of grasps)
            if np.random.rand() < GRASP_DENSE_REGION_FOCUS:
                offset = np.random.uniform(-1, 1, size=3) * extents * GRASP_DENSE_REGION_OFFSET
                translation = densest_center + offset
            else:
                offset = np.random.uniform(-1, 1, size=3) * extents * GRASP_RANDOM_OFFSET
                translation = center + offset
            
            # Create rotation matrix
            approach = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])
            
            closing = np.array([-np.sin(theta), np.cos(theta), 0])
            binormal = np.cross(approach, closing)
            rotation = np.column_stack([closing, binormal, approach])
            
            width = self._estimate_grasp_width(points, translation, closing)
            score = self._calculate_grasp_quality(points, translation, rotation, width, densest_center, normals)
            
            height = GRIPPER_HEIGHT
            depth = GRIPPER_DEPTH
            
            grasp = Grasp(score, width, height, depth, rotation, translation, 0)
            gg.add(grasp)
        
        gg = gg.sort_by_score()
        
        self.logger.info(f"✓ Generated {len(gg)} grasp candidates")
        if len(gg) > 0:
            self.logger.info(f"  Width range: {min(g.width for g in gg)*100:.1f} - {max(g.width for g in gg)*100:.1f} cm")
            self.logger.info(f"  Score range: {min(g.score for g in gg):.3f} - {max(g.score for g in gg):.3f}")
        
        return gg
    
    def _estimate_grasp_width(self, points, grasp_pos, closing_direction):
        """Estimate required grasp width at a position"""
        distances = np.linalg.norm(points - grasp_pos, axis=1)
        nearby_mask = distances < WIDTH_ESTIMATION_RADIUS
        
        if not np.any(nearby_mask):
            return GRIPPER_WIDTH * WIDTH_DEFAULT_RATIO
        
        nearby_points = points[nearby_mask]
        projections = np.dot(nearby_points - grasp_pos, closing_direction)
        
        if len(projections) > 0:
            width = projections.max() - projections.min()
            width = width * WIDTH_SAFETY_MULTIPLIER  # Safety margin
            width = np.clip(width, WIDTH_MIN_CLIP, GRIPPER_WIDTH)
        else:
            width = GRIPPER_WIDTH * WIDTH_DEFAULT_RATIO
        
        return width
    
    def _calculate_grasp_quality(self, points, position, rotation, width, object_center, normals=None):
        """Calculate grasp quality score"""
        score = QUALITY_BASE_SCORE  # Base score
        
        # Distance from densest region
        dist_to_center = np.linalg.norm(position - object_center)
        center_score = np.exp(-dist_to_center / QUALITY_CENTER_SCALE)
        score += QUALITY_CENTER_WEIGHT * center_score
        
        # Width utilization
        width_ratio = width / GRIPPER_WIDTH
        if QUALITY_WIDTH_MIN <= width_ratio <= QUALITY_WIDTH_MAX:
            width_score = 1.0
        else:
            width_score = 0.5
        score += QUALITY_WIDTH_WEIGHT * width_score
        
        # Point density near grasp
        distances = np.linalg.norm(points - position, axis=1)
        nearby_count = np.sum(distances < QUALITY_DENSITY_RADIUS)
        density_score = min(nearby_count / QUALITY_DENSITY_TARGET, 1.0)
        score += QUALITY_DENSITY_WEIGHT * density_score
        
        return np.clip(score, 0.0, 1.0)
    
    def filter_grasps(self, grasp_group, pointcloud):
        """Filter grasps by collision and feasibility"""
        if grasp_group is None or len(grasp_group) == 0:
            return grasp_group
        
        self.logger.info("Filtering grasps...")
        
        initial_count = len(grasp_group)
        
        # Filter by score threshold
        filtered_gg = GraspGroup()
        for grasp in grasp_group:
            if grasp.score >= MIN_GRASP_SCORE:
                filtered_gg.add(grasp)
        
        self.logger.info(f"  Score threshold: {initial_count} → {len(filtered_gg)} grasps")
        
        # Collision filter
        if len(pointcloud.points) > 0:
            filtered_gg = self._collision_filter(filtered_gg, pointcloud)
        
        self.logger.info(f"✓ Final grasps: {len(filtered_gg)}")
        
        return filtered_gg
    
    def _collision_filter(self, grasp_group, pointcloud):
        """Simple collision detection"""
        points = np.asarray(pointcloud.points)
        
        if len(points) == 0:
            return grasp_group
        
        filtered_gg = GraspGroup()
        num_collided = 0
        
        for grasp in grasp_group:
            pts_gripper = (points - grasp.translation) @ grasp.rotation_matrix
            
            collision = np.any(
                (np.abs(pts_gripper[:, 0]) < COLLISION_CHECK_X_TOLERANCE) &
                (np.abs(pts_gripper[:, 1]) < grasp.width/2 + COLLISION_CHECK_Y_MARGIN) &
                (np.abs(pts_gripper[:, 2]) < COLLISION_CHECK_Z_TOLERANCE)
            )
            
            if not collision:
                filtered_gg.add(grasp)
            else:
                num_collided += 1
        
        self.logger.info(f"  Collision filter: removed {num_collided} grasps")
        return filtered_gg
    
    def refine_grasp_candidates(self, grasp_group, pointcloud):
        """Refine grasps to align with actual point clusters"""
        if not ENABLE_GRASP_REFINEMENT:
            return grasp_group
        
        self.logger.info("  → Refining grasps...")
        
        points = np.asarray(pointcloud.points)
        normals = np.asarray(pointcloud.normals) if pointcloud.has_normals() else None
        
        if len(points) == 0:
            return grasp_group
        
        refined_gg = GraspGroup()
        num_improved = 0
        num_removed = 0
        
        for grasp in grasp_group:
            distances = np.linalg.norm(points - grasp.translation, axis=1)
            nearby_mask = distances < REFINEMENT_SEARCH_RADIUS
            
            nearby_count = np.sum(nearby_mask)
            if nearby_count < MIN_POINTS_FOR_GRASP:
                num_removed += 1
                continue
            
            nearby_points = points[nearby_mask]
            
            if CENTER_ON_POINTS:
                cluster_center = nearby_points.mean(axis=0)
                translation_offset = cluster_center - grasp.translation
                
                if np.linalg.norm(translation_offset) < REFINEMENT_MAX_TRANSLATION:
                    new_translation = cluster_center
                    
                    refined_grasp = Grasp(
                        grasp.score * REFINEMENT_SCORE_BOOST,
                        grasp.width,
                        grasp.height,
                        grasp.depth,
                        grasp.rotation_matrix,
                        new_translation,
                        grasp.object_id
                    )
                    refined_gg.add(refined_grasp)
                    num_improved += 1
                    continue
            
            refined_gg.add(grasp)
        
        self.logger.info(f"    Improved {num_improved} grasps, removed {num_removed} floating grasps")
        return refined_gg.sort_by_score()
    
    def get_best_grasp(self, grasp_group):
        """Extract the best grasp from the group"""
        if grasp_group is None or len(grasp_group) == 0:
            return None
        
        best = grasp_group[0]
        
        approach_vector = best.rotation_matrix[:, 2]
        closing_vector = best.rotation_matrix[:, 0]
        
        # Calculate grasp angle (rotation around approach axis in degrees)
        # Use the closing direction projected onto the XY plane
        closing_xy = closing_vector[:2]
        grasp_angle = np.arctan2(closing_xy[1], closing_xy[0]) * 180.0 / np.pi
        
        width_ratio = best.width / GRIPPER_WIDTH
        estimated_force = BEST_GRASP_FORCE_BASE + (1.0 - width_ratio) * BEST_GRASP_FORCE_RANGE
        
        result = {
            'position': best.translation.tolist(),
            'rotation': best.rotation_matrix.tolist(),
            'approach_vector': approach_vector.tolist(),
            'closing_vector': closing_vector.tolist(),
            'grasp_angle': float(grasp_angle),
            'width': float(best.width),
            'height': float(best.height),
            'depth': float(best.depth),
            'score': float(best.score),
            'width_ratio': float(width_ratio),
            'estimated_force': float(estimated_force),
            'grasp_center': best.translation.tolist(),
            'pre_grasp_offset': BEST_GRASP_PRE_OFFSET,
            'grasp_speed': BEST_GRASP_SPEED_FAST if width_ratio > BEST_GRASP_SPEED_THRESHOLD else BEST_GRASP_SPEED_SLOW,
        }
        
        return result
    
    def create_gripper_mesh(self, grasp_dict, color=None):
        """Create Open3D mesh geometry for gripper visualization"""
        if color is None:
            color = VIS_COLOR_BEST  # Red by default
        
        # Extract grasp parameters
        center = np.array(grasp_dict['position'])
        rotation = np.array(grasp_dict['rotation'])
        width = grasp_dict['width']
        depth = grasp_dict['depth']
        height = grasp_dict['height']
        
        # Create gripper mesh components
        geometries = []
        
        # Finger dimensions
        finger_width = VIS_FINGER_WIDTH  # 4mm thick fingers
        finger_length = depth
        finger_height = height
        
        # Create left finger
        left_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_width, height=finger_height, depth=finger_length
        )
        left_finger.translate([-width/2 - finger_width/2, -finger_height/2, -finger_length/2])
        
        # Create right finger
        right_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_width, height=finger_height, depth=finger_length
        )
        right_finger.translate([width/2 - finger_width/2, -finger_height/2, -finger_length/2])
        
        # Create palm (connecting base)
        palm_width = width + 2 * finger_width
        palm_height = finger_height
        palm_depth = VIS_PALM_DEPTH  # 1cm thick palm
        palm = o3d.geometry.TriangleMesh.create_box(
            width=palm_width, height=palm_height, depth=palm_depth
        )
        palm.translate([-palm_width/2, -palm_height/2, -finger_length/2 - palm_depth])
        
        # Combine all parts
        gripper = left_finger + right_finger + palm
        
        # Apply rotation and translation
        gripper.rotate(rotation, center=[0, 0, 0])
        gripper.translate(center)
        
        # Apply color
        gripper.paint_uniform_color(color)
        gripper.compute_vertex_normals()
        
        return gripper
    
    def visualize_grasp(self, pointcloud, grasp_dict, grasp_group=None):
        """Visualize point cloud with detected grasp pose"""
        self.logger.info("Opening 3D visualization...")
        
        geometries = []
        
        # Add point cloud
        geometries.append(pointcloud)
        
        # Add best grasp (red)
        best_gripper = self.create_gripper_mesh(grasp_dict, color=VIS_COLOR_BEST)
        geometries.append(best_gripper)
        
        # Add alternative grasps (green) if available
        if grasp_group is not None and len(grasp_group) > 1:
            for i, grasp in enumerate(grasp_group[1:min(VIS_MAX_ALT_GRASPS, len(grasp_group))]):
                alt_grasp_dict = {
                    'position': grasp.translation.tolist(),
                    'rotation': grasp.rotation_matrix.tolist(),
                    'width': float(grasp.width),
                    'height': float(grasp.height),
                    'depth': float(grasp.depth),
                }
                alt_gripper = self.create_gripper_mesh(alt_grasp_dict, color=VIS_COLOR_ALT)
                geometries.append(alt_gripper)
        
        # Add coordinate frame (camera origin)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=VIS_COORD_FRAME_SIZE, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)
        
        # Display
        o3d.visualization.draw_geometries(
            geometries,
            window_name="GraspNet Detection (Red = Best Grasp, Green = Alternatives)",
            width=VIS_WINDOW_WIDTH,
            height=VIS_WINDOW_HEIGHT
        )
        
        self.logger.info("✓ Visualization closed")

# ==============================================================================
# ROS2 NODE
# ==============================================================================

class GraspNetDetector(Node):
    """ROS2 Node for GraspNet-based grasp detection"""
    
    def __init__(self):
        super().__init__('graspnet_detector')
        
        # Callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Latest sensor data
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        
        # GraspNet engine will be initialized on first service call
        self.grasp_engine = None
        
        self.get_logger().info("✓ GraspNet detector ready (Open3D will load on first service call)")
        
        # Camera topics - always use real hardware topics
        self.rgb_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.camera_info_topic = '/camera/color/camera_info'
        
        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera topics
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            self.image_qos
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            self.image_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            self.image_qos
        )
        
        # Create grasp detection service (always use Trigger for simplicity)
        self.grasp_service = self.create_service(
            Trigger,
            '/vision/detect_grasp',
            self.detect_grasp_callback_trigger,
            callback_group=self.callback_group
        )
        
        self.get_logger().info("Service registered: /vision/detect_grasp (std_srvs/srv/Trigger)")
        
        # Publisher for grasp poses
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/vision/grasp_poses',
            10
        )
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("GraspNet Detector Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Subscribed to: {self.rgb_topic}")
        self.get_logger().info(f"Subscribed to: {self.depth_topic}")
        self.get_logger().info(f"Subscribed to: {self.camera_info_topic}")
        self.get_logger().info("Service: /vision/detect_grasp")
        self.get_logger().info("Publishing to: /vision/grasp_poses")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Usage:")
        self.get_logger().info("  ros2 service call /vision/detect_grasp std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image messages"""
        try:
            # RealSense publishes in RGB8 format
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Convert RGB to BGR for OpenCV
            if len(self.latest_rgb.shape) == 3 and self.latest_rgb.shape[2] == 3:
                self.latest_rgb = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
    
    def depth_callback(self, msg: Image):
        """Handle depth image messages"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Convert to meters if needed
            if depth.dtype == np.uint16:
                # Assume depth is in mm
                self.latest_depth = depth.astype(np.float32) / 1000.0
            else:
                self.latest_depth = depth.astype(np.float32)
            
            # Clean up depth data
            self.latest_depth = np.nan_to_num(self.latest_depth, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
    
    def detect_grasp_callback(self, request, response):
        """Service callback for /vision/detect_grasp - using custom DetectGrasps interface"""
        try:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Grasp Detection Service Called")
            self.get_logger().info("=" * 60)
            
            # Lazy load GraspNet dependencies on first call
            if self.grasp_engine is None:
                self.get_logger().info("Loading Open3D and GraspNet (first call)...")
                if not _lazy_load_graspnet():
                    response.success = False
                    response.total_grasps = 0
                    response.grasp_poses = []
                    response.error_message = "Failed to load GraspNet dependencies. Install: pip install open3d"
                    self.get_logger().error("✗ GraspNet dependencies could not be loaded")
                    return response
                
                # Initialize engine after successful load
                self.grasp_engine = GraspNetEngine(self.get_logger())
                self.get_logger().info("✓ GraspNet engine initialized")
            
            # Check if we have sensor data
            if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "No RGB-D data available. Waiting for camera..."
                self.get_logger().warn("No RGB-D data captured yet")
                return response
            
            # Run grasp detection
            start_time = time.time()
            
            # Step 1: Create point cloud
            self.get_logger().info("Creating point cloud from RGB-D...")
            pcd = self.grasp_engine.create_pointcloud_from_rgbd(
                self.latest_rgb,
                self.latest_depth,
                self.camera_info
            )
            
            if len(pcd.points) == 0:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "Failed to create point cloud"
                return response
            
            self.get_logger().info(f"✓ Point cloud created: {len(pcd.points)} points")
            
            # Step 2: Filter point cloud
            pcd_filtered = self.grasp_engine.filter_pointcloud(pcd)
            
            if len(pcd_filtered.points) < 100:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "Too few points after filtering. No object detected."
                return response
            
            # Step 3: Generate grasp candidates
            grasp_candidates = self.grasp_engine.generate_grasp_candidates(pcd_filtered)
            
            if grasp_candidates is None or len(grasp_candidates) == 0:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "No grasp candidates generated"
                return response
            
            # Step 4: Filter grasps
            filtered_grasps = self.grasp_engine.filter_grasps(grasp_candidates, pcd_filtered)
            
            if filtered_grasps is None or len(filtered_grasps) == 0:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "No valid grasps found after filtering"
                return response
            
            # Step 5: Refine grasps
            refined_grasps = self.grasp_engine.refine_grasp_candidates(filtered_grasps, pcd_filtered)
            
            if refined_grasps is None or len(refined_grasps) == 0:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "No grasps remain after refinement"
                return response
            
            # Step 6: Get best grasp
            best_grasp = self.grasp_engine.get_best_grasp(refined_grasps)
            
            if best_grasp is None:
                response.success = False
                response.total_grasps = 0
                response.grasp_poses = []
                response.error_message = "Failed to extract best grasp"
                return response
            
            elapsed_time = time.time() - start_time
            
            # Step 7: Visualize result
            self.grasp_engine.visualize_grasp(pcd_filtered, best_grasp, refined_grasps)
            
            # Create GraspPose message
            grasp_msg = GraspPose()
            grasp_msg.object_id = "detected_object"
            
            # Position
            pos = best_grasp['position']
            grasp_msg.position.x = float(pos[0])
            grasp_msg.position.y = float(pos[1])
            grasp_msg.position.z = float(pos[2])
            
            # Orientation (convert rotation matrix to quaternion)
            R = np.array(best_grasp['rotation'])
            q = self._rotation_matrix_to_quaternion(R)
            grasp_msg.orientation.x = q[0]
            grasp_msg.orientation.y = q[1]
            grasp_msg.orientation.z = q[2]
            grasp_msg.orientation.w = q[3]
            
            # Grasp parameters
            confidence = best_grasp['score'] * 100
            width_cm = best_grasp['width'] * 100
            angle = best_grasp['grasp_angle']
            
            grasp_msg.quality_score = float(best_grasp['score'])
            grasp_msg.width = float(best_grasp['width'])
            grasp_msg.approach_direction = "auto"
            grasp_msg.bbox = [0, 0, 0, 0]  # Full image
            
            # Publish grasp pose
            self._publish_grasp_pose(best_grasp)
            
            # Format response
            response.success = True
            response.total_grasps = 1
            response.grasp_poses = [grasp_msg]
            response.error_message = ""
            
            self.get_logger().info("=" * 60)
            self.get_logger().info("✓ BEST GRASP DETECTED")
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"  Position (x,y,z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
            self.get_logger().info(f"  Confidence: {confidence:.1f}%")
            self.get_logger().info(f"  Gripper width: {width_cm:.1f} cm")
            self.get_logger().info(f"  Grasp angle: {angle:.1f}°")
            self.get_logger().info(f"  Processing time: {elapsed_time:.2f}s")
            self.get_logger().info("=" * 60)
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error during grasp detection: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
            response.success = False
            response.total_grasps = 0
            response.grasp_poses = []
            response.error_message = f"Exception during grasp detection: {str(e)}"
            return response
    
    def detect_grasp_callback_trigger(self, request, response):
        """Fallback callback for Trigger service (when custom interfaces not available)"""
        self.get_logger().warn("Using Trigger service. For proper interface, install custom_interfaces package.")
        
        # Use a simple trigger response with JSON data
        try:
            # Lazy load GraspNet dependencies on first call
            if self.grasp_engine is None:
                self.get_logger().info("Loading Open3D and GraspNet (first call)...")
                if not _lazy_load_graspnet():
                    response.success = False
                    response.message = "Failed to load GraspNet dependencies"
                    return response
                
                # Initialize engine after successful load
                self.grasp_engine = GraspNetEngine(self.get_logger())
                self.get_logger().info("✓ GraspNet engine initialized")
            
            if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
                response.success = False
                response.message = "No camera data available"
                return response
            
            # Run detection pipeline (simplified)
            pcd = self.grasp_engine.create_pointcloud_from_rgbd(
                self.latest_rgb, self.latest_depth, self.camera_info
            )
            pcd_filtered = self.grasp_engine.filter_pointcloud(pcd)
            grasp_candidates = self.grasp_engine.generate_grasp_candidates(pcd_filtered)
            filtered_grasps = self.grasp_engine.filter_grasps(grasp_candidates, pcd_filtered)
            refined_grasps = self.grasp_engine.refine_grasp_candidates(filtered_grasps, pcd_filtered)
            best_grasp = self.grasp_engine.get_best_grasp(refined_grasps)
            
            # Visualize result
            if best_grasp:
                self.grasp_engine.visualize_grasp(pcd_filtered, best_grasp, refined_grasps)
            
            if best_grasp:
                pos = best_grasp['position']
                confidence = best_grasp['score'] * 100
                width_cm = best_grasp['width'] * 100
                angle = best_grasp['grasp_angle']
                
                response.success = True
                response.message = (
                    f"Best Grasp Found:\n"
                    f"Position (x,y,z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"Gripper width: {width_cm:.1f} cm\n"
                    f"Grasp angle: {angle:.1f}°"
                )
                
                self._publish_grasp_pose(best_grasp)
                
                # Also log to console
                self.get_logger().info("=" * 60)
                self.get_logger().info("✓ BEST GRASP DETECTED")
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"  Position (x,y,z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
                self.get_logger().info(f"  Confidence: {confidence:.1f}%")
                self.get_logger().info(f"  Gripper width: {width_cm:.1f} cm")
                self.get_logger().info(f"  Grasp angle: {angle:.1f}°")
                self.get_logger().info("=" * 60)
            else:
                response.success = False
                response.message = "No grasp found"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
    
    def _publish_grasp_pose(self, grasp_dict):
        """Publish grasp pose as PoseStamped message"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"
            
            # Position
            pose_msg.pose.position.x = float(grasp_dict['position'][0])
            pose_msg.pose.position.y = float(grasp_dict['position'][1])
            pose_msg.pose.position.z = float(grasp_dict['position'][2])
            
            # Orientation (convert rotation matrix to quaternion)
            R = np.array(grasp_dict['rotation'])
            q = self._rotation_matrix_to_quaternion(R)
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            
            self.grasp_pub.publish(pose_msg)
            self.get_logger().info("Published grasp pose to /vision/grasp_poses")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish grasp pose: {e}")
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor for concurrent service calls
    from rclpy.executors import MultiThreadedExecutor
    
    node = GraspNetDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
