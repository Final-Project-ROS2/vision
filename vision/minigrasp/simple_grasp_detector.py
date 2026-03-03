#!/usr/bin/env python3
"""
MiniGrasp - Simplified Grasp Detection
Captures ONE depth image and returns the BEST grasp pose

Usage:
    from minigrasp import simple_grasp_detector
    best_grasp = simple_grasp_detector.detect_best_grasp()
    print(f"Best grasp position: {best_grasp['position']}")
    print(f"Best grasp orientation: {best_grasp['rotation']}")
    print(f"Best grasp score: {best_grasp['score']}")
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from graspnetAPI.grasp import GraspGroup, Grasp
import sys
import os

# Import configuration
try:
    from . import config
except ImportError:
    import config


class SimpleGraspDetector:
    """Simple grasp detector that captures once and returns best grasp"""
    
    def __init__(self, config_obj=None):
        """
        Initialize the detector
        
        Args:
            config_obj: Configuration object (uses default config if None)
        """
        self.config = config_obj if config_obj else config
        self.pipeline = None
        self.align = None
        
    def _initialize_camera(self):
        """Initialize RealSense camera"""
        print("Initializing camera...")
        
        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Configure streams
            rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = self.pipeline.start(rs_config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create align object
            self.align = rs.align(rs.stream.color)
            
            print(f"✓ Camera initialized (depth_scale={self.depth_scale:.4f})")
            
            # Warm up camera (skip first few frames)
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize camera: {e}")
            return False
    
    def _capture_single_frame(self):
        """
        Capture a single depth and color frame
        
        Returns:
            color_image, depth_image, pointcloud or None if failed
        """
        if not self.pipeline:
            if not self._initialize_camera():
                return None, None, None
        
        print("\nCapturing frame...")
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("✗ Failed to get frames")
                return None, None, None
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get intrinsics
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Create point cloud
            pointcloud = self._create_pointcloud(depth_image, depth_intrinsics)
            
            print(f"✓ Captured frame: {color_image.shape} color, {depth_image.shape} depth")
            print(f"✓ Point cloud: {len(pointcloud.points)} points")
            
            return color_image, depth_image, pointcloud
            
        except Exception as e:
            print(f"✗ Failed to capture frame: {e}")
            return None, None, None
    
    def _create_pointcloud(self, depth_image, intrinsics):
        """
        Create point cloud from depth image
        
        Args:
            depth_image: Depth image in mm
            intrinsics: Camera intrinsics
            
        Returns:
            Open3D point cloud
        """
        # Convert depth to meters and apply filtering
        depth_meters = depth_image.astype(np.float32) * self.depth_scale
        
        # Filter invalid depths
        min_depth = self.config.MIN_VALID_DEPTH
        max_depth = self.config.MAX_VALID_DEPTH
        depth_meters[(depth_meters < min_depth) | (depth_meters > max_depth)] = 0
        
        # Create Open3D image
        o3d_depth = o3d.geometry.Image((depth_meters * 1000).astype(np.uint16))
        
        # Create camera intrinsics for Open3D
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            o3d_intrinsics,
            depth_scale=1000.0,  # Convert mm to meters
            depth_trunc=max_depth
        )
        
        return pcd
    
    def _filter_pointcloud(self, pointcloud):
        """
        Filter point cloud: workspace crop, RANSAC plane removal, outlier removal
        This removes table/floor background as recommended for GraspNet
        
        Args:
            pointcloud: Raw point cloud
            
        Returns:
            Filtered point cloud (object only, no background)
        """
        if len(pointcloud.points) == 0:
            print("  ✗ Point cloud is empty!")
            return pointcloud
        
        print("\nFiltering point cloud (removing background)...")
        
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
        
        # Show raw statistics
        print(f"  Raw points: {len(points)}")
        print(f"  Range: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        # Step 1: Workspace bounds filtering (pass-through filter)
        ws = self.config.WORKSPACE_BOUNDS
        mask = (
            (points[:, 0] >= ws['x_min']) & (points[:, 0] <= ws['x_max']) &
            (points[:, 1] >= ws['y_min']) & (points[:, 1] <= ws['y_max']) &
            (points[:, 2] >= ws['z_min']) & (points[:, 2] <= ws['z_max'])
        )
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        if colors is not None:
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        print(f"  → Workspace crop: {len(filtered_pcd.points)} points")
        
        if len(filtered_pcd.points) < 100:
            print("\n  ⚠ WARNING: Too few points after workspace crop!")
            print("  💡 Run: python check_camera.py to get correct WORKSPACE_BOUNDS")
            return filtered_pcd
        
        # Step 2: RANSAC plane segmentation (remove table/floor)
        if self.config.REMOVE_PLANE and len(filtered_pcd.points) >= 100:
            print(f"  → RANSAC plane removal...")
            
            plane_model, inliers = filtered_pcd.segment_plane(
                distance_threshold=self.config.RANSAC_DISTANCE_THRESHOLD,
                ransac_n=self.config.RANSAC_MIN_POINTS,
                num_iterations=self.config.RANSAC_NUM_ITERATIONS
            )
            
            if len(inliers) > 0:
                # Extract plane equation
                [a, b, c, d] = plane_model
                print(f"    Plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                print(f"    Plane points: {len(inliers)} ({100*len(inliers)/len(filtered_pcd.points):.1f}%)")
                
                # Remove plane (keep only outliers = object)
                filtered_pcd = filtered_pcd.select_by_index(inliers, invert=True)
                print(f"  → After plane removal: {len(filtered_pcd.points)} points (object only)")
                
                # Store plane equation for potential extrusion
                plane_equation = [a, b, c, d]
            else:
                print("    No dominant plane found (good - object may be floating)")
                plane_equation = None
        else:
            plane_equation = None
        
        
        if len(filtered_pcd.points) < 50:
            print("\n  ⚠ WARNING: Too few points after plane removal!")
            print("  💡 Possible issues:")
            print("     - No object in workspace (only table visible)")
            print("     - Object is part of the plane (very flat)")
            print("     - Workspace bounds too tight")
            return filtered_pcd
        
        # Step 2.5: Handle top-view occlusion (create 3D volume from 2D top shell)
        if self.config.EXTRUDE_TOP_SURFACE and plane_equation is not None:
            filtered_pcd = self._extrude_top_surface(filtered_pcd, plane_equation)
        
        # Step 3: Statistical outlier removal (clean floating noise)
        if len(filtered_pcd.points) > 10:
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            print(f"  → Outlier removal: {len(filtered_pcd.points)} points")
        
        # Step 4: Voxel downsampling (uniform density for GraspNet)
        if len(filtered_pcd.points) > 10000:
            original_count = len(filtered_pcd.points)
            filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)  # 5mm voxels
            print(f"  → Voxel downsampling: {len(filtered_pcd.points)} points (was {original_count})")
        
        # Step 5: Estimate normals (helps with grasp quality)
        if len(filtered_pcd.points) > 0:
            filtered_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        print(f"✓ Filtering complete: {len(filtered_pcd.points)} object points (background removed)")
        
        return filtered_pcd
    
    def _find_densest_region(self, points):
        """
        Find the densest region in point cloud (largest graspable area)
        
        Args:
            points: Object point cloud as numpy array
            
        Returns:
            Center of densest region (3D point)
        """
        # Use grid-based density estimation
        # Divide space into 3cm x 3cm x 3cm voxels and count points
        voxel_size = 0.03
        
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        # Create voxel grid
        grid_dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
        
        # Avoid too large grids
        if np.prod(grid_dims) > 10000:
            # Fall back to simple centroid if grid too large
            return points.mean(axis=0)
        
        # Assign each point to a voxel
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
        
        # Count points in each voxel
        voxel_counts = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_counts:
                voxel_counts[key] = []
            voxel_counts[key].append(i)
        
        # Find voxel with most points
        densest_voxel = max(voxel_counts.items(), key=lambda x: len(x[1]))
        densest_indices = densest_voxel[1]
        
        # Return center of points in densest voxel
        densest_points = points[densest_indices]
        return densest_points.mean(axis=0)
    
    def _extrude_top_surface(self, pointcloud, plane_equation):
        """
        Extrude top surface downward to create 3D volume from 2D shell
        Handles top-view occlusion problem
        
        Args:
            pointcloud: Object point cloud (top surface only)
            plane_equation: [a, b, c, d] of table plane
            
        Returns:
            Extended point cloud with synthetic volume
        """
        print(f"  → Top-view extrusion (solving occlusion)...")
        
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
        
        if len(points) == 0:
            return pointcloud
        
        # Calculate extrusion direction (downward toward table)
        [a, b, c, d] = plane_equation
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Extrusion direction is toward the table (negative normal)
        extrusion_dir = -plane_normal
        
        # Estimate object height from point cloud thickness
        # Project points onto plane normal to get height variation
        heights = np.dot(points, plane_normal)
        point_thickness = heights.max() - heights.min()
        
        # Use configured depth or estimated thickness
        if point_thickness < 0.01:  # Very thin (< 1cm) = top-view only
            extrusion_depth = self.config.EXTRUSION_DEPTH
            print(f"    Detected thin top view ({point_thickness*100:.1f}cm thick)")
            print(f"    Extruding {extrusion_depth*100:.1f}cm downward to create volume")
        else:
            # Already has some thickness, reduce extrusion
            extrusion_depth = max(0.02, self.config.EXTRUSION_DEPTH - point_thickness)
            print(f"    Detected {point_thickness*100:.1f}cm thickness")
            print(f"    Adding {extrusion_depth*100:.1f}cm extrusion")
        
        # Create extruded points (multiple layers)
        num_layers = 5  # Create 5 layers downward
        extruded_points_list = [points]  # Start with original
        if colors is not None:
            extruded_colors_list = [colors]
        
        for i in range(1, num_layers):
            # Extrude each layer progressively deeper
            offset = extrusion_dir * (extrusion_depth * i / num_layers)
            layer_points = points + offset
            extruded_points_list.append(layer_points)
            
            if colors is not None:
                extruded_colors_list.append(colors * 0.9)  # Slightly darker
        
        # Combine all layers
        all_points = np.vstack(extruded_points_list)
        
        extruded_pcd = o3d.geometry.PointCloud()
        extruded_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        if colors is not None:
            all_colors = np.vstack(extruded_colors_list)
            extruded_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        print(f"    Created 3D volume: {len(points)} → {len(all_points)} points")
        
        return extruded_pcd
    
    def _generate_grasp_candidates(self, pointcloud):
        """
        Generate grasp candidates from point cloud
        
        Args:
            pointcloud: Filtered point cloud
            
        Returns:
            GraspGroup with candidates
        """
        print("\nGenerating grasp candidates...")
        
        if len(pointcloud.points) < 100:
            print("✗ Too few points for grasp generation")
            return None
        
        points = np.asarray(pointcloud.points)
        normals = np.asarray(pointcloud.normals) if pointcloud.has_normals() else None
        
        # Calculate object center and extents
        center = points.mean(axis=0)
        extents = points.max(axis=0) - points.min(axis=0)
        
        print(f"  Object center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"  Object size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}] m")
        
        # Find densest region (largest grasp area)
        densest_center = self._find_densest_region(points)
        print(f"  Densest region: [{densest_center[0]:.3f}, {densest_center[1]:.3f}, {densest_center[2]:.3f}]")
        
        gg = GraspGroup()
        num_candidates = self.config.NUM_GRASP_CANDIDATES
        
        # Generate grasps focused on densest region
        for i in range(num_candidates):
            # Random approach angles
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Heavily favor densest region (80% of grasps there)
            if np.random.rand() < 0.8:
                # Focus on densest region with small random offset
                offset = np.random.uniform(-1, 1, size=3) * extents * 0.1
                translation = densest_center + offset
            else:
                # Some grasps at geometric center for diversity
                offset = np.random.uniform(-1, 1, size=3) * extents * 0.15
                translation = center + offset
            
            # Create rotation matrix
            # Approach direction (z-axis of gripper)
            approach = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])
            
            # Closing direction (x-axis of gripper) - perpendicular to approach
            closing = np.array([-np.sin(theta), np.cos(theta), 0])
            
            # Binormal (y-axis)
            binormal = np.cross(approach, closing)
            
            rotation = np.column_stack([closing, binormal, approach])
            
            # Calculate required grasp width at this point
            # Measure distance across object in closing direction
            width = self._estimate_grasp_width(points, translation, closing)
            
            # Calculate grasp quality score based on multiple factors
            score = self._calculate_grasp_quality(
                points, translation, rotation, width, densest_center, normals
            )
            
            # Standard gripper dimensions
            height = 0.02
            depth = 0.02
            
            grasp = Grasp(score, width, height, depth, rotation, translation, 0)
            gg.add(grasp)
        
        # Sort by score
        gg = gg.sort_by_score()
        
        print(f"✓ Generated {len(gg)} grasp candidates")
        if len(gg) > 0:
            print(f"  Width range: {min(g.width for g in gg)*100:.1f} - {max(g.width for g in gg)*100:.1f} cm")
            print(f"  Score range: {min(g.score for g in gg):.3f} - {max(g.score for g in gg):.3f}")
        
        return gg
    
    def _estimate_grasp_width(self, points, grasp_pos, closing_direction):
        """
        Estimate required grasp width at a position
        
        Args:
            points: Object point cloud
            grasp_pos: Grasp position
            closing_direction: Direction fingers close
            
        Returns:
            Required width in meters
        """
        # Find points near the grasp position
        distances = np.linalg.norm(points - grasp_pos, axis=1)
        nearby_mask = distances < 0.05  # 5cm radius
        
        if not np.any(nearby_mask):
            # No nearby points, use default
            return self.config.GRIPPER_WIDTH * 0.7
        
        nearby_points = points[nearby_mask]
        
        # Project points onto closing direction
        projections = np.dot(nearby_points - grasp_pos, closing_direction)
        
        if len(projections) > 0:
            # Width is distance from min to max projection
            width = projections.max() - projections.min()
            
            # Add safety margin
            width = width * 1.1
            
            # Clamp to gripper limits
            min_width = 0.01  # 1cm minimum
            max_width = self.config.GRIPPER_WIDTH
            width = np.clip(width, min_width, max_width)
        else:
            width = self.config.GRIPPER_WIDTH * 0.7
        
        return width
    
    def _calculate_grasp_quality(self, points, position, rotation, width, 
                                  object_center, normals=None):
        """
        Calculate grasp quality score based on multiple factors
        Focus heavily on densest region
        
        Args:
            points: Object points
            position: Grasp position
            rotation: Grasp rotation matrix
            width: Required grasp width
            object_center: Center of densest region (not geometric center)
            normals: Point normals (optional)
            
        Returns:
            Quality score (0-1)
        """
        score = 0.4  # Base score
        
        # Factor 1: Distance from densest region (heavily favor this!)
        dist_to_center = np.linalg.norm(position - object_center)
        center_score = np.exp(-dist_to_center / 0.05)  # Steep decay - stay close!
        score += 0.4 * center_score  # Increased weight from 0.2 to 0.4
        
        # Factor 2: Width utilization (prefer using 40-80% of gripper range)
        width_ratio = width / self.config.GRIPPER_WIDTH
        if 0.4 <= width_ratio <= 0.8:
            width_score = 1.0
        else:
            width_score = 0.5
        score += 0.1 * width_score  # Reduced weight
        
        # Factor 3: Point density near grasp (more points = larger area)
        distances = np.linalg.norm(points - position, axis=1)
        nearby_count = np.sum(distances < 0.04)  # 4cm radius
        density_score = min(nearby_count / 30.0, 1.0)  # Normalize
        score += 0.2 * density_score  # Increased weight
        
        # Clamp to [0, 1]
        score = np.clip(score, 0.0, 1.0)
        
        return score
    
    def _filter_grasps(self, grasp_group, pointcloud):
        """
        Filter grasps by collision and feasibility
        
        Args:
            grasp_group: Candidate grasps
            pointcloud: Object point cloud
            
        Returns:
            Filtered grasp group
        """
        if grasp_group is None or len(grasp_group) == 0:
            return grasp_group
        
        print("\nFiltering grasps...")
        
        initial_count = len(grasp_group)
        print(f"  Starting with: {initial_count} candidates")
        
        # Filter by score threshold
        filtered_gg = GraspGroup()
        for grasp in grasp_group:
            if grasp.score >= self.config.MIN_GRASP_SCORE:
                filtered_gg.add(grasp)
        
        removed_by_score = initial_count - len(filtered_gg)
        print(f"  Score threshold (>={self.config.MIN_GRASP_SCORE}): {len(grasp_group)} → {len(filtered_gg)} grasps")
        if removed_by_score > 0:
            print(f"    Removed {removed_by_score} low-score grasps")
        
        # Apply collision filter
        if len(pointcloud.points) > 0:
            filtered_gg = self._collision_filter(filtered_gg, pointcloud)
        
        # COMMENTED OUT: Apply approach angle filter (too restrictive initially)
        # filtered_gg = self._approach_filter(filtered_gg)
        print(f"  ⚠ Approach angle filter DISABLED (allows all angles)")
        
        print(f"✓ Final grasps: {len(filtered_gg)}")
        
        return filtered_gg
    
    def _collision_filter(self, grasp_group, pointcloud):
        """Simple collision detection (very permissive)"""
        points = np.asarray(pointcloud.points)
        
        if len(points) == 0:
            return grasp_group
        
        filtered_gg = GraspGroup()
        num_collided = 0
        
        for grasp in grasp_group:
            # Check if any points are too close to gripper fingers
            R_inv = grasp.rotation_matrix.T
            t = grasp.translation
            
            # Transform points to gripper frame
            pts_gripper = (points - t) @ grasp.rotation_matrix
            
            # Very relaxed box collision check (only reject obvious collisions)
            collision = np.any(
                (np.abs(pts_gripper[:, 0]) < 0.005) &  # Very thin finger collision zone (5mm)
                (np.abs(pts_gripper[:, 1]) < grasp.width/2 + 0.01) &
                (np.abs(pts_gripper[:, 2]) < 0.005)  # Very thin approach collision (5mm)
            )
            
            if not collision:
                filtered_gg.add(grasp)
            else:
                num_collided += 1
        
        print(f"  Collision filter (very permissive): removed {num_collided} grasps")
        return filtered_gg
    
    def _approach_filter(self, grasp_group):
        """Filter by approach angle"""
        target_vector = np.array(self.config.APPROACH_TARGET_VECTOR)
        target_vector = target_vector / np.linalg.norm(target_vector)
        max_angle = self.config.MAX_APPROACH_ANGLE
        
        filtered_gg = GraspGroup()
        num_removed = 0
        
        for grasp in grasp_group:
            # Approach vector is z-axis (column 2) of rotation matrix
            approach = grasp.rotation_matrix[:, 2]
            approach = approach / np.linalg.norm(approach)
            
            # Calculate angle
            cos_theta = np.clip(np.dot(approach, target_vector), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            
            if angle <= max_angle:
                filtered_gg.add(grasp)
            else:
                num_removed += 1
        
        print(f"  Approach filter: removed {num_removed} grasps")
        return filtered_gg
    
    def _refine_grasp_candidates(self, grasp_group, pointcloud):
        """
        Refine grasps to align with actual point clusters
        Solves 'floating grasp' problem by centering on real points
        
        Args:
            grasp_group: Initial grasp candidates
            pointcloud: Object point cloud
            
        Returns:
            Refined grasp group
        """
        if not self.config.ENABLE_GRASP_REFINEMENT:
            return grasp_group
        
        print(f"  → Refining grasps (aligning to point clusters)...")
        
        points = np.asarray(pointcloud.points)
        normals = np.asarray(pointcloud.normals) if pointcloud.has_normals() else None
        
        if len(points) == 0:
            return grasp_group
        
        refined_gg = GraspGroup()
        num_improved = 0
        num_removed = 0
        
        for grasp in grasp_group:
            # Find nearby points
            distances = np.linalg.norm(points - grasp.translation, axis=1)
            nearby_mask = distances < 0.08  # Within 8cm (increased from 5cm)
            
            nearby_count = np.sum(nearby_mask)
            if nearby_count < self.config.MIN_POINTS_FOR_GRASP:
                num_removed += 1
                continue
            
            nearby_points = points[nearby_mask]
            
            if self.config.CENTER_ON_POINTS:
                # Re-center grasp on actual point cluster
                cluster_center = nearby_points.mean(axis=0)
                
                # Move grasp to cluster center
                translation_offset = cluster_center - grasp.translation
                
                # Only apply if reasonable (< 3cm adjustment)
                if np.linalg.norm(translation_offset) < 0.03:
                    new_translation = cluster_center
                    
                    # Adjust approach to align with local surface normal
                    if normals is not None:
                        nearby_normals = normals[nearby_mask]
                        avg_normal = nearby_normals.mean(axis=0)
                        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-6)
                        
                        # Get current approach vector
                        approach = grasp.rotation_matrix[:, 2]
                        
                        # Check if approach is reasonably aligned with normal
                        alignment = np.dot(approach, avg_normal)
                        
                        if abs(alignment) > 0.3:  # At least 30% aligned
                            # Adjust approach to match surface better
                            # Keep closing direction, update approach
                            closing = grasp.rotation_matrix[:, 0]
                            
                            # New approach along surface normal (downward)
                            new_approach = -avg_normal / np.linalg.norm(avg_normal)
                            
                            # Recompute binormal
                            new_binormal = np.cross(closing, new_approach)
                            new_binormal = new_binormal / (np.linalg.norm(new_binormal) + 1e-6)
                            
                            # Recompute closing to ensure orthogonality
                            new_closing = np.cross(new_binormal, new_approach)
                            new_closing = new_closing / (np.linalg.norm(new_closing) + 1e-6)
                            
                            new_rotation = np.column_stack([new_closing, new_binormal, new_approach])
                            
                            # Create refined grasp
                            refined_grasp = Grasp(
                                grasp.score * 1.05,  # Slight boost for refinement
                                grasp.width,
                                grasp.height,
                                grasp.depth,
                                new_rotation,
                                new_translation,
                                grasp.object_id
                            )
                            refined_gg.add(refined_grasp)
                            num_improved += 1
                            continue
                    
                    # Surface refinement not applied, just center
                    refined_grasp = Grasp(
                        grasp.score * 1.02,
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
            
            # Keep original if refinement not applied
            refined_gg.add(grasp)
        
        print(f"    Improved {num_improved} grasps, removed {num_removed} floating grasps")
        return refined_gg.sort_by_score()
    
    def _get_best_grasp(self, grasp_group):
        """
        Extract the best grasp from the group
        
        Args:
            grasp_group: Filtered grasp group
            
        Returns:
            dict with best grasp information
        """
        if grasp_group is None or len(grasp_group) == 0:
            return None
        
        # Get top grasp (already sorted by score)
        best = grasp_group[0]
        
        # Calculate additional useful metrics
        approach_vector = best.rotation_matrix[:, 2]
        closing_vector = best.rotation_matrix[:, 0]
        
        # Estimate required gripper force (simplified)
        # Force increases with smaller objects
        width_ratio = best.width / self.config.GRIPPER_WIDTH
        estimated_force = 20.0 + (1.0 - width_ratio) * 30.0  # 20-50N range
        
        # Convert to simple dictionary format
        result = {
            # Position and orientation
            'position': best.translation.tolist(),
            'rotation': best.rotation_matrix.tolist(),
            'approach_vector': approach_vector.tolist(),
            'closing_vector': closing_vector.tolist(),
            
            # Grasp parameters
            'width': float(best.width),
            'height': float(best.height),
            'depth': float(best.depth),
            'score': float(best.score),
            
            # Additional metrics
            'width_ratio': float(width_ratio),  # % of max gripper opening
            'estimated_force': float(estimated_force),  # Estimated required force (N)
            'grasp_center': best.translation.tolist(),  # Same as position
            
            # Robot control hints
            'pre_grasp_offset': 0.05,  # Move 5cm back along approach before closing
            'grasp_speed': 0.5 if width_ratio > 0.7 else 0.3,  # Slower for small objects
        }
        
        return result
    
    def detect_best_grasp(self, visualize=False):
        """
        Main function: Capture once and return best grasp
        
        Args:
            visualize: If True, show visualization
            
        Returns:
            dict with best grasp information or None if failed
        """
        print("=" * 60)
        print("MiniGrasp - Simple Grasp Detection")
        print("=" * 60)
        
        try:
            # Step 1: Capture frame
            color, depth, pcd = self._capture_single_frame()
            
            if pcd is None:
                print("\n✗ Failed to capture frame")
                return None
            
            # Step 2: Filter point cloud
            pcd_filtered = self._filter_pointcloud(pcd)
            
            if len(pcd_filtered.points) < 100:
                print("\n✗ Too few points after filtering")
                return None
            
            # Step 3: Generate grasp candidates
            grasp_candidates = self._generate_grasp_candidates(pcd_filtered)
            
            if grasp_candidates is None or len(grasp_candidates) == 0:
                print("\n✗ No grasp candidates generated")
                return None
            
            # Step 4: Filter grasps
            filtered_grasps = self._filter_grasps(grasp_candidates, pcd_filtered)
            
            if filtered_grasps is None or len(filtered_grasps) == 0:
                print("\n✗ No valid grasps found")
                return None
            
            # Step 5: Refine grasps (align to point clusters)
            refined_grasps = self._refine_grasp_candidates(filtered_grasps, pcd_filtered)
            
            if refined_grasps is None or len(refined_grasps) == 0:
                print("\n✗ No grasps remain after refinement")
                return None
            
            # Step 6: Get best grasp
            best_grasp = self._get_best_grasp(refined_grasps)
            
            # Step 7: Visualize if requested
            if visualize:
                self._visualize_result(pcd_filtered, refined_grasps)
            
            print("\n" + "=" * 60)
            print("✓ BEST GRASP DETECTED")
            print("=" * 60)
            print(f"Position (x,y,z): [{best_grasp['position'][0]:.3f}, {best_grasp['position'][1]:.3f}, {best_grasp['position'][2]:.3f}] m")
            print(f"Confidence: {best_grasp['score']:.1%}")
            print(f"\nGripper Settings:")
            print(f"  Width: {best_grasp['width']*100:.1f} cm ({best_grasp['width_ratio']:.1%} of max)")
            print(f"  Est. Force: {best_grasp['estimated_force']:.1f} N")
            print(f"  Grasp Speed: {best_grasp['grasp_speed']:.1f} (0=slow, 1=fast)")
            print(f"\nRobot Control:")
            print(f"  Approach vector: [{best_grasp['approach_vector'][0]:+.3f}, {best_grasp['approach_vector'][1]:+.3f}, {best_grasp['approach_vector'][2]:+.3f}]")
            print(f"  Pre-grasp offset: {best_grasp['pre_grasp_offset']*100:.1f} cm")
            print("=" * 60)
            
            return best_grasp
            
        except Exception as e:
            print(f"\n✗ Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up camera
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
    
    def _visualize_result(self, pointcloud, grasp_group):
        """Visualize point cloud with grasps"""
        print("\nVisualizing result...")
        
        # Create grasp geometries (show top 5)
        geometries = [pointcloud]
        
        for i, grasp in enumerate(grasp_group[:5]):
            color = [1, 0, 0] if i == 0 else [0, 1, 0]  # Red for best, green for others
            grasp_mesh = grasp.to_open3d_geometry()
            grasp_mesh.paint_uniform_color(color)
            geometries.append(grasp_mesh)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Best Grasp Detection (Red = Best, Green = Alternatives)"
        )


def detect_best_grasp(visualize=False):
    """
    Convenience function: Detect best grasp with default config
    
    Args:
        visualize: Show visualization window
        
    Returns:
        dict with best grasp or None
    """
    detector = SimpleGraspDetector()
    return detector.detect_best_grasp(visualize=visualize)


if __name__ == "__main__":
    # Run with visualization
    result = detect_best_grasp(visualize=True)
    
    if result:
        print("\n✓ Success! Best grasp found.")
    else:
        print("\n✗ Failed to detect grasp.")
