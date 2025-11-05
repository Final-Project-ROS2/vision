#!/usr/bin/env python3
"""
GraspNet Detector Node
Performs 6D grasp pose estimation on RGB-D images

This node:
1. Subscribes to /camera/image_raw (RGB) and /camera/depth/image_raw (Depth)
2. Runs GraspNet-1Billion inference to detect grasp poses
3. Outputs grasp poses with 6D pose (position + orientation) and quality scores
4. Visualizes grasps on RGB image

Usage:
    ros2 run vision graspnet_detector
    
Service:
    ros2 service call /vision/detect_grasps std_srvs/srv/Trigger

Dependencies:
    - Camera must be publishing to /camera/image_raw and /camera/depth/image_raw
    - GraspNet model (optional - will use geometric fallback if not available)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
import time
from pathlib import Path

# Try to import GraspNet (if available)
try:
    # Import GraspNet dependencies
    import torch
    GRASPNET_AVAILABLE = False  # Set to True if you have GraspNet installed
    print("⚠️ GraspNet not available. Using geometric grasp estimation.")
except ImportError:
    GRASPNET_AVAILABLE = False
    print("⚠️ GraspNet dependencies not available. Using geometric grasp estimation.")


class GraspNetDetector(Node):
    """
    GraspNet-based grasp pose detector
    
    Subscribes to:
        - /camera/image_raw (RGB images)
        - /camera/depth/image_raw (Depth images)
        - /camera/camera_info (Camera intrinsics)
    
    Services:
        - /vision/detect_grasps (Trigger grasp detection)
    
    Publishes:
        - /vision/grasp_poses (PoseStamped messages for each grasp)
    
    Display:
        - Shows RGB image with grasp visualizations
    """
    
    def __init__(self):
        super().__init__('graspnet_detector')
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest sensor data
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.captured_rgb = None
        self.captured_depth = None
        self.frame_captured = False
        self.frame_counter = 0
        
        # Grasp detection results
        self.latest_grasps = []
        
        # Output directory for saving results
        self.output_dir = Path.home() / "graspnet_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # OpenCV window for visualization
        self.window_name = "GraspNet Detector - Grasp Poses"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 750)
        
        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.detection_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize GraspNet model (if available)
        self._init_graspnet_model()
        
        # Subscribe to camera topics
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            self.image_qos
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            self.image_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            self.detection_qos
        )
        
        # Create grasp detection service
        self.grasp_service = self.create_service(
            Trigger,
            '/vision/detect_grasps',
            self.detect_grasps_callback
        )
        
        # Publisher for grasp poses
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/vision/grasp_poses',
            self.detection_qos
        )
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 GraspNet Detector Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 Subscribed to: /camera/image_raw")
        self.get_logger().info("📡 Subscribed to: /camera/depth/image_raw")
        self.get_logger().info("📡 Subscribed to: /camera/camera_info")
        self.get_logger().info(f"📁 Output Directory: {self.output_dir}")
        self.get_logger().info(f"👁️  OpenCV Window: '{self.window_name}'")
        self.get_logger().info("🔧 Service: /vision/detect_grasps")
        self.get_logger().info("📤 Publishing to: /vision/grasp_poses")
        self.get_logger().info("=" * 80)
        self.get_logger().info("💡 Usage: ros2 service call /vision/detect_grasps std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)
    
    def _init_graspnet_model(self):
        """Initialize GraspNet model (if available)"""
        if GRASPNET_AVAILABLE:
            self.get_logger().info("🔧 Loading GraspNet model...")
            try:
                # Load GraspNet model here
                # self.graspnet_model = load_graspnet_model()
                self.get_logger().info("✅ GraspNet model loaded successfully")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to load GraspNet model: {e}")
                self.get_logger().info("   Using geometric grasp estimation instead")
        else:
            self.get_logger().info("ℹ️  Using geometric grasp estimation (GraspNet not available)")
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image messages"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_counter += 1
            
            # Capture first frame
            if not self.frame_captured and self.latest_depth is not None:
                self.captured_rgb = self.latest_rgb.copy()
                self.captured_depth = self.latest_depth.copy()
                self.frame_captured = True
                self.get_logger().info(f"📸 Captured RGB-D frame {self.frame_counter}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
    
    def depth_callback(self, msg: Image):
        """Handle depth image messages"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Clean up depth data
            self.latest_depth = np.nan_to_num(self.latest_depth, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Capture first frame if RGB is available
            if not self.frame_captured and self.latest_rgb is not None:
                self.captured_rgb = self.latest_rgb.copy()
                self.captured_depth = self.latest_depth.copy()
                self.frame_captured = True
                self.get_logger().info(f"📸 Captured RGB-D frame {self.frame_counter}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
    
    def detect_grasps_callback(self, request, response):
        """Service callback for grasp detection"""
        try:
            # Check if RGB-D data is available
            if self.captured_rgb is None or self.captured_depth is None:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "No RGB-D data available. Waiting for camera...",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No RGB-D data captured yet")
                return response
            
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 Running GraspNet Detection on RGB-D")
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"   RGB shape: {self.captured_rgb.shape}")
            self.get_logger().info(f"   Depth shape: {self.captured_depth.shape}")
            
            # Run grasp detection
            start_time = time.time()
            grasps = self._detect_grasps(self.captured_rgb, self.captured_depth)
            detection_time = int((time.time() - start_time) * 1000)
            
            if not grasps:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "No grasps detected",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No grasps detected")
                return response
            
            self.latest_grasps = grasps
            
            # Build JSON response
            grasp_results = self._build_grasp_schema(grasps, detection_time)
            
            # Save results
            json_path = self._save_json_output(grasp_results)
            grasp_results['output']['json_file'] = str(json_path)
            
            # Publish grasp poses
            self._publish_grasp_poses(grasps)
            
            # Visualize results
            self._visualize_grasps(self.captured_rgb, grasps)
            
            response.success = True
            response.message = json.dumps(grasp_results, indent=2)
            
            # Print JSON to terminal
            self.get_logger().info("=" * 80)
            self.get_logger().info("📋 GRASP DETECTION JSON OUTPUT:")
            self.get_logger().info("=" * 80)
            self.get_logger().info(response.message)
            self.get_logger().info("=" * 80)
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"✅ GRASP DETECTION COMPLETE")
            self.get_logger().info(f"   Detected: {len(grasps)} grasp poses")
            self.get_logger().info(f"   Detection time: {detection_time}ms")
            self.get_logger().info(f"   JSON saved to: {json_path}")
            self.get_logger().info("=" * 80)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"❌ Grasp detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def _detect_grasps(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """
        Detect grasp poses from RGB-D images
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W)
            
        Returns:
            List of grasp dictionaries with pose, quality, and metadata
        """
        grasps = []
        
        if GRASPNET_AVAILABLE:
            # Use actual GraspNet model
            grasps = self._graspnet_inference(rgb_image, depth_image)
        else:
            # Use geometric grasp estimation
            grasps = self._geometric_grasp_estimation(rgb_image, depth_image)
        
        return grasps
    
    def _geometric_grasp_estimation(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """
        Geometric grasp estimation (fallback when GraspNet is not available)
        
        Detects object contours and proposes grasps based on:
        - Object centroid
        - Principal axis orientation
        - Depth information
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            
        Returns:
            List of grasp poses
        """
        h, w = rgb_image.shape[:2]
        grasps = []
        
        # Convert to grayscale and detect objects
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        min_area = (w * h) * 0.001  # Minimum 0.1% of image
        max_area = (w * h) * 0.5    # Maximum 50% of image
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # Get object properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Skip if centroid is out of bounds
            if not (0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]):
                continue
            
            # Get depth at centroid
            depth_value = depth_image[cy, cx]
            if depth_value == 0 or np.isnan(depth_value):
                # Try to get average depth from nearby pixels
                roi_size = 5
                y1 = max(0, cy - roi_size)
                y2 = min(depth_image.shape[0], cy + roi_size)
                x1 = max(0, cx - roi_size)
                x2 = min(depth_image.shape[1], cx + roi_size)
                depth_roi = depth_image[y1:y2, x1:x2]
                valid_depths = depth_roi[depth_roi > 0]
                if len(valid_depths) > 0:
                    depth_value = np.median(valid_depths)
                else:
                    depth_value = 0.5  # Default depth in meters
            
            # Convert depth to meters (adjust based on your camera)
            depth_m = float(depth_value) / 1000.0 if depth_value > 100 else float(depth_value)
            
            # Get orientation using PCA
            try:
                # Fit ellipse to get orientation
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]  # Angle in degrees
                else:
                    angle = 0.0
            except:
                angle = 0.0
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Calculate grasp width (estimate from contour)
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            grasp_width = min(width, height) * 0.8  # 80% of smaller dimension
            grasp_width_m = grasp_width / 1000.0  # Convert to meters
            
            # Convert pixel coordinates to camera frame (simplified)
            # Assuming standard pinhole camera model
            fx = 525.0  # Focal length (adjust for your camera)
            fy = 525.0
            cx_cam = w / 2.0
            cy_cam = h / 2.0
            
            if self.camera_info is not None:
                K = np.array(self.camera_info.k).reshape(3, 3)
                fx = K[0, 0]
                fy = K[1, 1]
                cx_cam = K[0, 2]
                cy_cam = K[1, 2]
            
            # 3D position in camera frame
            x_3d = (cx - cx_cam) * depth_m / fx
            y_3d = (cy - cy_cam) * depth_m / fy
            z_3d = depth_m
            
            # Orientation quaternion (simplified - grasp approaching from above)
            # Rotation around Z-axis by angle
            qz = np.sin(angle_rad / 2.0)
            qw = np.cos(angle_rad / 2.0)
            
            # Quality score based on depth validity and contour properties
            quality = 0.5  # Base quality
            if depth_value > 0:
                quality += 0.2
            if area > min_area * 10:  # Larger objects
                quality += 0.2
            quality = min(1.0, quality)
            
            grasp = {
                "grasp_id": i,
                "position": {
                    "x": float(x_3d),
                    "y": float(y_3d),
                    "z": float(z_3d)
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": float(qz),
                    "w": float(qw)
                },
                "quality_score": float(quality),
                "grasp_width": float(grasp_width_m),
                "approach_angle": float(angle),
                "pixel_location": [int(cx), int(cy)],
                "depth_value": float(depth_m),
                "contour_area": int(area)
            }
            
            grasps.append(grasp)
        
        # Sort grasps by quality score (descending)
        grasps.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Keep top 10 grasps
        grasps = grasps[:10]
        
        self.get_logger().info(f"   Geometric estimation: Found {len(grasps)} grasp poses")
        
        return grasps
    
    def _graspnet_inference(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """
        Run GraspNet model inference (placeholder for actual implementation)
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            
        Returns:
            List of grasp poses from GraspNet
        """
        # TODO: Implement actual GraspNet inference
        # This is a placeholder that would call the actual GraspNet model
        self.get_logger().info("   Running GraspNet model inference...")
        return []
    
    def _build_grasp_schema(self, grasps: List[Dict], detection_time: int) -> Dict:
        """
        Build JSON schema for grasp results
        
        Args:
            grasps: List of grasp dictionaries
            detection_time: Detection time in milliseconds
            
        Returns:
            Dictionary with grasp results in JSON schema format
        """
        schema = {
            "pipeline": "graspnet",
            "success": True,
            "input": {
                "rgb_shape": list(self.captured_rgb.shape),
                "depth_shape": list(self.captured_depth.shape),
                "frame_id": f"frame_{self.frame_counter:06d}"
            },
            "output": {
                "grasps": grasps,
                "summary": {
                    "total_grasps": len(grasps),
                    "detection_time_ms": detection_time,
                    "top_quality_score": grasps[0]["quality_score"] if grasps else 0.0
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "method": "graspnet" if GRASPNET_AVAILABLE else "geometric",
                "output_directory": str(self.output_dir)
            }
        }
        
        return schema
    
    def _save_json_output(self, results: Dict) -> Path:
        """Save grasp results as JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"graspnet_results_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.get_logger().info(f"   ✅ JSON saved: {json_path}")
        return json_path
    
    def _publish_grasp_poses(self, grasps: List[Dict]):
        """Publish grasp poses to ROS topic"""
        for grasp in grasps:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"
            
            pos = grasp["position"]
            ori = grasp["orientation"]
            
            pose_msg.pose.position = Point(
                x=float(pos["x"]),
                y=float(pos["y"]),
                z=float(pos["z"])
            )
            pose_msg.pose.orientation = Quaternion(
                x=float(ori["x"]),
                y=float(ori["y"]),
                z=float(ori["z"]),
                w=float(ori["w"])
            )
            
            self.grasp_pub.publish(pose_msg)
    
    def _visualize_grasps(self, rgb_image: np.ndarray, grasps: List[Dict]):
        """Visualize grasp poses on RGB image and save"""
        vis_image = rgb_image.copy()
        
        # Color palette for different grasps
        colors = [
            (0, 255, 0),    # Green (best)
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 0, 128),  # Pink
            (128, 255, 0),  # Light Green
        ]
        
        for i, grasp in enumerate(grasps[:8]):  # Show top 8
            px, py = grasp["pixel_location"]
            quality = grasp["quality_score"]
            angle = grasp["approach_angle"]
            
            color = colors[i % len(colors)]
            
            # Draw grasp center
            cv2.circle(vis_image, (px, py), 5, color, -1)
            cv2.circle(vis_image, (px, py), 7, (255, 255, 255), 2)
            
            # Draw grasp orientation
            length = 40
            angle_rad = np.deg2rad(angle)
            end_x = int(px + length * np.cos(angle_rad))
            end_y = int(py + length * np.sin(angle_rad))
            cv2.arrowedLine(vis_image, (px, py), (end_x, end_y), color, 3, tipLength=0.3)
            
            # Draw perpendicular line (gripper width)
            perp_angle = angle_rad + np.pi/2
            width = 30
            px1 = int(px + width/2 * np.cos(perp_angle))
            py1 = int(py + width/2 * np.sin(perp_angle))
            px2 = int(px - width/2 * np.cos(perp_angle))
            py2 = int(py - width/2 * np.sin(perp_angle))
            cv2.line(vis_image, (px1, py1), (px2, py2), color, 2)
            
            # Draw label
            label = f"#{i}: Q={quality:.2f}"
            cv2.putText(vis_image, label, (px + 10, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add title
        title = f"GraspNet Detector | Grasps: {len(grasps)}"
        cv2.putText(vis_image, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = self.output_dir / f"grasps_visualization_{timestamp}.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        self.get_logger().info(f"   ✅ Visualization saved: {vis_path}")
        
        # Update display
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
    
    def visualization_callback(self):
        """Display current RGB image with grasps"""
        if self.latest_rgb is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        # Show latest RGB (grasp visualization is done in service callback)
        display_img = self.latest_rgb.copy()
        
        if not self.latest_grasps:
            cv2.putText(display_img, "Call /vision/detect_grasps service", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, display_img)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = GraspNetDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
