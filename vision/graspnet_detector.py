#!/usr/bin/env python3
"""
GraspNet Detector Node

Services:
    1. /vision/detect_grasp
       Use bounding boxes from /vision/detect_objects to find grasp pose for each object
       Returns grasp poses for all detected objects
       ros2 service call /vision/detect_grasp std_srvs/srv/Trigger

    2. /vision/detect_grasp_bb
       Find grasp position in specific bounding box region [x1,y1,x2,y2]
       ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"
    
    3. /vision/run_pipeline
       Run full SAM + CLIP + GraspNet pipeline
       Subscribes to /vision/sam_detections (SAMDetections message)
       Automatically runs grasp detection when SAM publishes
       No manual service call needed - automatic on SAM publish
       ros2 service call /vision/run_pipeline std_srvs/srv/Trigger

Setup:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 run vision clip_classifier
    Terminal 3: ros2 run vision graspnet_detector
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
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

# Import custom interfaces
try:
    from custom_interfaces.srv import DetectObjects, DetectGrasps, DetectGraspBBox
    from custom_interfaces.msg import SAMDetections, SAMDetection, GraspPose
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    CUSTOM_INTERFACES_AVAILABLE = False
    # Fallback type hints
    SAMDetections = None
    SAMDetection = None
    GraspPose = None
    DetectObjects = None
    DetectGrasps = None
    DetectGraspBBox = None
    print("Custom interfaces not available. Build custom_interfaces package first.")

# Try to import GraspNet (if available)
try:
    # Import GraspNet dependencies
    import torch
    GRASPNET_AVAILABLE = False  # Set to True if you have GraspNet installed
    print("GraspNet not available. Using geometric grasp estimation.")
except ImportError:
    GRASPNET_AVAILABLE = False
    print("GraspNet dependencies not available. Using geometric grasp estimation.")


class GraspNetDetector(Node):
    """
    GraspNet-based grasp pose detector
    
    Subscribes to:
        - /camera/image_raw (RGB images)
        - /camera/depth/image_raw (Depth images)
        - /camera/camera_info (Camera intrinsics)
        - /vision/sam_detections (SAMDetections for pipeline mode)
    
    Services:
        - /vision/detect_grasp (Use /vision/detect_objects to get bboxes, then detect grasps)
        - /vision/detect_grasp_bb (Detect grasp in specific bbox region)
        - /vision/run_pipeline (Auto-detect grasps when SAM publishes)
    
    Publishes:
        - /vision/grasp_poses (PoseStamped messages for each grasp)
    
    Display:
        - Shows RGB image with grasp visualizations
    """
    
    def __init__(self):
        super().__init__('graspnet_detector')
        
        # Create callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        
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
        self.latest_region_grasps = []
        
        # Service clients
        self.detect_objects_client = None
        if CUSTOM_INTERFACES_AVAILABLE:
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects',
                callback_group=self.callback_group
            )
        
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
        
        # Subscribe to SAM detections for pipeline mode
        if CUSTOM_INTERFACES_AVAILABLE:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                10
            )
        
        # Create grasp detection services
        # Always use Trigger for /vision/detect_grasp to ensure compatibility
        self.grasp_service = self.create_service(
            Trigger,
            '/vision/detect_grasp',
            self.detect_grasp_callback,
            callback_group=self.callback_group
        )
        
        if CUSTOM_INTERFACES_AVAILABLE:
            self.grasp_bb_service = self.create_service(
                DetectGraspBBox,
                '/vision/detect_grasp_bb',
                self.detect_grasp_bb_callback,
                callback_group=self.callback_group
            )
        
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/run_pipeline',
            self.run_pipeline_callback,
            callback_group=self.callback_group
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
        self.get_logger().info("GraspNet Detector Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Subscribed to: /camera/image_raw")
        self.get_logger().info("Subscribed to: /camera/depth/image_raw")
        self.get_logger().info("Subscribed to: /camera/camera_info")
        if CUSTOM_INTERFACES_AVAILABLE:
            self.get_logger().info("Subscribed to: /vision/sam_detections")
        self.get_logger().info(f"Output Directory: {self.output_dir}")
        self.get_logger().info(f"OpenCV Window: '{self.window_name}'")
        self.get_logger().info("Service: /vision/detect_grasp")
        if CUSTOM_INTERFACES_AVAILABLE:
            self.get_logger().info("Service: /vision/detect_grasp_bb")
        self.get_logger().info("Service: /vision/run_pipeline")
        self.get_logger().info("Publishing to: /vision/grasp_poses")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Usage:")
        self.get_logger().info("  ros2 service call /vision/detect_grasp std_srvs/srv/Trigger")
        if CUSTOM_INTERFACES_AVAILABLE:
            self.get_logger().info("  ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox \"{x1: 100, y1: 100, x2: 200, y2: 300}\"")
        self.get_logger().info("=" * 80)
    
    def _init_graspnet_model(self):
        """Initialize GraspNet model (if available)"""
        if GRASPNET_AVAILABLE:
            self.get_logger().info("Loading GraspNet model...")
            try:
                # Load GraspNet model here
                # self.graspnet_model = load_graspnet_model()
                self.get_logger().info("GraspNet model loaded successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to load GraspNet model: {e}")
                self.get_logger().info("   Using geometric grasp estimation instead")
        else:
            self.get_logger().info("Using geometric grasp estimation (GraspNet not available)")
    
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
                self.get_logger().info(f"Captured RGB-D frame {self.frame_counter}")
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
                self.get_logger().info(f"Captured RGB-D frame {self.frame_counter}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
    
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
                self.get_logger().info(f"Captured RGB-D frame {self.frame_counter}")
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
                self.get_logger().info(f"Captured RGB-D frame {self.frame_counter}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
    
    def sam_detections_callback(self, msg: 'SAMDetections'):
        """Handle incoming SAM detections for pipeline mode"""
        try:
            if self.captured_rgb is None or self.captured_depth is None:
                self.get_logger().warn("No RGB-D data captured, waiting for camera")
                return
            
            bboxes = []
            for det in msg.detections:
                bbox = list(det.bbox)
                if len(bbox) == 4:
                    bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            
            if not bboxes:
                self.get_logger().warn("SAM detections message has no valid bounding boxes")
                return
            
            self.get_logger().info(f"Received {len(bboxes)} SAM regions, detecting grasps...")
            
            # Detect grasps for all regions
            all_grasps = []
            for i, bbox in enumerate(bboxes):
                grasps = self._detect_grasps_in_bbox(self.captured_rgb, self.captured_depth, bbox)
                for grasp in grasps:
                    grasp['region_id'] = i
                    grasp['bbox'] = bbox
                all_grasps.extend(grasps)
            
            self.latest_region_grasps = all_grasps
            self.latest_grasps = []
            
            # Log results
            self.get_logger().info(f"Pipeline grasp detection complete: {len(all_grasps)} grasps found")
            for grasp in all_grasps[:5]:
                self.get_logger().info(
                    f"  Region {grasp['region_id']}: grasp_id={grasp['grasp_id']}, "
                    f"quality={grasp['quality_score']:.2f}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error handling SAM detections: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def detect_grasp_callback(self, request, response):
        """Service callback for /vision/detect_grasp - uses Trigger service"""
        try:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Grasp Detection Service Called")
            self.get_logger().info("=" * 60)
            
            if not CUSTOM_INTERFACES_AVAILABLE or self.detect_objects_client is None:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "Custom interfaces not available or detect_objects service not found",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("Custom interfaces not available")
                return response
            
            if self.captured_rgb is None or self.captured_depth is None:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "No RGB-D data available. Waiting for camera...",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No RGB-D data captured yet")
                return response
            
            # Call /vision/detect_objects service
            self.get_logger().info("Calling /vision/detect_objects service...")
            
            if not self.detect_objects_client.wait_for_service(timeout_sec=5.0):
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "/vision/detect_objects service not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("/vision/detect_objects service not available")
                return response
            
            detect_request = DetectObjects.Request()
            future = self.detect_objects_client.call_async(detect_request)
            
            # Wait for future with timeout
            start_time = time.time()
            timeout = 10.0
            while not future.done() and (time.time() - start_time) < timeout:
                time.sleep(0.01)
            
            if not future.done():
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "/vision/detect_objects service call timeout",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("Service call timeout")
                return response
            
            detect_response = future.result()
            
            if not detect_response.success:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": f"Object detection failed: {detect_response.error_message}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error(f"Object detection failed: {detect_response.error_message}")
                return response
            
            # Extract bboxes from detection response
            bboxes = []
            object_ids = []
            for i in range(detect_response.total_detections):
                bbox = [
                    detect_response.bbox_x1[i],
                    detect_response.bbox_y1[i],
                    detect_response.bbox_x2[i],
                    detect_response.bbox_y2[i]
                ]
                bboxes.append(bbox)
                object_ids.append(detect_response.object_ids[i])
            
            self.get_logger().info(f"Detected {len(bboxes)} objects, finding grasps...")
            
            # Store all bboxes for visualization (even those without grasps)
            self.latest_all_bboxes = bboxes
            self.latest_all_object_ids = object_ids
            
            # Detect grasps for each bounding box
            all_grasps = []
            grasp_data = []
            
            for i, bbox in enumerate(bboxes):
                grasps = self._detect_grasps_in_bbox(self.captured_rgb, self.captured_depth, bbox)
                
                if grasps:
                    for grasp in grasps:
                        grasp['object_id'] = object_ids[i]
                        grasp['bbox'] = bbox
                        all_grasps.append(grasp)
                        
                        # Store grasp data for JSON response
                        grasp_data.append({
                            "object_id": object_ids[i],
                            "bbox": bbox,
                            "position": grasp['position'],
                            "orientation": grasp['orientation'],
                            "quality_score": grasp['quality_score'],
                            "grasp_width": grasp['grasp_width'],
                            "approach_angle": grasp['approach_angle']
                        })
                else:
                    self.get_logger().warn(f"No grasps found for object {object_ids[i]} at bbox {bbox}")
            
            if not all_grasps:
                # Still store bboxes for visualization even if no grasps
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "No grasps detected in any bounding box",
                    "total_objects": len(bboxes),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No grasps detected")
                return response
            
            self.latest_grasps = all_grasps
            self.latest_region_grasps = []
            
            # Publish grasp poses
            self._publish_grasp_poses(all_grasps)
            
            response.success = True
            response.message = json.dumps({
                "success": True,
                "total_grasps": len(all_grasps),
                "total_objects": len(bboxes),
                "objects_with_grasps": len(set([g['object_id'] for g in all_grasps])),
                "grasps": grasp_data,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"✓ Grasp Detection Complete!")
            self.get_logger().info(f"  Total Objects: {len(bboxes)}")
            self.get_logger().info(f"  Objects with Grasps: {len(set([g['object_id'] for g in all_grasps]))}")
            self.get_logger().info(f"  Total Grasps: {len(all_grasps)}")
            self.get_logger().info("=" * 60)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Grasp detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def detect_grasp_bb_callback(self, request, response):
        """Service callback for /vision/detect_grasp_bb - detect grasp in specific bbox"""
        try:
            if self.captured_rgb is None or self.captured_depth is None:
                response.success = False
                response.grasp_pose = GraspPose()
                response.error_message = "No RGB-D data available. Waiting for camera..."
                self.get_logger().warn("No RGB-D data captured yet")
                return response
            
            bbox = [request.x1, request.y1, request.x2, request.y2]
            
            self.get_logger().info(f"Detecting grasp in bbox: {bbox}")
            
            # Detect grasps in bounding box
            grasps = self._detect_grasps_in_bbox(self.captured_rgb, self.captured_depth, bbox)
            
            if not grasps:
                response.success = False
                response.grasp_pose = GraspPose()
                response.error_message = "No grasps detected in bounding box"
                self.get_logger().warn("No grasps detected in bbox")
                return response
            
            self.latest_region_grasps = grasps
            self.latest_grasps = []
            
            # Get best grasp
            best_grasp = grasps[0]
            
            # Create GraspPose message
            grasp_msg = GraspPose()
            grasp_msg.object_id = "bbox_region"
            grasp_msg.bbox = bbox
            
            pos = best_grasp['position']
            grasp_msg.position.x = float(pos['x'])
            grasp_msg.position.y = float(pos['y'])
            grasp_msg.position.z = float(pos['z'])
            
            ori = best_grasp['orientation']
            grasp_msg.orientation.x = float(ori['x'])
            grasp_msg.orientation.y = float(ori['y'])
            grasp_msg.orientation.z = float(ori['z'])
            grasp_msg.orientation.w = float(ori['w'])
            
            grasp_msg.quality_score = float(best_grasp['quality_score'])
            grasp_msg.width = float(best_grasp['grasp_width'])
            grasp_msg.approach_direction = "top"
            
            # Publish grasp poses
            self._publish_grasp_poses(grasps)
            
            response.success = True
            response.grasp_pose = grasp_msg
            response.error_message = ""
            
            self.get_logger().info(
                f"Bbox grasp detection complete: {len(grasps)} grasps, "
                f"best quality={best_grasp['quality_score']:.2f}"
            )
            
        except Exception as e:
            response.success = False
            response.grasp_pose = GraspPose()
            response.error_message = str(e)
            self.get_logger().error(f"Bbox grasp detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def run_pipeline_callback(self, request, response):
        """Service callback for /vision/run_pipeline - waits for SAM detections"""
        try:
            self.get_logger().info("Pipeline mode activated - waiting for SAM detections on /vision/sam_detections")
            
            response.success = True
            response.message = json.dumps({
                "success": True,
                "message": "Pipeline mode active. GraspNet will process SAM detections automatically.",
                "listening_to": "/vision/sam_detections",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Pipeline activation error: {e}")
        
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
    
    def _detect_grasps_in_bbox(self, rgb_image: np.ndarray, depth_image: np.ndarray, bbox: List[int]) -> List[Dict]:
        """
        Detect grasp poses within a specific bounding box
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            List of grasp dictionaries
        """
        x1, y1, x2, y2 = bbox
        
        # Clamp bbox to image bounds
        h, w = rgb_image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            self.get_logger().warn(f"Skipping invalid bbox: {bbox}")
            return []
        
        # Crop region
        roi_rgb = rgb_image[y1:y2, x1:x2]
        roi_depth = depth_image[y1:y2, x1:x2]
        
        # Run grasp detection on ROI
        if GRASPNET_AVAILABLE:
            grasps = self._graspnet_inference(roi_rgb, roi_depth)
        else:
            grasps = self._geometric_grasp_estimation_bbox(roi_rgb, roi_depth, [x1, y1, x2, y2])
        
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
    
    def _geometric_grasp_estimation_bbox(self, roi_rgb: np.ndarray, roi_depth: np.ndarray, bbox: List[int]) -> List[Dict]:
        """
        Geometric grasp estimation within a bounding box region
        
        Args:
            roi_rgb: RGB region of interest
            roi_depth: Depth region of interest
            bbox: Original bounding box [x1, y1, x2, y2]
            
        Returns:
            List of grasp poses
        """
        x1, y1, x2, y2 = bbox
        h, w = roi_rgb.shape[:2]
        grasps = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate center of bbox
        cx_roi = w // 2
        cy_roi = h // 2
        
        # Get depth at center
        depth_value = roi_depth[cy_roi, cx_roi] if (0 <= cy_roi < roi_depth.shape[0] and 0 <= cx_roi < roi_depth.shape[1]) else 0
        
        if depth_value == 0 or np.isnan(depth_value):
            # Try to get average depth
            roi_size = 5
            y1_d = max(0, cy_roi - roi_size)
            y2_d = min(roi_depth.shape[0], cy_roi + roi_size)
            x1_d = max(0, cx_roi - roi_size)
            x2_d = min(roi_depth.shape[1], cx_roi + roi_size)
            depth_roi = roi_depth[y1_d:y2_d, x1_d:x2_d]
            valid_depths = depth_roi[depth_roi > 0]
            if len(valid_depths) > 0:
                depth_value = np.median(valid_depths)
            else:
                depth_value = 0.5
        
        depth_m = float(depth_value) / 1000.0 if depth_value > 100 else float(depth_value)
        
        # Camera intrinsics
        fx = 525.0
        fy = 525.0
        img_w = self.captured_rgb.shape[1] if self.captured_rgb is not None else 640
        img_h = self.captured_rgb.shape[0] if self.captured_rgb is not None else 480
        cx_cam = img_w / 2.0
        cy_cam = img_h / 2.0
        
        if self.camera_info is not None:
            K = np.array(self.camera_info.k).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx_cam = K[0, 2]
            cy_cam = K[1, 2]
        
        # Generate multiple grasp candidates
        num_grasps = 3
        for i in range(num_grasps):
            # Angle variation
            angle = i * 60.0  # 0, 60, 120 degrees
            angle_rad = np.deg2rad(angle)
            
            # Position in original image coordinates
            cx_global = x1 + cx_roi
            cy_global = y1 + cy_roi
            
            # 3D position in camera frame
            x_3d = (cx_global - cx_cam) * depth_m / fx
            y_3d = (cy_global - cy_cam) * depth_m / fy
            z_3d = depth_m
            
            # Orientation quaternion
            qz = np.sin(angle_rad / 2.0)
            qw = np.cos(angle_rad / 2.0)
            
            # Quality score
            quality = 0.7 - (i * 0.1)  # First grasp has highest quality
            
            # Grasp width estimate
            grasp_width_m = min(w, h) * 0.001  # Convert pixels to meters
            
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
                "pixel_location": [int(cx_global), int(cy_global)],
                "depth_value": float(depth_m)
            }
            
            grasps.append(grasp)
        
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
        
        self.get_logger().info(f"   JSON saved: {json_path}")
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
        self.get_logger().info(f"   Visualization saved: {vis_path}")
        
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
        
        # Use captured frame for visualization
        display_img = self.captured_rgb.copy() if self.captured_rgb is not None else self.latest_rgb.copy()
        
        # Check if we have region grasps (from SAM pipeline or bbox detection)
        if self.latest_region_grasps:
            # Visualize region grasps with bounding boxes
            for grasp in self.latest_region_grasps:
                if 'bbox' in grasp:
                    bbox = grasp['bbox']
                    # Draw bounding box
                    cv2.rectangle(
                        display_img,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 255),  # Yellow
                        2
                    )
                
                px, py = grasp["pixel_location"]
                quality = grasp["quality_score"]
                angle = grasp["approach_angle"]
                
                color = (0, 255, 0)  # Green
                
                # Draw grasp center
                cv2.circle(display_img, (px, py), 5, color, -1)
                cv2.circle(display_img, (px, py), 7, (255, 255, 255), 2)
                
                # Draw grasp orientation
                length = 30
                angle_rad = np.deg2rad(angle)
                end_x = int(px + length * np.cos(angle_rad))
                end_y = int(py + length * np.sin(angle_rad))
                cv2.arrowedLine(display_img, (px, py), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Draw perpendicular line (gripper width)
                perp_angle = angle_rad + np.pi/2
                width = 20
                px1 = int(px + width/2 * np.cos(perp_angle))
                py1 = int(py + width/2 * np.sin(perp_angle))
                px2 = int(px - width/2 * np.cos(perp_angle))
                py2 = int(py - width/2 * np.sin(perp_angle))
                cv2.line(display_img, (px1, py1), (px2, py2), color, 2)
                
                # Draw label
                label = f"Q={quality:.2f}"
                cv2.putText(display_img, label, (px + 10, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add title
            title = f"GraspNet | Region Grasps: {len(self.latest_region_grasps)}"
            cv2.putText(display_img, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Check if we have full image grasps
        elif self.latest_grasps:
            # Group grasps by object_id to show them more clearly
            grasps_by_object = {}
            for grasp in self.latest_grasps:
                obj_id = grasp.get('object_id', 'unknown')
                if obj_id not in grasps_by_object:
                    grasps_by_object[obj_id] = []
                grasps_by_object[obj_id].append(grasp)
            
            colors = [
                (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 255, 0),
                (128, 128, 255), (255, 128, 128)
            ]
            
            # Draw grasps grouped by object (show only best grasp per object for clarity)
            obj_idx = 0
            total_grasps_shown = 0
            for obj_id, obj_grasps in grasps_by_object.items():
                if obj_idx >= len(colors):
                    break
                    
                color = colors[obj_idx]
                
                # Draw bounding box if available
                if obj_grasps and 'bbox' in obj_grasps[0]:
                    bbox = obj_grasps[0]['bbox']
                    cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Draw object label at top of bbox
                    label_obj = f"{obj_id} ({len(obj_grasps)}g)"
                    cv2.putText(display_img, label_obj, (bbox[0], bbox[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show only the best grasp (first one, highest quality) for each object
                grasp = obj_grasps[0]
                px, py = grasp["pixel_location"]
                quality = grasp["quality_score"]
                angle = grasp["approach_angle"]
                
                # Draw grasp center
                cv2.circle(display_img, (px, py), 6, color, -1)
                cv2.circle(display_img, (px, py), 8, (255, 255, 255), 2)
                
                # Draw grasp orientation
                length = 40
                angle_rad = np.deg2rad(angle)
                end_x = int(px + length * np.cos(angle_rad))
                end_y = int(py + length * np.sin(angle_rad))
                cv2.arrowedLine(display_img, (px, py), (end_x, end_y), color, 3, tipLength=0.3)
                
                # Draw perpendicular line (gripper width)
                perp_angle = angle_rad + np.pi/2
                width = 30
                px1 = int(px + width/2 * np.cos(perp_angle))
                py1 = int(py + width/2 * np.sin(perp_angle))
                px2 = int(px - width/2 * np.cos(perp_angle))
                py2 = int(py - width/2 * np.sin(perp_angle))
                cv2.line(display_img, (px1, py1), (px2, py2), color, 2)
                
                # Draw quality label
                label = f"Q={quality:.2f}"
                cv2.putText(display_img, label, (px + 10, py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                
                obj_idx += 1
                total_grasps_shown += 1
            
            # Add title with object count
            title = f"GraspNet | Objects: {len(grasps_by_object)} | Total Grasps: {len(self.latest_grasps)} (showing best per object)"
            cv2.putText(display_img, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # No grasps detected yet
            cv2.putText(display_img, "Call service to detect grasps", (10, 30),
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
        from rclpy.executors import MultiThreadedExecutor
        node = GraspNetDetector()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
