#!/usr/bin/env python3
"""
OBB (Oriented Bounding Box) Angle Service Node

Provides OBB angle calculation and visualization similar to SAM detector.
While SAM provides AABB (Axis-Aligned Bounding Box), this service provides
OBB (Oriented Bounding Box) with rotation angle.

Services:
    1. /obb/find_object_angle_bb
       Calculate OBB angle for a specific bounding box with real-time visualization
       Input: x1, y1, x2, y2 (AABB pixel coordinates)
       Output: center (u,v), theta (radians), width, height, success status
       ros2 service call /obb/find_object_angle_bb custom_interfaces/srv/FindObjectAngleBB "{x1: 100, y1: 100, x2: 300, y2: 300}"
    
    2. /obb/find_object_angle
       Calculate OBB angles for ALL detected objects in the scene
       Input: None (automatically detects objects)
       Output: arrays of object_ids, centers, angles, sizes, bboxes
       ros2 service call /obb/find_object_angle custom_interfaces/srv/FindObjectAngle

Key Features:
- Real-time object detection integration
- OpenCV visualization with OBB overlay
- Angle parallel to WIDTH (longer dimension)
- Angle range: -90° to 90° (-π/2 to π/2 radians)
- Multi-threaded executor for non-blocking service calls

Usage:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 run vision obb_angle_service_node
    Terminal 3: ros2 service call /obb/find_object_angle_bb custom_interfaces/srv/FindObjectAngleBB "{x1: 100, y1: 100, x2: 300, y2: 300}"
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import FindObjectAngleBB, FindObjectAngle, DetectObjects
from custom_interfaces.msg import SAMDetections
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import time


class OBBAngleServiceNode(Node):
    """
    Service node for OBB (Oriented Bounding Box) angle detection and visualization.
    Works similar to SAM detector but provides oriented bounding boxes instead of axis-aligned boxes.
    """

    def __init__(self):
        super().__init__('obb_angle_service_node')
        
        # Use reentrant callback group for nested service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Storage for latest detections and images
        self.latest_detections = None
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.bridge = CvBridge()
        
        # Thread lock for thread-safe access
        import threading
        self.detections_lock = threading.Lock()
        
        # OpenCV visualization window (unified for both single and multi-object)
        self.window_name = 'OBB Angle Detection'
        
        # Camera topic configuration (adjust based on hardware/simulation)
        self.declare_parameter('real_hardware', False)
        self.real_hardware = self.get_parameter('real_hardware').value
        
        if self.real_hardware:
            self.rgb_topic = '/camera/color/image_raw'
        else:
            self.rgb_topic = '/camera/image_raw'
        
        # Subscribe to RGB camera for visualization
        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10
        )
        
        # Subscribe to SAM detections for multi-object OBB
        self.sam_subscription = self.create_subscription(
            SAMDetections,
            '/vision/sam_detections',
            self.sam_detections_callback,
            10
        )
        
        # Subscribe to depth camera
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_rect_raw",
            self.depth_callback,
            10
        )
        
        # Subscribe to camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.info_callback,
            10
        )
        
        # Service client for real-time detection
        self.detect_objects_client = self.create_client(
            DetectObjects,
            '/vision/detect_objects',
            callback_group=self.callback_group
        )
        
        # Create OBB angle service servers
        self.find_object_angle_bb_srv = self.create_service(
            FindObjectAngleBB,
            '/obb/find_object_angle_bb',
            self.find_object_angle_bb_callback,
            callback_group=self.callback_group
        )
        
        self.find_object_angle_srv = self.create_service(
            FindObjectAngle,
            '/obb/find_object_angle',
            self.find_object_angle_callback,
            callback_group=self.callback_group
        )
        
        # Create OpenCV window (unified)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        
        # Create timer for continuous window update (keeps window responsive)
        self.viz_timer = self.create_timer(0.1, self.keep_window_alive)
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('OBB Angle Service Node Started')
        self.get_logger().info('=' * 80)
        self.get_logger().info(f'RGB Topic: {self.rgb_topic}')
        self.get_logger().info(f'Real Hardware Mode: {self.real_hardware}')
        self.get_logger().info('Services:')
        self.get_logger().info('  - /obb/find_object_angle_bb (Single object OBB)')
        self.get_logger().info('  - /obb/find_object_angle (All objects OBB)')
        self.get_logger().info('Subscriptions:')
        self.get_logger().info('  - /vision/sam_detections (SAMDetections)')
        self.get_logger().info(f'  - {self.rgb_topic} (Image)')
        self.get_logger().info('  - /camera/depth/image_rect_raw (Image)')
        self.get_logger().info('  - /camera/color/camera_info (CameraInfo)')
        self.get_logger().info('Visualization:')
        self.get_logger().info(f'  - {self.window_name} (Unified OpenCV window)')
        self.get_logger().info('=' * 80)
        self.get_logger().info('Usage:')
        self.get_logger().info('  Single: ros2 service call /obb/find_object_angle_bb custom_interfaces/srv/FindObjectAngleBB "{x1: 100, y1: 100, x2: 300, y2: 300}"')
        self.get_logger().info('  Multi:  ros2 service call /obb/find_object_angle custom_interfaces/srv/FindObjectAngle')
        self.get_logger().info('=' * 80)
    
    def rgb_callback(self, msg: Image):
        """Store latest RGB image for visualization"""
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Log first successful image reception
            if not hasattr(self, '_rgb_received'):
                self._rgb_received = True
                self.get_logger().info(f'First RGB image received: {self.latest_rgb_image.shape}, {self.latest_rgb_image.dtype}')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert RGB image: {e}')

    
    def sam_detections_callback(self, msg: SAMDetections):
        """Store latest SAM detections"""
        with self.detections_lock:
            self.latest_detections = msg
            self.get_logger().debug(f'Received SAM detections: {msg.total_detections} objects')
    
    def depth_callback(self, msg: Image):
        """Store latest depth image"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert depth image: {e}')
    
    def info_callback(self, msg: CameraInfo):
        """Store latest camera info"""
        self.camera_info = msg
    
    def keep_window_alive(self):
        """Timer callback to keep OpenCV window responsive"""
        # This just calls waitKey to process window events
        # Without this, the window may freeze or not display properly
        cv2.waitKey(1)
    
    def calculate_obb_from_bbox(self, x1, y1, x2, y2, mask=None):
        """
        Calculate OBB (Oriented Bounding Box) from AABB coordinates
        
        Args:
            x1, y1, x2, y2: AABB (Axis-Aligned Bounding Box) coordinates
            mask: Optional binary mask for better contour detection
            
        Returns:
            tuple: (u, v, theta, width, height)
                u, v: center coordinates in pixels
                theta: rotation angle in radians (-pi/2 to pi/2)
                       PARALLEL TO WIDTH (longer dimension)
                width: longer dimension of OBB
                height: shorter dimension of OBB
        """
        try:
            # Calculate AABB center as fallback
            u_aabb = float((x1 + x2) / 2.0)
            v_aabb = float((y1 + y2) / 2.0)
            
            # Method 1: Use mask for better OBB calculation
            if mask is not None:
                try:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Calculate minimum area rectangle (OBB)
                        rect = cv2.minAreaRect(largest_contour)
                        center, (obb_width, obb_height), angle_deg = rect
                        
                        # Update center from mask
                        u, v = float(center[0]), float(center[1])
                        
                        # Convert angle to radians
                        theta = np.deg2rad(angle_deg)
                        
                        # Ensure width is longer dimension (angle parallel to width)
                        if obb_height > obb_width:
                            obb_width, obb_height = obb_height, obb_width
                            theta += np.pi / 2
                        
                        # Normalize theta to [-pi/2, pi/2]
                        while theta > np.pi / 2:
                            theta -= np.pi
                        while theta < -np.pi / 2:
                            theta += np.pi
                        
                        return u, v, float(theta), float(obb_width), float(obb_height)
                
                except Exception as e:
                    self.get_logger().debug(f'Mask-based OBB failed: {e}, using AABB method')
            
            # Method 2: Use AABB corners to calculate OBB
            points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            
            rect = cv2.minAreaRect(points)
            center, (obb_width, obb_height), angle_deg = rect
            
            u, v = float(center[0]), float(center[1])
            theta = np.deg2rad(angle_deg)
            
            # Ensure width is longer dimension
            if obb_height > obb_width:
                obb_width, obb_height = obb_height, obb_width
                theta += np.pi / 2
            
            # Normalize theta to [-pi/2, pi/2]
            while theta > np.pi / 2:
                theta -= np.pi
            while theta < -np.pi / 2:
                theta += np.pi
            
            return u, v, float(theta), float(obb_width), float(obb_height)
            
        except Exception as e:
            self.get_logger().error(f'OBB calculation error: {e}')
            # Return AABB center with zero rotation
            u = float((x1 + x2) / 2.0)
            v = float((y1 + y2) / 2.0)
            width = float(x2 - x1)
            height = float(y2 - y1)
            return u, v, 0.0, width, height
    
    def get_obb_corner_points(self, u, v, theta, width, height):
        """
        Calculate the 4 corner points of an OBB
        
        Args:
            u, v: Center coordinates
            theta: Rotation angle in radians
            width, height: OBB dimensions
            
        Returns:
            np.ndarray: 4x2 array of corner points
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        hw = width / 2.0
        hh = height / 2.0
        
        # Local corners (before rotation)
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotation matrix
        rotation = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Rotate and translate
        rotated_corners = corners @ rotation.T
        rotated_corners[:, 0] += u
        rotated_corners[:, 1] += v
        
        return rotated_corners.astype(np.int32)
    
    def visualize_obb(self, results, mode="auto"):
        """
        Unified visualization for single or multiple OBBs
        
        Args:
            results: List of tuples (object_id, u, v, theta, width, height, bbox_optional)
                     For single object: [(id, u, v, theta, w, h, [x1,y1,x2,y2])]
                     For multiple: [(id1, u1, v1, theta1, w1, h1), (id2, ...), ...]
            mode: "single" or "multi" or "auto" (auto-detect based on count)
        """
        if self.latest_rgb_image is None:
            self.get_logger().warn('No RGB image available for visualization')
            # Show a blank placeholder window
            blank = np.zeros((800, 1200, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for RGB camera image...", 
                       (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(blank, f"RGB Topic: {self.rgb_topic}", 
                       (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        vis_image = self.latest_rgb_image.copy()
        
        # Auto-detect mode
        if mode == "auto":
            mode = "single" if len(results) == 1 else "multi"
        
        # Color palette for objects
        colors = [
            (255, 255, 0),   # Cyan
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Yellow
            (255, 128, 0),   # Orange
            (128, 255, 0),   # Lime
            (0, 255, 128),   # Spring Green
            (255, 0, 128),   # Pink
            (128, 0, 255),   # Purple
        ]
        
        # Process each OBB
        for idx, result_tuple in enumerate(results):
            # Unpack (handle optional bbox)
            if len(result_tuple) == 7:
                object_id, u, v, theta, width, height, bbox = result_tuple
            else:
                object_id, u, v, theta, width, height = result_tuple
                bbox = None
            
            color = colors[idx % len(colors)]
            
            # Draw AABB first if provided (for single object mode)
            if bbox is not None and mode == "single":
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 2)
                cv2.putText(vis_image, "AABB (Input)", (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Get OBB corner points
            box_points = self.get_obb_corner_points(u, v, theta, width, height)
            
            # Draw OBB
            cv2.drawContours(vis_image, [box_points], 0, color, 3)
            
            # Draw center point
            if mode == "single":
                cv2.circle(vis_image, (int(u), int(v)), 8, (0, 0, 255), -1)  # Red center
                cv2.circle(vis_image, (int(u), int(v)), 10, (255, 255, 255), 2)  # White outline
            else:
                cv2.circle(vis_image, (int(u), int(v)), 6, (255, 255, 255), -1)
                cv2.circle(vis_image, (int(u), int(v)), 8, color, 2)
            
            # Draw angle arrow
            arrow_length = max(width, height) / 3.0
            end_x = int(u + arrow_length * np.cos(theta))
            end_y = int(v + arrow_length * np.sin(theta))
            arrow_thickness = 3 if mode == "single" else 2
            cv2.arrowedLine(vis_image, (int(u), int(v)), (end_x, end_y), 
                          (255, 0, 255) if mode == "single" else color, 
                          arrow_thickness, tipLength=0.3)
            
            # Draw label
            angle_deg = np.rad2deg(theta)
            
            if mode == "single":
                # Detailed info box for single object
                info_lines = [
                    f"Object: {object_id}",
                    f"Center: ({int(u)}, {int(v)}) px",
                    f"Angle: {angle_deg:.2f}deg",
                    f"Theta: {theta:.4f} rad",
                    f"Size: {width:.1f} x {height:.1f} px",
                    f"Width: {width:.1f} px (angle || to this)",
                    f"Height: {height:.1f} px"
                ]
                
                # Draw info box at top-right
                font_scale = 0.7
                font_thickness = 2
                line_spacing = 30
                
                max_width = 0
                for line in info_lines:
                    (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          font_scale, font_thickness)
                    max_width = max(max_width, text_w)
                
                box_height = len(info_lines) * line_spacing + 20
                box_x = vis_image.shape[1] - max_width - 30
                box_y = 20
                
                # Background
                cv2.rectangle(vis_image, (box_x - 10, box_y - 10),
                            (box_x + max_width + 10, box_y + box_height), (0, 0, 0), -1)
                # Border
                cv2.rectangle(vis_image, (box_x - 10, box_y - 10),
                            (box_x + max_width + 10, box_y + box_height), color, 3)
                
                # Text lines
                for i, line in enumerate(info_lines):
                    y_pos = box_y + (i * line_spacing) + 20
                    cv2.putText(vis_image, line, (box_x, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            else:
                # Compact label for multi-object
                label = f"#{idx} {angle_deg:.1f}deg"
                label_x = int(u + 15)
                label_y = int(v)
                
                # Text with outline
                cv2.putText(vis_image, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
                cv2.putText(vis_image, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add title
        if mode == "single":
            title = "OBB Angle Detection - Single Object"
        else:
            title = f"OBB Angle Detection - {len(results)} Objects"
        
        cv2.putText(vis_image, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(vis_image, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        
        # Add legend at bottom
        legend_y = vis_image.shape[0] - 30
        if mode == "single":
            legend_text = "Angle Range: -90deg to +90deg | Width: Longer dimension (angle parallel to width)"
        else:
            legend_text = f"Total Objects: {len(results)} | Angle Range: -90deg to +90deg | Width: Longer dimension"
        
        cv2.putText(vis_image, legend_text, (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
        
        # Debug logging
        self.get_logger().info(f'Image displayed: shape={vis_image.shape}, dtype={vis_image.dtype}')
        self.get_logger().info(f'Image min={vis_image.min()}, max={vis_image.max()}, mean={vis_image.mean():.1f}')
        
        if mode == "single":
            self.get_logger().info(f'OBB visualization updated: 1 object')
        else:
            self.get_logger().info(f'OBB visualization updated: {len(results)} objects')
    
    def trigger_real_time_detection(self):
        """
        Trigger real-time detection from simple_sam_detector
        Returns True if successful, False otherwise
        """
        if not self.detect_objects_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('detect_objects service not available')
            return False
        
        try:
            detect_request = DetectObjects.Request()
            future = self.detect_objects_client.call_async(detect_request)
            
            # Non-blocking wait with timeout
            start_time = time.time()
            timeout = 5.0
            while not future.done() and (time.time() - start_time) < timeout:
                time.sleep(0.01)
            
            if future.done():
                detect_response = future.result()
                if detect_response and detect_response.success:
                    self.get_logger().info(f'Real-time detection complete: {detect_response.total_detections} objects')
                    return True
                else:
                    self.get_logger().warn('Detection failed')
                    return False
            else:
                self.get_logger().warn('Detection timeout')
                return False
                
        except Exception as e:
            self.get_logger().warn(f'Detection call failed: {e}')
            return False
    
    def find_object_angle_bb_callback(self, request, response):
        """
        Service callback for /obb/find_object_angle_bb
        Calculate OBB for a single bounding box with visualization
        """
        self.get_logger().info('=' * 80)
        self.get_logger().info(f'[/obb/find_object_angle_bb] Request received')
        self.get_logger().info(f'Input AABB: [{request.x1}, {request.y1}, {request.x2}, {request.y2}]')
        self.get_logger().info('=' * 80)
        
        try:
            # Validate input
            if request.x2 <= request.x1 or request.y2 <= request.y1:
                response.success = False
                response.message = 'Invalid bounding box: x2 must be > x1 and y2 must be > y1'
                response.u = 0.0
                response.v = 0.0
                response.theta = 0.0
                response.width = 0.0
                response.height = 0.0
                self.get_logger().error('Invalid bounding box coordinates')
                return response
            
            # Trigger real-time detection for latest image
            self.get_logger().info('Triggering real-time detection...')
            detection_success = self.trigger_real_time_detection()
            
            if detection_success:
                self.get_logger().info('Detection successful, calculating OBB...')
            else:
                self.get_logger().info('Detection skipped, proceeding with OBB calculation...')
            
            # Calculate OBB from AABB
            u, v, theta, width, height = self.calculate_obb_from_bbox(
                request.x1, request.y1, request.x2, request.y2
            )
            
            # Populate response
            response.success = True
            response.message = f'OBB calculated successfully for bbox [{request.x1}, {request.y1}, {request.x2}, {request.y2}]'
            response.u = u
            response.v = v
            response.theta = theta
            response.width = width
            response.height = height
            
            # Log results
            angle_deg = np.rad2deg(theta)
            self.get_logger().info('=' * 80)
            self.get_logger().info('[OBB RESULT]')
            self.get_logger().info(f'  Center (u,v): ({u:.2f}, {v:.2f}) pixels')
            self.get_logger().info(f'  Angle: {angle_deg:.2f} degrees')
            self.get_logger().info(f'  Theta: {theta:.4f} radians')
            self.get_logger().info(f'  Width: {width:.2f} pixels (angle parallel to this)')
            self.get_logger().info(f'  Height: {height:.2f} pixels')
            self.get_logger().info('=' * 80)
            
            # Visualize with unified function
            viz_data = [("Input_BBox", u, v, theta, width, height, [request.x1, request.y1, request.x2, request.y2])]
            self.visualize_obb(viz_data, mode="single")
            
        except Exception as e:
            self.get_logger().error(f'Error in find_object_angle_bb: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f'Internal error: {str(e)}'
            response.u = 0.0
            response.v = 0.0
            response.theta = 0.0
            response.width = 0.0
            response.height = 0.0
        
        return response
    
    def find_object_angle_callback(self, request, response):
        """
        Service callback for /obb/find_object_angle
        Calculate OBB for all detected objects with visualization
        """
        self.get_logger().info('=' * 80)
        self.get_logger().info('[/obb/find_object_angle] Request received for all objects')
        self.get_logger().info('=' * 80)
        
        try:
            # Trigger real-time detection
            self.get_logger().info('Triggering real-time detection for all objects...')
            detection_success = self.trigger_real_time_detection()
            
            # Small delay to ensure detections are updated
            time.sleep(0.1)
            
            # Get latest detections
            with self.detections_lock:
                if self.latest_detections is None:
                    response.success = False
                    response.message = 'No SAM detections available. Ensure simple_sam_detector is running.'
                    response.total_objects = 0
                    response.object_ids = []
                    response.centers_u = []
                    response.centers_v = []
                    response.thetas = []
                    response.widths = []
                    response.heights = []
                    response.bboxes = []
                    self.get_logger().warn('No detections available')
                    return response
                
                detections = self.latest_detections
            
            total_objects = detections.total_detections
            self.get_logger().info(f'Processing {total_objects} SAM detections for OBB calculation')
            
            if total_objects == 0:
                response.success = True
                response.message = 'No objects detected in scene'
                response.total_objects = 0
                response.object_ids = []
                response.centers_u = []
                response.centers_v = []
                response.thetas = []
                response.widths = []
                response.heights = []
                response.bboxes = []
                self.get_logger().info('Scene is empty - no objects to process')
                return response
            
            # Process each detection
            object_ids = []
            centers_u = []
            centers_v = []
            thetas = []
            widths = []
            heights = []
            bboxes = []
            viz_results = []
            
            for detection in detections.detections:
                object_id = detection.object_id
                
                # Validate bbox
                if len(detection.bbox) < 4:
                    self.get_logger().warn(f'Invalid bbox for {object_id}, skipping')
                    continue
                
                x1, y1, x2, y2 = detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
                
                # Get mask if available
                mask = None
                try:
                    if detection.mask is not None:
                        mask = self.bridge.imgmsg_to_cv2(detection.mask, desired_encoding='mono8')
                except Exception as e:
                    self.get_logger().debug(f'Failed to convert mask for {object_id}: {e}')
                
                # Calculate OBB
                u, v, theta, width, height = self.calculate_obb_from_bbox(x1, y1, x2, y2, mask)
                
                # Store results
                object_ids.append(object_id)
                centers_u.append(u)
                centers_v.append(v)
                thetas.append(theta)
                widths.append(width)
                heights.append(height)
                bboxes.extend([int(x1), int(y1), int(x2), int(y2)])
                
                # For visualization
                viz_results.append((object_id, u, v, theta, width, height))
                
                angle_deg = np.rad2deg(theta)
                self.get_logger().info(
                    f'  {object_id}: center=({u:.1f},{v:.1f}), '
                    f'angle={angle_deg:.1f}deg, size=({width:.1f}x{height:.1f})'
                )
            
            # Build response
            response.success = True
            response.message = f'Successfully calculated OBB for {len(object_ids)} objects'
            response.total_objects = len(object_ids)
            response.object_ids = object_ids
            response.centers_u = centers_u
            response.centers_v = centers_v
            response.thetas = thetas
            response.widths = widths
            response.heights = heights
            response.bboxes = bboxes
            
            self.get_logger().info('=' * 80)
            self.get_logger().info(f'[OBB RESULT] Processed {len(object_ids)} objects successfully')
            self.get_logger().info('=' * 80)
            
            # Visualize all objects with unified function
            if len(viz_results) > 0:
                self.visualize_obb(viz_results, mode="multi")
            
        except Exception as e:
            self.get_logger().error(f'Error in find_object_angle: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f'Internal error: {str(e)}'
            response.total_objects = 0
            response.object_ids = []
            response.centers_u = []
            response.centers_v = []
            response.thetas = []
            response.widths = []
            response.heights = []
            response.bboxes = []
        
        return response
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = OBBAngleServiceNode()
    
    # Use MultiThreadedExecutor for non-blocking service calls
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info('OBB Angle Service Node spinning...')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down OBB Angle Service Node...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
