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
- Visualization: 0° arrow = VERTICAL (pointing UP)
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from custom_interfaces.srv import FindObjectAngleBB, FindObjectAngle, DetectObjects
from custom_interfaces.msg import SAMDetections
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import threading


class OBBAngleServiceNode(Node):
    """
    Service node for OBB (Oriented Bounding Box) angle detection and visualization.
    Works similar to SAM detector but provides oriented bounding boxes instead of axis-aligned boxes.
    """

    def __init__(self):
        super().__init__('obb_angle_service_node')
        
        # Keep subscriptions and services in separate callback groups so
        # detection updates can be processed while service callbacks are waiting.
        self.subscription_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()
        
        # Storage for latest detections and images
        self.latest_detections = None
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.bridge = CvBridge()
        
        # Thread lock for thread-safe access
        self.detections_lock = threading.Lock()
        self.detections_condition = threading.Condition(self.detections_lock)
        self.latest_detections_stamp_ns = 0

        # Queue visualization work so service callbacks can return immediately.
        self.viz_lock = threading.Lock()
        self.pending_viz = None
        
        # OpenCV visualization window (unified for both single and multi-object)
        self.window_name = 'OBB Angle Detection'
        
        # Camera topic configuration (adjust based on hardware/simulation)
        self.declare_parameter('real_hardware', True)
        self.real_hardware = self.get_parameter('real_hardware').value
        
        if self.real_hardware:
            self.rgb_topic = '/camera/color/image_raw'
        else:
            self.rgb_topic = '/camera/image_raw'

        # Best-effort QoS is generally more compatible with high-rate detector topics.
        self.sam_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # Subscribe to RGB camera for visualization
        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
            callback_group=self.subscription_callback_group,
        )
        
        # Subscribe to SAM detections for multi-object OBB
        self.sam_subscription = self.create_subscription(
            SAMDetections,
            '/vision/sam_detections',
            self.sam_detections_callback,
            self.sam_qos,
            callback_group=self.subscription_callback_group,
        )
        
        # Subscribe to depth camera
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_rect_raw",
            self.depth_callback,
            10,
            callback_group=self.subscription_callback_group,
        )
        
        # Subscribe to camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.info_callback,
            10,
            callback_group=self.subscription_callback_group,
        )
        
        # Service client for real-time detection
        self.detect_objects_client = self.create_client(
            DetectObjects,
            '/vision/detect_objects',
            callback_group=self.service_callback_group
        )
        
        # Create OBB angle service servers
        self.find_object_angle_bb_srv = self.create_service(
            FindObjectAngleBB,
            '/obb/find_object_angle_bb',
            self.find_object_angle_bb_callback,
            callback_group=self.service_callback_group
        )
        
        self.find_object_angle_srv = self.create_service(
            FindObjectAngle,
            '/obb/find_object_angle',
            self.find_object_angle_callback,
            callback_group=self.service_callback_group
        )
        
        # Create OpenCV window (unified)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        
        # GUI updates are pumped from the main thread in main() to avoid
        # HighGUI freezes under multi-threaded executors.
        self.viz_timer = None
        
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
        self.get_logger().info('  - /vision/sam_detections QoS: BEST_EFFORT, VOLATILE, depth=10')
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
        with self.detections_condition:
            self.latest_detections = msg
            self.latest_detections_stamp_ns = time.monotonic_ns()
            self.detections_condition.notify_all()
            if not hasattr(self, '_sam_received'):
                self._sam_received = True
                self.get_logger().info(f'First SAM detections received: {msg.total_detections} objects')
            self.get_logger().debug(f'Received SAM detections: {msg.total_detections} objects')

    def wait_for_sam_detections(self, timeout_sec=1.5, min_stamp_ns=0):
        """Wait until at least one SAM detection message is available."""
        deadline = time.monotonic() + timeout_sec
        with self.detections_condition:
            while True:
                if self.latest_detections is not None and self.latest_detections_stamp_ns >= min_stamp_ns:
                    return True

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False

                self.detections_condition.wait(timeout=min(0.05, remaining))
    
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
        # Render queued visualization outside service callbacks to avoid
        # blocking service responses on GUI operations.
        pending = None
        with self.viz_lock:
            if self.pending_viz is not None:
                pending = self.pending_viz
                self.pending_viz = None

        if pending is not None:
            results, mode = pending
            try:
                self.visualize_obb(results, mode=mode)
            except Exception as e:
                self.get_logger().warn(f'Visualization update failed: {e}')
        else:
            # Process window events even when no new frame is queued.
            cv2.waitKey(1)

    def queue_visualization(self, results, mode="auto"):
        """Queue the latest visualization payload for timer-based rendering."""
        with self.viz_lock:
            self.pending_viz = (results, mode)
    
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
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for RGB camera...", (80, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
            cv2.putText(blank, self.rgb_topic, (80, 258),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return

        vis_image = self.latest_rgb_image.copy()
        img_h, img_w = vis_image.shape[:2]

        # Auto-detect mode
        if mode == "auto":
            mode = "single" if len(results) == 1 else "multi"

        # Distinct color palette (BGR)
        COLORS = [
            (0,  200, 255),   # amber
            (80, 255,  80),   # lime
            (255,  80,  80),  # blue
            (255,   0, 200),  # magenta
            (0,  230, 230),   # yellow
            (200,  80, 255),  # violet
            (0,  255, 180),   # spring green
            (255, 180,   0),  # sky blue
        ]

        def draw_corner_bracket(img, pts, color, lw=2):
            """Draw corner-bracket accents on an OBB polygon."""
            n = len(pts)
            for i in range(n):
                A = pts[(i - 1) % n].astype(float)
                B = pts[i].astype(float)
                C = pts[(i + 1) % n].astype(float)
                ab = A - B;  ab_len = np.linalg.norm(ab)
                cb = C - B;  cb_len = np.linalg.norm(cb)
                if ab_len == 0 or cb_len == 0:
                    continue
                clen = max(10, int(min(ab_len, cb_len) * 0.22))
                p1 = (B + (ab / ab_len) * clen).astype(int)
                p2 = (B + (cb / cb_len) * clen).astype(int)
                cv2.line(img, tuple(B.astype(int)), tuple(p1), color, lw)
                cv2.line(img, tuple(B.astype(int)), tuple(p2), color, lw)

        def semi_transparent_rect(img, x1, y1, x2, y2, fill, alpha=0.72):
            """Blend a dark rectangle over a sub-region."""
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(img.shape[1] - 1, x2);  y2 = min(img.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                return
            roi = img[y1:y2, x1:x2]
            bg  = np.full_like(roi, fill)
            img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, bg, alpha, 0)

        # ── Process each OBB ─────────────────────────────────────────────────
        for idx, result_tuple in enumerate(results):
            if len(result_tuple) == 7:
                object_id, u, v, theta, width, height, bbox = result_tuple
            else:
                object_id, u, v, theta, width, height = result_tuple
                bbox = None

            color = COLORS[idx % len(COLORS)]
            cx, cy = int(u), int(v)
            angle_geom_deg = np.rad2deg(theta)
            angle_deg = 90.0 - angle_geom_deg

            # ── Input AABB (single mode only) ─────────────────────────────
            if bbox is not None and mode == "single":
                ax1, ay1, ax2, ay2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                # Dashed-style thin rectangle (draw as corner brackets only)
                aabb_pts = np.array([[ax1,ay1],[ax2,ay1],[ax2,ay2],[ax1,ay2]])
                draw_corner_bracket(vis_image, aabb_pts, (80, 200, 80), lw=1)
                cv2.rectangle(vis_image, (ax1, ay1), (ax2, ay2), (80, 200, 80), 1)
                # Small label
                ts, _ = cv2.getTextSize("AABB", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
                semi_transparent_rect(vis_image, ax1, ay1 - ts[1] - 6, ax1 + ts[0] + 4, ay1, (18, 18, 18))
                cv2.putText(vis_image, "AABB", (ax1 + 2, ay1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 200, 80), 1)

            # ── OBB outline ───────────────────────────────────────────────
            box_pts = self.get_obb_corner_points(u, v, theta, width, height)
            # Thin full outline
            cv2.drawContours(vis_image, [box_pts], 0, color, 1)
            # Corner bracket accents
            draw_corner_bracket(vis_image, box_pts, color, lw=2)

            # ── Center dot ────────────────────────────────────────────────
            cv2.circle(vis_image, (cx, cy), 5, (15, 15, 15), -1)   # dark fill
            cv2.circle(vis_image, (cx, cy), 5, color, 2)            # color ring
            cv2.circle(vis_image, (cx, cy), 2, (240, 240, 240), -1) # white center

            # ── Angle arrow ───────────────────────────────────────────────
            visual_theta = theta + np.pi / 2 - np.pi / 2
            arrow_len = int(max(height * 0.55, 20))
            end_x = int(cx + arrow_len * np.cos(visual_theta))
            end_y = int(cy + arrow_len * np.sin(visual_theta))
            cv2.arrowedLine(vis_image, (cx, cy), (end_x, end_y),
                            color, 2, tipLength=0.35)

            # ── Info panel (single) / compact label (multi) ───────────────
            if mode == "single":
                info_lines = [
                    f"{object_id}",
                    f"Center  ({cx}, {cy})",
                    f"Angle   {angle_deg:.1f} deg",
                    f"Size    {width:.0f} x {height:.0f} px",
                ]
                fs, ft = 0.42, 1
                pad = 8
                line_h = 20
                max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0]
                            for l in info_lines)
                panel_w = max_w + pad * 2 + 4   # +4 for accent bar
                panel_h = len(info_lines) * line_h + pad
                px = img_w - panel_w - 10
                py = 28   # sits below top bar

                # Background
                semi_transparent_rect(vis_image, px - 2, py, px + panel_w, py + panel_h, (15, 15, 15))
                # Left accent bar
                cv2.rectangle(vis_image, (px - 2, py), (px + 2, py + panel_h), color, -1)
                # Border
                cv2.rectangle(vis_image, (px - 2, py), (px + panel_w, py + panel_h), color, 1)

                for i, line in enumerate(info_lines):
                    ty = py + pad + i * line_h + line_h // 2
                    # Dim the label for the first line (object id) — draw in color
                    text_color = color if i == 0 else (210, 210, 210)
                    cv2.putText(vis_image, line, (px + 6, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, ft)
            else:
                # Compact floating label near center
                label = f"#{idx}  {angle_deg:.1f}°"
                fs, ft = 0.38, 1
                (lw_px, lh_px), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
                pad = 4
                lx = cx + 12
                ly = cy - 6
                # Clamp
                lx = min(lx, img_w - lw_px - pad * 2 - 5)
                ly = max(ly, lh_px + pad + 2)

                semi_transparent_rect(vis_image, lx - 2, ly - lh_px - pad,
                                      lx + lw_px + pad * 2 + 3, ly + pad, (15, 15, 15))
                cv2.rectangle(vis_image, (lx - 2, ly - lh_px - pad),
                              (lx + 3, ly + pad), color, -1)  # accent bar
                cv2.putText(vis_image, label, (lx + 5, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (230, 230, 230), ft)

        # ── Top info bar ─────────────────────────────────────────────────────
        bar_h = 24
        obj_count = len(results)
        title = f"OBB Detection  |  Objects: {obj_count}  |  0 deg = Vertical"
        semi_transparent_rect(vis_image, 0, 0, img_w, bar_h, (12, 12, 12), alpha=0.78)
        cv2.putText(vis_image, title, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
        
        # Concise logging
        if mode == "single":
            self.get_logger().info(f'Visualization updated: 1 object')
        else:
            self.get_logger().info(f'Visualization updated: {len(results)} objects')
    
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
        Find object within the given AABB and calculate its OBB using mask detection
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
            
            # Trigger real-time detection to get fresh detections
            request_stamp_ns = time.monotonic_ns()
            self.get_logger().info('Triggering real-time detection...')
            detection_success = self.trigger_real_time_detection()
            min_stamp_ns = request_stamp_ns if detection_success else 0

            # Wait for SAM callback instead of fixed sleep.
            got_sam = self.wait_for_sam_detections(timeout_sec=1.5, min_stamp_ns=min_stamp_ns)
            if not got_sam:
                self.get_logger().warn('Timed out waiting for SAM detections update')
            
            # Get latest detections
            with self.detections_lock:
                if self.latest_detections is None:
                    response.success = False
                    response.message = 'No SAM detections available. Ensure simple_sam_detector is running.'
                    response.u = 0.0
                    response.v = 0.0
                    response.theta = 0.0
                    response.width = 0.0
                    response.height = 0.0
                    self.get_logger().warn('No detections available')
                    return response
                
                detections = self.latest_detections
            
            # Find object within the given bbox using IoU (Intersection over Union)
            best_detection = None
            best_iou = 0.0
            
            req_area = (request.x2 - request.x1) * (request.y2 - request.y1)
            
            for detection in detections.detections:
                if len(detection.bbox) < 4:
                    continue
                
                det_x1, det_y1, det_x2, det_y2 = detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
                
                # Calculate intersection
                inter_x1 = max(request.x1, det_x1)
                inter_y1 = max(request.y1, det_y1)
                inter_x2 = min(request.x2, det_x2)
                inter_y2 = min(request.y2, det_y2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                    union_area = req_area + det_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = detection
            
            if best_detection is None:
                response.success = False
                response.message = f'No object found within bbox [{request.x1}, {request.y1}, {request.x2}, {request.y2}]'
                response.u = 0.0
                response.v = 0.0
                response.theta = 0.0
                response.width = 0.0
                response.height = 0.0
                self.get_logger().warn(f'No object detected in the specified region (checked {len(detections.detections)} detections)')
                return response
            
            self.get_logger().info(f'Found object: {best_detection.object_id} with IoU={best_iou:.3f}')
            
            # Get the object's bbox and mask
            x1, y1, x2, y2 = best_detection.bbox[0], best_detection.bbox[1], best_detection.bbox[2], best_detection.bbox[3]
            
            # Get mask if available
            mask = None
            try:
                if best_detection.mask is not None:
                    mask = self.bridge.imgmsg_to_cv2(best_detection.mask, desired_encoding='mono8')
                    self.get_logger().info(f'Using mask for accurate OBB calculation')
            except Exception as e:
                self.get_logger().debug(f'Failed to convert mask: {e}')
            
            # Calculate OBB using the detected object's bbox and mask
            u, v, theta_geom, width, height = self.calculate_obb_from_bbox(x1, y1, x2, y2, mask)

            # Remap angle for result only: angle_out_deg = 90 - angle_deg
            angle_orig_deg = np.rad2deg(theta_geom)
            angle_result_deg = 90.0 - angle_orig_deg
            theta_result = np.deg2rad(angle_result_deg)

            # Populate response (use remapped theta, keep geometry unchanged)
            response.success = True
            response.message = f'OBB calculated for {best_detection.object_id} within bbox [{request.x1}, {request.y1}, {request.x2}, {request.y2}]'
            response.u = u
            response.v = v
            response.theta = theta_result
            response.width = width
            response.height = height
            
            # Log results concisely
            angle_deg = angle_result_deg
            self.get_logger().info('=' * 60)
            self.get_logger().info(f'OBB Result ({best_detection.object_id}): center=({u:.1f},{v:.1f}), angle={angle_deg:.1f}deg, size={width:.0f}x{height:.0f}')
            self.get_logger().info('=' * 60)
            
            # Queue visualization; do not block service response path.
            viz_data = [(best_detection.object_id, u, v, theta_geom, width, height, [request.x1, request.y1, request.x2, request.y2])]
            self.queue_visualization(viz_data, mode="single")
            
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
            request_stamp_ns = time.monotonic_ns()
            self.get_logger().info('Triggering real-time detection for all objects...')
            detection_success = self.trigger_real_time_detection()
            min_stamp_ns = request_stamp_ns if detection_success else 0

            # Wait for SAM callback instead of fixed sleep.
            got_sam = self.wait_for_sam_detections(timeout_sec=1.5, min_stamp_ns=min_stamp_ns)
            if not got_sam:
                self.get_logger().warn('Timed out waiting for SAM detections update')
            
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
                
                # Calculate OBB (geometry angle)
                u, v, theta_geom, width, height = self.calculate_obb_from_bbox(x1, y1, x2, y2, mask)

                # Remap angle for result only: angle_out_deg = 90 - angle_deg
                angle_orig_deg = np.rad2deg(theta_geom)
                angle_result_deg = 90.0 - angle_orig_deg
                theta_result = np.deg2rad(angle_result_deg)

                # Store results (use remapped theta in response arrays)
                object_ids.append(object_id)
                centers_u.append(u)
                centers_v.append(v)
                thetas.append(theta_result)
                widths.append(width)
                heights.append(height)
                bboxes.extend([int(x1), int(y1), int(x2), int(y2)])
                
                # For visualization, keep original geometry angle
                viz_results.append((object_id, u, v, theta_geom, width, height))
                
                angle_deg = angle_result_deg
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
            
            self.get_logger().info('=' * 60)
            self.get_logger().info(f'OBB Result: Processed {len(object_ids)} objects successfully')
            self.get_logger().info('=' * 60)
            
            # Visualize all objects with unified function
            if len(viz_results) > 0:
                self.queue_visualization(viz_results, mode="multi")
            
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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        node.get_logger().info('OBB Angle Service Node spinning...')
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.03)
            node.keep_window_alive()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down OBB Angle Service Node...')
    finally:
        executor.shutdown()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
