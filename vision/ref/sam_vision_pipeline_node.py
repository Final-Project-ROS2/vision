#!/usr/bin/env python3
"""
ROS2 SAM Vision Pipeline Node
Integrates the 4-stage SAM pipeline (SAM/CLIP/GraspNet/Scene Understanding) with ROS2
for robotic vision in Gazebo simulation and real robot applications.

Pipeline stages:
1. SAM (Segment Anything) Object Detection & Segmentation
2. CLIP Semantic Tagging  
3. GraspNet 6D Grasp Prediction
4. Scene Graph Construction & Understanding

Author: ROS2 Vision Pipeline Team
License: Apache-2.0
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
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add SAM pipeline to path
current_dir = Path(__file__).parent
final_proj_dir = current_dir.parent / "Final-proj"
sam_pipeline_dir = final_proj_dir / "src" / "pipeline"
sys.path.insert(0, str(final_proj_dir))
sys.path.insert(0, str(sam_pipeline_dir))

# Import custom message types (we'll create these)
try:
    from vision_msgs.msg import (
        DetectionResult, 
        GraspPose, 
        SceneGraph,
        SemanticObject
    )
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False
    print("⚠️ Custom vision messages not available. Using standard messages.")

# Import SAM pipeline components
try:
    from vision.ref.sam_pipeline_adapter import SAMPipelineAdapter
    SAM_AVAILABLE = True
except ImportError:
    try:
        from vision.ref.sam_pipeline_adapter import SAMPipelineAdapter
        SAM_AVAILABLE = True
    except ImportError:
        SAM_AVAILABLE = False
        print("⚠️ SAM Pipeline Adapter not available. Running in basic mode.")


class ROS2SAMVisionPipeline(Node):
    """
    ROS2 node for SAM-based robotic vision pipeline
    
    Subscribes to:
        - /camera/image_raw (RGB images)
        - /camera/depth/image_raw (Depth images) 
        - /camera/camera_info (Camera parameters)
    
    Publishes:
        - /vision/detections (Object detections)
        - /vision/grasps (Grasp poses)
        - /vision/scene_graph (Scene understanding)
        - /vision/debug_image (Visualization)
    
    Services:
        - /vision/process_scene (Trigger full pipeline)
        - /vision/detect_objects (Run object detection only)
        - /vision/classify_objects (Run semantic classification)
        - /vision/generate_grasps (Generate 6D grasp poses)
        - /vision/get_positions (Get 3D object positions)
        - /vision/build_scene_graph (Build scene graph with relations)
        - /vision/reset_pipeline (Reset internal state)
    """
    
    def __init__(self):
        super().__init__('sam_vision_pipeline')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Pipeline state
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.pipeline_ready = False
        
        # Cached pipeline results for individual service calls
        self.cached_detections = None
        self.cached_classifications = None
        self.cached_grasps = None
        self.cached_positions = None
        self.cached_scene_graph = None
        self.last_processed_time = None
        
        # Output directory for saving results
        self.output_base = Path.home() / "ros2_vision_outputs"
        self.output_base.mkdir(exist_ok=True)
        
        # QoS profiles for different data types
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
        
        # Initialize SAM pipeline
        self._init_sam_pipeline()
        
        # Set up ROS2 interfaces
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()
        
        # Parameters
        self.declare_parameter('auto_process', False)
        self.declare_parameter('save_results', True)
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('processing_rate', 1.0)  # Hz
        
        # Auto-processing timer
        if self.get_parameter('auto_process').get_parameter_value().bool_value:
            timer_period = 1.0 / self.get_parameter('processing_rate').get_parameter_value().double_value
            self.processing_timer = self.create_timer(timer_period, self.auto_process_callback)
        
        self.get_logger().info("🚀 ROS2 SAM Vision Pipeline Node Started!")
        self.get_logger().info(f"   SAM Pipeline Available: {SAM_AVAILABLE}")
        self.get_logger().info(f"   Custom Messages Available: {CUSTOM_MSGS_AVAILABLE}")
        self.get_logger().info(f"   Output Directory: {self.output_base}")
    
    def _init_sam_pipeline(self):
        """Initialize the SAM vision pipeline"""
        try:
            if SAM_AVAILABLE:
                self.get_logger().info("🔧 Initializing SAM Pipeline...")
                self.sam_pipeline = SAMPipelineAdapter(device="auto")
                self.pipeline_ready = True
                self.get_logger().info("✅ SAM Pipeline initialized successfully!")
            else:
                self.get_logger().warn("⚠️ SAM Pipeline not available - running in simulation mode")
                self.sam_pipeline = None
                self.pipeline_ready = False
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize SAM Pipeline: {e}")
            self.sam_pipeline = None
            self.pipeline_ready = False
    
    def _setup_subscribers(self):
        """Set up ROS2 subscribers"""
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
        
        self.get_logger().info("📡 Subscribers initialized")
    
    def _setup_publishers(self):
        """Set up ROS2 publishers"""
        # Debug visualization
        self.debug_image_pub = self.create_publisher(
            Image,
            '/vision/debug_image',
            self.detection_qos
        )
        
        # Detection results (using standard messages for now)
        self.detection_pub = self.create_publisher(
            Image,  # Will publish annotated image for now
            '/vision/detection_result',
            self.detection_qos
        )
        
        # Grasp poses
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/vision/grasp_poses',
            self.detection_qos
        )
        
        self.get_logger().info("📤 Publishers initialized")
    
    def _setup_services(self):
        """Set up ROS2 services"""
        # Main pipeline service
        self.process_service = self.create_service(
            Trigger,
            '/vision/process_scene',
            self.process_scene_callback
        )
        
        # Individual component services
        self.detection_service = self.create_service(
            Trigger,
            '/vision/detect_objects',
            self.detect_objects_callback
        )
        
        self.classification_service = self.create_service(
            Trigger,
            '/vision/classify_objects',
            self.classify_objects_callback
        )
        
        self.grasp_service = self.create_service(
            Trigger,
            '/vision/generate_grasps',
            self.generate_grasps_callback
        )
        
        self.position_service = self.create_service(
            Trigger,
            '/vision/get_positions',
            self.get_positions_callback
        )
        
        self.scene_graph_service = self.create_service(
            Trigger,
            '/vision/build_scene_graph',
            self.build_scene_graph_callback
        )
        
        # Utility services
        self.reset_service = self.create_service(
            Trigger,
            '/vision/reset_pipeline',
            self.reset_pipeline_callback
        )
        
        self.get_logger().info("Services initialized: process_scene, detect_objects, classify_objects, generate_grasps, get_positions, build_scene_graph, reset_pipeline")
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image messages"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.get_logger().debug("📸 RGB image received")
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
    
    def depth_callback(self, msg: Image):
        """Handle depth image messages"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # self.get_logger().debug("🗺️ Depth image received")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
        # self.get_logger().debug("📷 Camera info received")
    
    def auto_process_callback(self):
        """Automatic processing timer callback"""
        if self.latest_rgb is not None and self.latest_depth is not None:
            self.process_current_scene()
    
    def process_scene_callback(self, request, response):
        """Service callback to process current scene"""
        try:
            if self.latest_rgb is None or self.latest_depth is None:
                response.success = False
                response.message = "No RGB-D data available"
                return response
            
            success = self.process_current_scene()
            response.success = success
            response.message = "Scene processed successfully" if success else "Scene processing failed"
            
        except Exception as e:
            response.success = False
            response.message = f"Error processing scene: {e}"
            self.get_logger().error(f"Service error: {e}")
        
        return response
    
    def reset_pipeline_callback(self, request, response):
        """Service callback to reset pipeline state"""
        try:
            self.latest_rgb = None
            self.latest_depth = None
            self.camera_info = None
            
            # Clear cached results
            self.cached_detections = None
            self.cached_classifications = None
            self.cached_grasps = None
            self.cached_positions = None
            self.cached_scene_graph = None
            self.last_processed_time = None
            
            # Reinitialize pipeline if needed
            if not self.pipeline_ready and SAM_AVAILABLE:
                self._init_sam_pipeline()
            
            response.success = True
            response.message = "Pipeline reset successfully"
            self.get_logger().info("Pipeline state reset")
            
        except Exception as e:
            response.success = False
            response.message = f"Error resetting pipeline: {e}"
            self.get_logger().error(f"Reset error: {e}")
        
        return response
    
    def detect_objects_callback(self, request, response):
        """Service callback for object detection only"""
        try:
            if self.latest_rgb is None:
                response.success = False
                response.message = "No RGB image available"
                return response
            
            self.get_logger().info("Running object detection...")
            
            # Run detection using SAM pipeline adapter (will use OpenCV-based detection in simulation mode)
            from datetime import datetime
            
            if self.sam_pipeline is not None:
                # Use SAM pipeline adapter for real detection
                results = self.sam_pipeline.process_rgbd(
                    self.latest_rgb,
                    self.latest_depth,
                    output_dir=None
                )
                detections = results.get("detections", [])
                masks = [d.get("mask", np.zeros(self.latest_rgb.shape[:2], dtype=np.uint8)) for d in detections]
                self.get_logger().info(f" SAM adapter detected {len(detections)} objects from real image")
            else:
                # Fallback: Use OpenCV contour detection directly
                self.get_logger().info("🔍 Using OpenCV contour detection (SAM adapter not available)")
                detections, masks = self._opencv_detect_objects(self.latest_rgb)
                self.get_logger().info(f"OpenCV detected {len(detections)} objects from real image")
            
            self.cached_detections = {"detections": detections, "masks": masks, "timestamp": datetime.now()}
            
            response.success = True
            if len(detections) > 0:
                response.message = f"Detected {len(detections)} real objects from camera image"
            else:
                response.message = "No objects detected in camera image"
            self.get_logger().info(f"Detection complete: {len(detections)} objects found")
            
        except Exception as e:
            response.success = False
            response.message = f"Detection error: {e}"
            self.get_logger().error(f"Detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def classify_objects_callback(self, request, response):
        """Service callback for object classification"""
        try:
            if self.cached_detections is None:
                response.success = False
                response.message = "No detections available. Run /vision/detect_objects first"
                return response
            
            self.get_logger().info("Running object classification...")
            
            # Run classification stage - use simulation mode for now
            from datetime import datetime
            detections = self.cached_detections["detections"]
            
            # Simulate semantic classification
            semantic_objects = []
            class_names = ["tool", "part", "container", "object"]
            for i, det in enumerate(detections):
                obj = {
                    "id": det["id"],
                    "class": class_names[i % len(class_names)],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "center": [(det["bbox"][0] + det["bbox"][2])//2, (det["bbox"][1] + det["bbox"][3])//2],
                    "attributes": ["manipulable", "rigid"]
                }
                semantic_objects.append(obj)
            
            self.cached_classifications = {"objects": semantic_objects, "timestamp": datetime.now()}
            
            response.success = True
            response.message = f"Classified {len(semantic_objects)} objects (simulation mode)"
            self.get_logger().info(f"Classification complete: {len(semantic_objects)} objects")
            
        except Exception as e:
            response.success = False
            response.message = f"Classification error: {e}"
            self.get_logger().error(f"Classification error: {e}")
        
        return response
    
    def generate_grasps_callback(self, request, response):
        """Service callback for grasp pose generation"""
        try:
            if self.cached_classifications is None:
                response.success = False
                response.message = "No classifications available. Run /vision/classify_objects first"
                return response
            
            if self.latest_depth is None:
                response.success = False
                response.message = "No depth data available"
                return response
            
            self.get_logger().info("Generating grasp poses...")
            
            # Run grasp generation stage - use simulation mode for now
            from datetime import datetime
            semantic_objects = self.cached_classifications["objects"]
            
            # Simulate grasp generation
            grasps = []
            for obj in semantic_objects:
                center = obj["center"]
                # Convert pixel coordinates to normalized coordinates
                x = center[0] / 1000.0
                y = center[1] / 1000.0
                z = 0.5  # Default depth
                
                grasp = {
                    "object_id": obj["id"],
                    "pose": {
                        "position": [x, y, z],
                        "orientation": [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
                    },
                    "quality": 0.8,
                    "width": 0.05
                }
                grasps.append(grasp)
            
            self.cached_grasps = {"grasps": grasps, "timestamp": datetime.now()}
            
            # Publish grasp poses
            for grasp in grasps:
                grasp_msg = PoseStamped()
                grasp_msg.header.stamp = self.get_clock().now().to_msg()
                grasp_msg.header.frame_id = "camera_link"
                
                pos = grasp["pose"]["position"]
                ori = grasp["pose"]["orientation"]
                
                grasp_msg.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                grasp_msg.pose.orientation = Quaternion(x=float(ori[0]), y=float(ori[1]), z=float(ori[2]), w=float(ori[3]))
                
                self.grasp_pub.publish(grasp_msg)
            
            response.success = True
            response.message = f"Generated {len(grasps)} grasp poses (simulation mode)"
            self.get_logger().info(f"Grasp generation complete: {len(grasps)} poses")
            
        except Exception as e:
            response.success = False
            response.message = f"Grasp generation error: {e}"
            self.get_logger().error(f"Grasp generation error: {e}")
        
        return response
    
    def get_positions_callback(self, request, response):
        """Service callback to get object positions"""
        try:
            if self.cached_classifications is None:
                response.success = False
                response.message = "No classifications available. Run /vision/classify_objects first"
                return response
            
            self.get_logger().info("Extracting object positions...")
            
            # Extract positions from classified objects
            semantic_objects = self.cached_classifications["objects"]
            positions = []
            
            for obj in semantic_objects:
                bbox = obj.get("bbox", [0, 0, 0, 0])
                center = obj.get("center", [(bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2])
                
                # Get depth if available
                depth_val = 0.5  # Default
                if self.latest_depth is not None and len(center) == 2:
                    y, x = int(center[1]), int(center[0])
                    if 0 <= y < self.latest_depth.shape[0] and 0 <= x < self.latest_depth.shape[1]:
                        depth_val = float(self.latest_depth[y, x]) / 1000.0  # Convert to meters
                
                position = {
                    "object_id": obj.get("id", "unknown"),
                    "class": obj.get("class", "object"),
                    "position": {
                        "x": float(center[0]) / 1000.0,
                        "y": float(center[1]) / 1000.0,
                        "z": depth_val
                    },
                    "bbox": bbox,
                    "confidence": obj.get("confidence", 0.0)
                }
                positions.append(position)
            
            from datetime import datetime
            self.cached_positions = {"positions": positions, "timestamp": datetime.now()}
            
            response.success = True
            response.message = f"Retrieved {len(positions)} object positions"
            self.get_logger().info(f"Position extraction complete: {len(positions)} positions")
            
        except Exception as e:
            response.success = False
            response.message = f"Position extraction error: {e}"
            self.get_logger().error(f"Position extraction error: {e}")
        
        return response
    
    def build_scene_graph_callback(self, request, response):
        """Service callback to build scene graph"""
        try:
            if self.cached_classifications is None:
                response.success = False
                response.message = "No classifications available. Run /vision/classify_objects first"
                return response
            
            if self.cached_grasps is None:
                # Try to generate grasps if not available
                if self.latest_depth is not None:
                    grasp_req = Trigger.Request()
                    grasp_resp = Trigger.Response()
                    self.generate_grasps_callback(grasp_req, grasp_resp)
                else:
                    self.cached_grasps = {"grasps": [], "timestamp": None}
            
            self.get_logger().info("Building scene graph...")
            
            # Build scene graph - use simulation mode for now
            from datetime import datetime
            semantic_objects = self.cached_classifications["objects"]
            grasps = self.cached_grasps["grasps"]
            
            # Simulate scene graph construction
            scene_graph = {
                "objects": [obj["class"] for obj in semantic_objects],
                "relations": [],
                "spatial_layout": "table_top"
            }
            
            # Add simple spatial relations
            for i, obj1 in enumerate(semantic_objects):
                for j, obj2 in enumerate(semantic_objects):
                    if i < j:
                        c1 = obj1["center"]
                        c2 = obj2["center"]
                        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if dist < 200:
                            scene_graph["relations"].append({
                                "subject": obj1["class"],
                                "predicate": "near",
                                "object": obj2["class"]
                            })
            
            self.cached_scene_graph = {"scene_graph": scene_graph, "timestamp": datetime.now()}
            
            response.success = True
            response.message = f"Scene graph built with {len(scene_graph.get('objects', []))} objects and {len(scene_graph.get('relations', []))} relations (simulation mode)"
            self.get_logger().info("Scene graph construction complete")
            
        except Exception as e:
            response.success = False
            response.message = f"Scene graph error: {e}"
            self.get_logger().error(f"Scene graph error: {e}")
        
        return response
    
    def _opencv_detect_objects(self, rgb_image: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Detect objects using OpenCV contour detection (fallback when SAM not available)
        This provides REAL detection from the actual camera image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better object separation
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        masks = []
        h, w = rgb_image.shape[:2]
        min_area = (w * h) * 0.001  # Minimum 0.1% of image area
        max_area = (w * h) * 0.8    # Maximum 80% of image area
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter small boxes
            if w_box < 20 or h_box < 20:
                continue
            
            # Create binary mask for this object
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            confidence = min(0.95, 0.60 + circularity * 0.35)  # Higher confidence for more circular objects
            
            detection = {
                "id": f"obj_{i}",
                "class": "object",  # Will be classified by CLIP if available
                "confidence": float(confidence),
                "bbox": [x, y, x + w_box, y + h_box],
                "center": [x + w_box // 2, y + h_box // 2],
                "area": int(area),
                "mask": mask
            }
            
            detections.append(detection)
            masks.append(mask)
        
        # If no objects detected, log it clearly
        if not detections:
            self.get_logger().warn("⚠️ No objects detected in image - scene may be empty or thresholds need adjustment")
        
        return detections, masks
    
    def process_current_scene(self) -> bool:
        """Process the current RGB-D scene through DINO pipeline"""
        try:
            if not self.pipeline_ready or self.latest_rgb is None or self.latest_depth is None:
                return False
            
            self.get_logger().info("🔄 Processing scene through SAM pipeline...")
            
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scene_output_dir = self.output_base / f"scene_{timestamp}"
            scene_output_dir.mkdir(exist_ok=True)
            
            # Save input images for debugging
            if self.get_parameter('save_results').get_parameter_value().bool_value:
                cv2.imwrite(str(scene_output_dir / "input_rgb.jpg"), self.latest_rgb)
                cv2.imwrite(str(scene_output_dir / "input_depth.png"), self.latest_depth)
            
            if SAM_AVAILABLE and self.sam_pipeline is not None:
                # Run full SAM pipeline
                results = self._run_sam_pipeline(scene_output_dir)
            else:
                # Simulation mode
                results = self._simulate_pipeline_results(scene_output_dir)
            
            # Publish results
            self._publish_results(results, timestamp)
            
            self.get_logger().info(f"✅ Scene processing completed - results saved to {scene_output_dir}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Scene processing failed: {e}")
            return False
    
    def _run_sam_pipeline(self, output_dir: Path) -> Dict:
        """Run the actual SAM pipeline"""
        # Process through SAM pipeline adapter
        results = self.sam_pipeline.process_rgbd(
            self.latest_rgb,
            self.latest_depth,
            str(output_dir)
        )
        
        return results
    
    def _simulate_pipeline_results(self, output_dir: Path) -> Dict:
        """Simulate pipeline results when DINO is not available"""
        self.get_logger().info("🎭 Running in simulation mode")
        
        # Create mock results
        results = {
            "detections": [
                {"class": "tool", "confidence": 0.85, "bbox": [100, 100, 200, 200]},
                {"class": "object", "confidence": 0.75, "bbox": [300, 150, 400, 250]}
            ],
            "grasps": [
                {"position": [0.5, 0.3, 0.2], "orientation": [0, 0, 0, 1], "quality": 0.8}
            ],
            "scene_graph": {
                "objects": ["tool", "object"],
                "relations": [["tool", "near", "object"]]
            }
        }
        
        # Create visualization
        debug_image = self.latest_rgb.copy()
        for det in results["detections"]:
            bbox = det["bbox"]
            cv2.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(debug_image, f"{det['class']}: {det['confidence']:.2f}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        results["debug_image"] = debug_image
        
        # Save simulation results
        if self.get_parameter('save_results').get_parameter_value().bool_value:
            cv2.imwrite(str(output_dir / "simulation_result.jpg"), debug_image)
            with open(output_dir / "simulation_results.json", 'w') as f:
                json.dump({k: v for k, v in results.items() if k != "debug_image"}, f, indent=2)
        
        return results
    
    def _publish_results(self, results: Dict, timestamp: str):
        """Publish pipeline results to ROS2 topics"""
        try:
            # Publish debug visualization
            if "debug_image" in results and self.get_parameter('debug_visualization').get_parameter_value().bool_value:
                debug_msg = self.bridge.cv2_to_imgmsg(results["debug_image"], encoding="bgr8")
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.header.frame_id = "camera_link"
                self.debug_image_pub.publish(debug_msg)
            
            # Publish grasp poses
            if "grasps" in results:
                for grasp in results["grasps"]:
                    grasp_msg = PoseStamped()
                    grasp_msg.header.stamp = self.get_clock().now().to_msg()
                    grasp_msg.header.frame_id = "camera_link"
                    
                    grasp_msg.pose.position = Point(
                        x=float(grasp["position"][0]),
                        y=float(grasp["position"][1]), 
                        z=float(grasp["position"][2])
                    )
                    grasp_msg.pose.orientation = Quaternion(
                        x=float(grasp["orientation"][0]),
                        y=float(grasp["orientation"][1]),
                        z=float(grasp["orientation"][2]),
                        w=float(grasp["orientation"][3])
                    )
                    
                    self.grasp_pub.publish(grasp_msg)
            
            self.get_logger().debug(f"📤 Results published for timestamp {timestamp}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish results: {e}")


def main(args=None):
    """Main entry point for ROS2 SAM Vision Pipeline node"""
    rclpy.init(args=args)
    
    try:
        node = ROS2SAMVisionPipeline()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Node error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()