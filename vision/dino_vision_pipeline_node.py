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
    from vision.sam_pipeline_adapter import SAMPipelineAdapter
    SAM_AVAILABLE = True
except ImportError:
    try:
        from sam_pipeline_adapter import SAMPipelineAdapter
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
        self.process_service = self.create_service(
            Trigger,
            '/vision/process_scene',
            self.process_scene_callback
        )
        
        self.reset_service = self.create_service(
            Trigger,
            '/vision/reset_pipeline',
            self.reset_pipeline_callback
        )
        
        self.get_logger().info("🔧 Services initialized")
    
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
            
            # Reinitialize pipeline if needed
            if not self.pipeline_ready and SAM_AVAILABLE:
                self._init_sam_pipeline()
            
            response.success = True
            response.message = "Pipeline reset successfully"
            self.get_logger().info("🔄 Pipeline state reset")
            
        except Exception as e:
            response.success = False
            response.message = f"Error resetting pipeline: {e}"
            self.get_logger().error(f"Reset error: {e}")
        
        return response
    
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