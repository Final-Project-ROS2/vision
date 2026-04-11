#!/usr/bin/env python3
"""
CLIP Vision Classifier Node

Services:
    1. /vision/classify_all
       Classify entire camera image
       ros2 service call /vision/classify_all std_srvs/srv/Trigger
    
    2. /vision/classify_bb
       Classify specific bounding box region [x1,y1,x2,y2]
       ros2 service call /vision/classify_bb custom_interfaces/srv/ClassifyBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"

    3. /vision/run_pipeline
       Trigger the entire pipeline
       ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
       Subscribes to /vision/sam_detections (SAMDetections message)
       Automatically classifies each detected bounding box when SAM publishes
       No service call needed - automatic when SAM publishes
       ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
    
    4. /vision/find_object
       Find bounding box by label name
       ros2 service call /vision/find_object custom_interfaces/srv/FindObject "{label: 'piston_rod'}"
    
    5. /vision/classify_bbox_filtered
       Get filtered classifications (confidence > 0.5) from latest SAM detections
       ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger
       Note: Must call /vision/run_pipeline first to trigger SAM detection

Setup:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 run vision clip_classifier
    
Workflow:
    # Step 1: Run SAM detection (auto-classifies with CLIP via subscription)
    ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
    
    # Step 2: Get filtered results (only objects with confidence > 0.5)
    ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time

# Import custom interfaces
try:
    from custom_interfaces.srv import ClassifyBBox, FindMultiObject
    from custom_interfaces.msg import SAMDetections, SAMDetection, CLIPClassification
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    CUSTOM_INTERFACES_AVAILABLE = False
    print("Custom interfaces not available. Build custom_interfaces package first.")
    # Fallback imports
    try:
        from vision.msg import SAMDetections, SAMDetection
        SAM_MSGS_AVAILABLE = True
    except ImportError:
        SAM_MSGS_AVAILABLE = False

# Try to import CLIP (OpenAI official implementation)
try:
    import torch
    from PIL import Image as PILImage
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")

# SAM messages availability (already checked in custom_interfaces import above)
SAM_MSGS_AVAILABLE = CUSTOM_INTERFACES_AVAILABLE


class CLIPClassifier(Node):
    """
    CLIP-based image classifier for ROS2
    
    Subscribes to:
        - /camera/image_raw (RGB images from Gazebo camera)
        - /vision/sam_detections (SAMDetections for auto-classification)
    
    Services:
        - /vision/classify_all (Classify entire image, returns labels and confidence only)
        - /vision/classify_bb (Classify specific bounding box region)
    
    Display:
        - Shows live camera feed with top prediction in OpenCV window
        - Shows classified regions with labels from SAM detections
    """
    
    def __init__(self, candidate_labels: List[str] = None):
        super().__init__('clip_classifier')
        
        # Create reentrant callback group for nested service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Parameter toggles camera topics between simulation and hardware
        self.declare_parameter('real_hardware', False)
        self.real_hardware = bool(self.get_parameter('real_hardware').value)

        # Minimum raw cosine similarity to even consider a region a candidate.
        # Applied before inter-region softmax normalisation; regions below this
        # threshold are discarded entirely.  Default 0.1 rejects strongly
        # mismatched regions while keeping borderline ones in the softmax pool.
        self.declare_parameter('clip_min_confidence', 0.1)
        self.clip_min_confidence = float(self.get_parameter('clip_min_confidence').value)

        self.rgb_topic = '/camera/color/image_raw' if self.real_hardware else '/camera/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw' if self.real_hardware else '/camera/depth/image_raw'
        self.camera_info_topic = 'camera/color/camera_info' if self.real_hardware else '/camera/camera_info'
        self.desired_encoding = 'passthrough' if self.real_hardware else 'bgr8'

        # Default labels if none provided
        self.candidate_labels = candidate_labels or [
            # "cobot",
            "green_cube",
            "drill",
            "pink_cube",
            "measuring_tape",
            "screwdriver",
            # "gear",
            # "monkey_wrench",
            # "piston_rod",
            # "washer",
            # "cross_joint_part",
            # "white_ball",
            # "door_handle",
            # "red_ball",
            # "gasket_part",
            "beer_can",
            "bowl",
            "cinder_block",
            "coke_can",
            "roomba",
            # "plastic_cup",
            # "hammer",
            # "robotic_arm",
            # "white_robotic_hand",
            # "drone",
            # "t_brace_part",
            # "brown_box",
            # "u_joint_part",
            # "tube",
            # "steak_set",
            # "mug",
            # "tool_belt",
            # "magnifying_glass",
            # "ramekin",
            # "cracker_box",
            # "basket",
            # "square_plate",
            # "birthday_cake",
            # "plum",
            # "meat_can",
            # "lemon",
            # "salad_on_plate",
            # "mobile_robot",
            # "ink_cartridge",
            # "toilet_paper",
            # "bagel_with_cheese",
            # "shoe",
            # "boardgame",
            # "piston_rod_part"
        ]
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest image from camera
        self.latest_rgb = None
        self.captured_frame = None  # Single captured frame for classification
        self.latest_classification = None
        self.latest_region_classifications = []  # For SAM auto-classification
        self.latest_found_object = None  # For find_object service visualization
        self.frame_counter = 0
        self.frame_captured = False
        
        # CLIP model
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # QoS profile for image subscription
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize CLIP model
        self._init_clip_model()
        
        # Subscribe to camera
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            self.image_qos
        )
        
        # Classification services
        self.classification_all_service = self.create_service(
            Trigger,
            '/vision/classify_all',
            self.classify_all_callback,
            callback_group=self.callback_group
        )
        
        if CUSTOM_INTERFACES_AVAILABLE:
            self.classification_bb_service = self.create_service(
                ClassifyBBox,
                '/vision/classify_bb',
                self.classify_bb_callback,
                callback_group=self.callback_group
            )
        else:
            self.classification_bb_service = self.create_service(
                Trigger,
                '/vision/classify_bb',
                self.classify_bb_callback_fallback,
                callback_group=self.callback_group
            )
        
        # Find object by label service
        if CUSTOM_INTERFACES_AVAILABLE:
            try:
                from custom_interfaces.srv import FindObject
                self.find_object_service = self.create_service(
                    FindObject,
                    '/vision/find_object',
                    self.find_object_callback,
                    callback_group=self.callback_group
                )
                self.get_logger().info("Service created: /vision/find_object")
            except ImportError:
                self.get_logger().warn("FindObject service not available. Add to custom_interfaces.")

            self.find_multi_object_service = self.create_service(
                    FindMultiObject,
                    '/vision/find_multi_object',
                    self.find_multi_object_callback,
                    callback_group=self.callback_group
            )
            self.get_logger().info("Service created: /vision/find_multi_object (FindMultiObject)")
        
        # Classify all SAM bboxes with confidence filter service
        self.classify_filtered_service = self.create_service(
            Trigger,
            '/vision/classify_bbox_filtered',
            self.classify_bbox_filtered_callback,
            callback_group=self.callback_group
        )
        
        # Subscribe to live SAM detections for auto-classification
        if SAM_MSGS_AVAILABLE:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                10
            )
            self.get_logger().info("Subscribing to: /vision/sam_detections (SAMDetections)")
        else:
            # Fallback: placeholder Image publisher used by simple_sam_detector before custom msgs are built
            self.sam_sub = self.create_subscription(
                Image,
                '/vision/sam_detections',
                self.sam_detections_placeholder_callback,
                10
            )
            self.get_logger().warn("Subscribing to: /vision/sam_detections (placeholder Image). Build msgs for full integration.")
        
        # OpenCV window setup
        self.window_name = f"CLIP Classifier - {self.rgb_topic}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # GUI updates are pumped from the main thread in main() to avoid
        # HighGUI freezes under multi-threaded executors.
        self.viz_timer = None
        
        self.get_logger().info("CLIP Classifier Started")
        self.get_logger().info(f"Subscribing to: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic (unused currently): {self.depth_topic}")
        self.get_logger().info(f"Camera info topic (unused currently): {self.camera_info_topic}")
        self.get_logger().info(f"real_hardware parameter: {self.real_hardware}")
        self.get_logger().info(f"Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"Labels: {', '.join(self.candidate_labels)}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Service: /vision/classify_all")
        self.get_logger().info(f"Service: /vision/classify_bb")
        self.get_logger().info(f"Service: /vision/classify_bbox_filtered")
        self.get_logger().info(f"Service: /vision/find_object")
        self.get_logger().info("Service: /vision/find_multi_object")
        self.get_logger().info(f"Subscriber: /vision/sam_detections (auto-classify on SAM publish)")
        self.get_logger().info(f"OpenCV Window: '{self.window_name}'")
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("CLIP not available! Install: pip install git+https://github.com/openai/CLIP.git")
            return
        
        try:
            self.get_logger().info("Loading CLIP model...")
            model_name = "ViT-B/32"  # OpenAI CLIP model name
            
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.get_logger().info(f"CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.preprocess = None
    
    def rgb_callback(self, msg: Image):
        """Handle incoming RGB images from configured RGB topic"""
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.desired_encoding)
            self.frame_counter += 1
            
            # Always update captured frame to get fresh data for classification
            # (Fixed: was only capturing first frame, now updates continuously)
            self.captured_frame = self.latest_rgb.copy()
            if not self.frame_captured:
                self.frame_captured = True
                self.get_logger().info(f"First frame captured from {self.rgb_topic}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def classify_all_callback(self, request, response):
        """Service callback for /vision/classify_all - classify entire image"""
        
        import time
        start = time.perf_counter()
        time.sleep(0.01)
        # Use captured frame instead of latest_rgb for consistency
        frame_to_use = self.captured_frame if self.frame_captured else self.latest_rgb

        try:
            if self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "single_clip",
                    "success": False,
                    "error": f"No frame captured yet from {self.rgb_topic}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No frame captured yet")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "single_clip",
                    "success": False,
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("CLIP model not available")
                return response
            
            self.get_logger().info("Running CLIP classification on captured frame...")
            
            # Run classification on captured frame
            classification_data = self._classify_image(self.captured_frame)
            self.latest_classification = classification_data
            self.latest_region_classifications = []  # Clear region classifications
            
            response.success = True
            response.message = json.dumps(classification_data, indent=2)
            
            top_pred = classification_data['output']['top_prediction']
            self.get_logger().info(
                f"Classification complete: {top_pred['label']} "
                f"(confidence: {top_pred['confidence']:.2f})"
            )
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "pipeline": "single_clip",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Classification error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
        import traceback
        self.get_logger().error(traceback.format_exc())
        end = time.perf_counter()
        latency = end - start
        self.get_logger().info(f"Total detection latency: {latency:.6f} seconds")   
        
        return response
    
    def classify_bb_callback(self, request, response):
        """Service callback for /vision/classify_bb with custom ClassifyBBox service"""
        try:
            if self.captured_frame is None:
                response.success = False
                response.label = ""
                response.confidence = 0.0
                response.all_predictions = json.dumps({
                    "error": f"No frame captured yet from {self.rgb_topic}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                self.get_logger().warn("No frame captured yet")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.label = ""
                response.confidence = 0.0
                response.all_predictions = json.dumps({
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                self.get_logger().error("CLIP model not available")
                return response
            
            # Extract bbox from request
            bbox = [request.x1, request.y1, request.x2, request.y2]
            
            self.get_logger().info(f"Classifying bounding box region: {bbox}")
            
            # Classify single region
            classification_data = self._classify_regions(self.captured_frame, [bbox])
            region = classification_data['output']['classified_regions'][0]
            
            self.latest_region_classifications = [region]
            self.latest_classification = None
            
            response.success = True
            response.label = region['top_prediction']['label']
            response.confidence = float(region['top_prediction']['confidence'])
            response.all_predictions = json.dumps(region['all_predictions'])
            
            self.get_logger().info(
                f"Region classification complete: {response.label} "
                f"(confidence: {response.confidence:.2f})"
            )
            
        except Exception as e:
            response.success = False
            response.label = ""
            response.confidence = 0.0
            response.all_predictions = json.dumps({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            self.get_logger().error(f"Bounding box classification error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def classify_bb_callback_fallback(self, request, response):
        """Fallback service callback when ClassifyBBox not available (uses Trigger with center bbox)"""
        try:
            if self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "error": f"No frame captured yet from {self.rgb_topic}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No frame captured yet")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = json.dumps({
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("CLIP model not available")
                return response
            
            # Use center 50% of image as example bbox
            h, w = self.captured_frame.shape[:2]
            bbox = [w//4, h//4, 3*w//4, 3*h//4]
            
            self.get_logger().info(f"Classifying center bbox region: {bbox}")
            
            # Classify single region
            classification_data = self._classify_regions(self.captured_frame, [bbox])
            region = classification_data['output']['classified_regions'][0]
            
            self.latest_region_classifications = [region]
            self.latest_classification = None
            
            response.success = True
            response.message = json.dumps({
                "bbox": bbox,
                "top_prediction": region['top_prediction'],
                "all_predictions": region['all_predictions']
            }, indent=2)
            
            top_pred = region['top_prediction']
            self.get_logger().info(
                f"Region classification complete: {top_pred['label']} "
                f"(confidence: {top_pred['confidence']:.2f})"
            )
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Bounding box classification error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response

    def classify_bbox_filtered_callback(self, request, response):
        """
        Service callback for /vision/classify_bbox_filtered - filter classified regions by confidence > 0.5
        Uses: std_srvs/srv/Trigger
        Returns: JSON with filtered classifications from latest SAM detections
        
        NOTE: This service returns classifications from the latest SAM detection.
        Make sure to call 'ros2 service call /vision/run_pipeline std_srvs/srv/Trigger' first
        to trigger SAM detection and automatic CLIP classification.
        """
        try:
            if self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "error": f"No frame captured yet from {self.rgb_topic}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No frame captured yet")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = json.dumps({
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("CLIP model not available")
                return response
            
            # Check if we have classified regions from SAM subscription
            if not self.latest_region_classifications:
                response.success = False
                response.message = json.dumps({
                    "error": "No classified regions available. Call '/vision/run_pipeline' first to trigger SAM detection.",
                    "hint": "ros2 service call /vision/run_pipeline std_srvs/srv/Trigger",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No classified regions. Run SAM pipeline first.")
                return response
            
            self.get_logger().info(f"Filtering {len(self.latest_region_classifications)} classified regions by confidence > 0.5")
            
            # Filter regions by confidence >= 0.5
            filtered_regions = []
            for region in self.latest_region_classifications:
                top_pred = region['top_prediction']
                if top_pred['confidence'] >= 0.5:
                    filtered_regions.append({
                        'region_id': region['region_id'],
                        'bbox': region['bbox'],
                        'label': top_pred['label'],
                        'confidence': top_pred['confidence']
                    })
            
            # Build response
            result = {
                "pipeline": "sam_clip_filtered",
                "total_sam_regions": len(self.latest_region_classifications),
                "filtered_regions": len(filtered_regions),
                "regions": filtered_regions,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "confidence_threshold": 0.5
                }
            }
            
            response.success = True
            response.message = json.dumps(result, indent=2)
            
            self.get_logger().info(
                f"Filtered classification complete: {len(filtered_regions)}/{len(self.latest_region_classifications)} "
                f"regions passed confidence threshold"
            )
            
            # Log each filtered region
            for region in filtered_regions:
                self.get_logger().info(
                    f"Region #{region['region_id']}: {region['label']} "
                    f"(confidence: {region['confidence']:.2f})"
                )
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Filtered classification error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response

    def sam_detections_callback(self, msg: 'SAMDetections'):
        """Handle incoming SAM detections and automatically classify each region"""
        try:
            if not CLIP_AVAILABLE or self.model is None:
                self.get_logger().warn("CLIP model not available, ignoring SAM detections")
                return
            
            if self.captured_frame is None:
                self.get_logger().warn(f"No captured frame, waiting for {self.rgb_topic}")
                return

            # Extract bboxes from SAMDetections message
            bboxes: List[List[int]] = []
            for det in msg.detections:
                bbox = list(det.bbox)
                if len(bbox) == 4:
                    bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

            if not bboxes:
                self.get_logger().warn("SAM detections message has no valid bounding boxes")
                return

            self.get_logger().info(f"Received {len(bboxes)} SAM regions, classifying with CLIP...")
            
            # Classify all detected regions
            classification_data = self._classify_regions(self.captured_frame, bboxes)
            self.latest_region_classifications = classification_data['output']['classified_regions']
            self.latest_classification = None
            
            # Log results
            for region in self.latest_region_classifications:
                top_pred = region['top_prediction']
                self.get_logger().info(
                    f"Region {region['region_id']}: {top_pred['label']} "
                    f"(confidence: {top_pred['confidence']:.2f})"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error handling SAM detections: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def sam_detections_placeholder_callback(self, msg: Image):
        """Handle placeholder SAM detections (Image) before custom messages are built"""
        try:
            self.get_logger().info(
                f"Placeholder SAM detections received: count={msg.height}, frame={msg.width}. "
                "Build custom messages to enable auto-classification."
            )
        except Exception as e:
            self.get_logger().error(f"Error handling placeholder SAM detections: {e}")

    def _set_response_common(self, response, success: bool, message: str):
        """Set common response fields when available."""
        if hasattr(response, 'success'):
            response.success = success
        if hasattr(response, 'message'):
            response.message = message

    def _call_detect_objects(self) -> Tuple[List[List[int]], Optional[str]]:
        """Call /vision/detect_objects and return bboxes or error message."""
        try:
            from custom_interfaces.srv import DetectObjects
        except ImportError:
            return [], 'DetectObjects service interface not available'

        detect_client = self.create_client(
            DetectObjects,
            '/vision/detect_objects',
            callback_group=self.callback_group
        )

        if not detect_client.wait_for_service(timeout_sec=5.0):
            return [], "Detection service '/vision/detect_objects' not available"

        detect_request = DetectObjects.Request()
        detect_response = detect_client.call(detect_request)

        if detect_response is None:
            return [], 'Detection service returned no response'
        if not detect_response.success:
            return [], f"Object detection failed: {detect_response.error_message}"

        bboxes: List[List[int]] = []
        total = int(detect_response.total_detections)
        for index in range(total):
            bboxes.append([
                int(detect_response.bbox_x1[index]),
                int(detect_response.bbox_y1[index]),
                int(detect_response.bbox_x2[index]),
                int(detect_response.bbox_y2[index]),
            ])

        return bboxes, None

    def _compute_similarity(self, image_bgr: np.ndarray, label: str) -> float:
        """Compute raw CLIP cosine similarity between a single crop and label text.

        Returns a value roughly in [-1, 1].  Prefer
        _compute_inter_region_confidences() when you have multiple regions,
        as that method normalises across regions via softmax to produce proper
        probabilities in [0, 1].
        """
        region_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(region_rgb)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([f"a photo of a {label}"]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T)[0, 0].item()

        return float(similarity)

    def _compute_inter_region_confidences(
        self,
        crops_bgr: List[np.ndarray],
        label: str,
    ) -> List[float]:
        """Return per-region confidence values in [0, 1] for a query label.

        Rather than returning raw cosine similarity (which lives in roughly
        [0.15, 0.35] for typical queries and is hard to threshold), this method
        applies a temperature-scaled softmax *across all candidate regions*.
        The result is a proper probability distribution: the region most likely
        containing ``label`` gets a high score, the rest share the remainder.

        Args:
            crops_bgr: List of BGR image crops (one per detected region).
            label:     Text query, e.g. ``"bowl"`` or ``"green_cube"``.

        Returns:
            List of float probabilities, same length as ``crops_bgr``,
            summing to 1.0.  An empty list is returned when ``crops_bgr``
            is empty.
        """
        if not crops_bgr:
            return []

        # Use the same prompt template as _classify_regions / _classify_image
        text_tokens = clip.tokenize([f"a photo of a {label}"]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features_list = []
            for crop_bgr in crops_bgr:
                region_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(region_rgb)
                image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                img_feat = self.model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                image_features_list.append(img_feat)

            # Stack → (N, D), then cosine similarities → (N,)
            all_image_features = torch.cat(image_features_list, dim=0)  # (N, D)
            raw_sims = (all_image_features @ text_features.T).squeeze(1)  # (N,)

        # Temperature-scaled softmax over regions: a factor of 100 matches the
        # scale used in standard CLIP logits and keeps the distribution sharp.
        probs = (raw_sims * 100.0).softmax(dim=0)
        return [float(p.item()) for p in probs]

    def find_multi_object_callback(self, request, response):
        """
        Service callback for /vision/find_multi_object.

        Custom Interface Definition (custom_interfaces/srv/FindMultiObject.srv):
        # Request
        string label
        int32 k
        ---
        # Response
        bool success
        string message
        int32[] bboxes        # Flattened as [x1,y1,x2,y2, x1,y1,x2,y2, ...]
        float32[] confidences # Similarity score for each returned bbox
        int32 total_matches
        """
        try:
            target_label = request.label.strip()
            top_k = int(request.k)

            if not target_label:
                response.success = False
                response.message = 'Empty label provided'
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            if top_k <= 0:
                response.success = False
                response.message = 'Invalid k: must be >= 1'
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            if self.captured_frame is None:
                response.success = False
                response.message = f'No frame captured yet from {self.rgb_topic}'
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = 'CLIP model not available'
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            self.get_logger().info(
                f"Finding top-{top_k} regions for '{target_label}' by CLIP similarity"
            )

            bboxes, error_message = self._call_detect_objects()
            if error_message:
                response.success = False
                response.message = error_message
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            if not bboxes:
                payload = {
                    'success': True,
                    'label': target_label,
                    'requested_k': top_k,
                    'total_regions': 0,
                    'total_matches': 0,
                    'matches': [],
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                }
                response.success = True
                response.message = json.dumps(payload, indent=2)
                response.bboxes = []
                response.confidences = []
                response.total_matches = 0
                return response

            frame = self.captured_frame
            frame_h, frame_w = frame.shape[:2]

            # --- Pass 1: collect valid crops pre-filtered by raw cosine ---
            candidates: List[Dict] = []
            crops: List[np.ndarray] = []
            for region_id, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, frame_w))
                x2 = max(0, min(x2, frame_w))
                y1 = max(0, min(y1, frame_h))
                y2 = max(0, min(y2, frame_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                region_bgr = frame[y1:y2, x1:x2]
                raw_sim = self._compute_similarity(region_bgr, target_label)
                if raw_sim < self.clip_min_confidence:
                    continue
                candidates.append({
                    'region_id': region_id,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                })
                crops.append(region_bgr)

            matches: List[Dict] = []
            if candidates:
                # --- Pass 2: inter-region softmax → proper [0,1] probabilities ---
                probs = self._compute_inter_region_confidences(crops, target_label)
                for i, cand in enumerate(candidates):
                    matches.append({
                        'region_id': cand['region_id'],
                        'bbox': cand['bbox'],
                        'confidence': round(probs[i], 4),
                    })

            matches.sort(key=lambda item: item['confidence'], reverse=True)
            top_matches = matches[:top_k]
            top_matches.sort(key=lambda item: item['bbox'][0])

            payload = {
                'success': True,
                'label': target_label,
                'requested_k': top_k,
                'total_regions': len(bboxes),
                'total_matches': len(top_matches),
                'matches': top_matches,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
            }

            response.success = True
            response.message = json.dumps(payload, indent=2)
            flattened = []
            for match in top_matches:
                flattened.extend(match['bbox'])
            response.bboxes = flattened
            response.confidences = [float(match['confidence']) for match in top_matches]
            response.total_matches = len(top_matches)

            self.get_logger().info(
                f"Found top {len(top_matches)} matches for '{target_label}' out of {len(bboxes)} regions"
            )

        except Exception as e:
            response.success = False
            response.message = f'Error: {str(e)}'
            response.bboxes = []
            response.confidences = []
            response.total_matches = 0
            self.get_logger().error(f"Find multi object error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

        return response
    
    def find_object_callback(self, request, response):
        """
        Service callback for /vision/find_object - find bounding box by label using CLIP embeddings
        
        This service:
        1. Takes a label name as input (can be ANY text, not limited to candidate_labels)
        2. Gets text embedding for the label using CLIP
        3. Gets image embeddings for all SAM bounding boxes
        4. Computes cosine similarity between label and each bbox
        5. Returns the bounding box with highest similarity score
        
        Custom Interface Definition (add to custom_interfaces/srv/FindObject.srv):
        # Request
        string label
        ---
        # Response
        bool success
        string message
        int32[] bbox  # [x1, y1, x2, y2]
        float32 confidence
        """
        try:
            target_label = request.label.strip()
            
            if not target_label:
                response.success = False
                response.message = "Empty label provided"
                response.bbox = []
                response.confidence = 0.0
                return response
            
            if self.captured_frame is None:
                response.success = False
                response.message = f"No frame captured yet from {self.rgb_topic}"
                response.bbox = []
                response.confidence = 0.0
                self.get_logger().warn("No captured frame available")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = "CLIP model not available"
                response.bbox = []
                response.confidence = 0.0
                self.get_logger().error("CLIP model not available")
                return response
            
            # FIX: Clear cached classifications to force fresh detection every time
            # This ensures find -> move -> find gets updated scene data
            self.get_logger().info("Clearing cached region classifications to force fresh detection")
            self.latest_region_classifications = []
            
            # Check if we have classified regions from SAM, if not, get detections
            if not self.latest_region_classifications:
                self.get_logger().info("No SAM detections available. Calling /vision/detect_objects automatically...")
                
                # Create a client to call the detect_objects service
                try:
                    from custom_interfaces.srv import DetectObjects
                except ImportError:
                    response.success = False
                    response.message = "DetectObjects service interface not available"
                    response.bbox = []
                    response.confidence = 0.0
                    self.get_logger().error("DetectObjects interface not found")
                    return response
                
                detect_client = self.create_client(
                    DetectObjects, 
                    '/vision/detect_objects',
                    callback_group=self.callback_group
                )
                
                # Wait for service to be available (timeout 5 seconds)
                if not detect_client.wait_for_service(timeout_sec=5.0):
                    response.success = False
                    response.message = "Detection service '/vision/detect_objects' not available"
                    response.bbox = []
                    response.confidence = 0.0
                    self.get_logger().error("Detection service not available")
                    return response
                
                # Call the detection service synchronously
                detect_request = DetectObjects.Request()
                detect_response = detect_client.call(detect_request)
                if not detect_response.success:
                    response.success = False
                    response.message = f"Object detection failed: {detect_response.error_message}"
                    response.bbox = []
                    response.confidence = 0.0
                    self.get_logger().error(f"Detection failed: {detect_response.error_message}")
                    return response
                
                self.get_logger().info(f"Detection completed: {detect_response.total_detections} objects found")
                
                # Build bounding boxes from detection response
                bboxes = []
                for i in range(detect_response.total_detections):
                    bbox = [
                        detect_response.bbox_x1[i],
                        detect_response.bbox_y1[i],
                        detect_response.bbox_x2[i],
                        detect_response.bbox_y2[i]
                    ]
                    bboxes.append(bbox)
                
                if not bboxes:
                    response.success = False
                    response.message = "No objects detected in current frame"
                    response.bbox = []
                    response.confidence = 0.0
                    self.get_logger().warn("No detections found")
                    return response
                
                # Classify all detected regions using CLIP
                self.get_logger().info(f"Classifying {len(bboxes)} detected regions...")
                classification_data = self._classify_regions(self.captured_frame, bboxes)
                self.latest_region_classifications = classification_data['output']['classified_regions']
                self.latest_classification = None
                
                self.get_logger().info(f"Classification complete for {len(self.latest_region_classifications)} regions")
            
            self.get_logger().info(f"Computing CLIP confidences for '{target_label}' across {len(self.latest_region_classifications)} regions")

            # --- Pass 1: collect valid crops and their metadata ---------
            h, w = self.captured_frame.shape[:2]
            valid_regions = []
            crops = []
            for region in self.latest_region_classifications:
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    self.get_logger().warn(f"Skipping invalid bbox: {bbox}")
                    continue
                # Pre-filter by raw cosine similarity so obviously wrong regions
                # don't pollute the softmax pool.
                raw_sim = self._compute_similarity(
                    self.captured_frame[y1:y2, x1:x2], target_label
                )
                if raw_sim < self.clip_min_confidence:
                    self.get_logger().debug(
                        f"Region {region['region_id']} discarded: raw_sim={raw_sim:.4f} "
                        f"< clip_min_confidence={self.clip_min_confidence}"
                    )
                    continue
                valid_regions.append({
                    'bbox': [x1, y1, x2, y2],
                    'region_id': region['region_id'],
                })
                crops.append(self.captured_frame[y1:y2, x1:x2])

            # Check if any valid region survived the pre-filter
            if not valid_regions:
                response.success = False
                response.message = f"No valid regions found to compare with '{target_label}'"
                response.bbox = []
                response.confidence = 0.0
                self.latest_found_object = None
                self.get_logger().info("No valid regions to compare")
                return response

            # --- Pass 2: inter-region softmax → proper probabilities ----
            # Each crop gets a probability in [0, 1] that sums to 1 across
            # all regions.  A high score means "this region most likely
            # contains the queried label", which is a meaningful confidence.
            probs = self._compute_inter_region_confidences(crops, target_label)

            for i, region_info in enumerate(valid_regions):
                region_info['confidence'] = probs[i]
                self.get_logger().debug(
                    f"Region {region_info['region_id']}: confidence={probs[i]:.4f}"
                )

            # Best region = highest softmax probability
            best_match = max(valid_regions, key=lambda r: r['confidence'])

            # Minimum confidence guard (now in softmax probability space).
            # With clip_min_confidence already filtering raw similarities,
            # this catches the degenerate single-region case where probability
            # is trivially 1.0 but the raw match was borderline.
            min_prob_threshold = self.clip_min_confidence
            if best_match['confidence'] < min_prob_threshold:
                response.success = False
                response.message = (
                    f"Label '{target_label}' found but confidence too low "
                    f"({best_match['confidence']:.3f} < {min_prob_threshold})"
                )
                response.bbox = []
                response.confidence = float(best_match['confidence'])
                self.latest_found_object = None
                self.get_logger().info(
                    f"Label '{target_label}' confidence too low: {best_match['confidence']:.3f}"
                )
                return response

            # Store for visualization
            self.latest_found_object = {
                'label': target_label,
                'bbox': best_match['bbox'],
                'confidence': best_match['confidence'],
                'region_id': best_match['region_id']
            }

            # Return success with bbox
            response.success = True
            response.message = f"Found '{target_label}' with confidence {best_match['confidence']:.3f}"
            response.bbox = best_match['bbox']
            response.confidence = float(best_match['confidence'])
            # Note: FindObject.srv doesn't have object_id field (only FindObjectReal.srv does)

            self.get_logger().info(
                f"Found '{target_label}': bbox={best_match['bbox']}, "
                f"confidence={best_match['confidence']:.3f}, region_id={best_match['region_id']}"
            )
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            response.bbox = []
            response.confidence = 0.0
            self.latest_found_object = None
            self.get_logger().error(f"Find object error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def _classify_image(self, rgb_image: np.ndarray) -> Dict:
        """
        Classify image using CLIP model
        
        Args:
            rgb_image: BGR image from OpenCV
            
        Returns:
            Dictionary matching the CLIP JSON schema
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        # Prompt template improves accuracy over bare label names (CLIP paper)
        prompted_labels = [f"a photo of a {lbl}" for lbl in self.candidate_labels]

        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompted_labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (100.0 * image_features @ text_features.T)
            probs = logits_per_image.softmax(dim=-1)[0]
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(probs_np)[::-1]
        
        # Build predictions list
        all_predictions = []
        for idx in sorted_indices:
            all_predictions.append({
                "label": self.candidate_labels[idx],
                "confidence": round(float(probs_np[idx]), 2)
            })
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build JSON schema (without image vectors)
        schema = {
            "pipeline": "single_clip",
            "model": "openai/clip-vit-base-patch32",
            "input": {
                "image_path": f"frame_{self.frame_counter:06d}",
                "candidate_labels": self.candidate_labels
            },
            "output": {
                "top_prediction": {
                    "label": all_predictions[0]["label"],
                    "confidence": all_predictions[0]["confidence"]
                },
                "all_predictions": all_predictions,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "processing_time_ms": processing_time_ms,
                    "device": self.device
                }
            }
        }
        
        return schema
    
    def _classify_regions(self, rgb_image: np.ndarray, bboxes: List[List[int]]) -> Dict:
        """
        Classify multiple image regions using CLIP model
        
        Args:
            rgb_image: BGR image from OpenCV
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Dictionary with classification results for each region
        """
        start_time = time.time()
        
        classified_regions = []

        # Encode text once for all regions.  The candidate labels never change
        # within a call, so there is no reason to re-tokenize per region.
        # Prompt template "a photo of a {label}" matches CLIP's training
        # distribution far better than bare label names and measurably improves
        # zero-shot accuracy (CLIP paper: +~13 pp on ImageNet).
        prompted_labels = [f"a photo of a {lbl}" for lbl in self.candidate_labels]
        with torch.no_grad():
            text_tokens = clip.tokenize(prompted_labels).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        h, w = rgb_image.shape[:2]

        for region_id, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox

            # Clamp bbox to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                self.get_logger().warn(f"Skipping invalid bbox: {bbox}")
                continue

            # Crop region
            region_bgr = rgb_image[y1:y2, x1:x2]
            region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(region_rgb)

            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Intra-label softmax: probability over the closed label set.
                # Range [0, 1], sums to 1 across labels for this region.
                logits_per_image = (100.0 * image_features @ text_features.T)
                probs = logits_per_image.softmax(dim=-1)[0]

            probs_np = probs.cpu().numpy()
            sorted_indices = np.argsort(probs_np)[::-1]

            all_predictions = []
            for idx in sorted_indices:
                all_predictions.append({
                    "label": self.candidate_labels[idx],
                    "confidence": round(float(probs_np[idx]), 2)
                })

            region_result = {
                "region_id": region_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "top_prediction": {
                    "label": all_predictions[0]["label"],
                    "confidence": all_predictions[0]["confidence"]
                },
                "all_predictions": all_predictions[:10]
            }

            classified_regions.append(region_result)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build JSON schema
        schema = {
            "pipeline": "clip_with_detection",
            "model": "openai/clip-vit-base-patch32",
            "input": {
                "image_path": f"frame_{self.frame_counter:06d}",
                "num_regions": len(bboxes),
                "candidate_labels": self.candidate_labels
            },
            "output": {
                "classified_regions": classified_regions,
                "summary": {
                    "total_regions": len(classified_regions),
                    "processing_time_ms": processing_time_ms
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "device": self.device
                }
            }
        }
        
        return schema
    
    def visualization_callback(self):
        """Display camera feed with classification in OpenCV window"""
        frame_to_display = self.latest_rgb if self.latest_rgb is not None else self.captured_frame

        if frame_to_display is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Waiting for {self.rgb_topic}...", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return

        vis_image = frame_to_display.copy()
        h, w = vis_image.shape[:2]

        # ── Helpers ──────────────────────────────────────────────────────────
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
        FOUND_COLOR = (60, 230, 60)   # bright green for "found" object

        def semi_rect(img, x1, y1, x2, y2, fill=(15, 15, 15), alpha=0.75):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1:
                return
            roi = img[y1:y2, x1:x2]
            img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1-alpha, np.full_like(roi, fill), alpha, 0)

        def corner_bracket(img, x1, y1, x2, y2, color, lw=2):
            clen = max(10, int(min(x2-x1, y2-y1) * 0.15))
            for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(img, (px, py), (px + dx*clen, py), color, lw)
                cv2.line(img, (px, py), (px, py + dy*clen), color, lw)

        def draw_label(img, text, x, y, color, fs=0.36, ft=1, pad=4, accent=3):
            """Draw a semi-transparent dark label with a color accent bar."""
            (lw_px, lh_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            bx1, by1 = x, y - lh_px - pad
            bx2, by2 = x + accent + pad + lw_px + pad, y + pad
            semi_rect(img, bx1, by1, bx2, by2)
            cv2.rectangle(img, (max(0,bx1), max(0,by1)), (max(0,bx1)+accent, max(0,by2)), color, -1)
            cv2.putText(img, text, (bx1 + accent + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), ft)

        # ── Region classifications (SAM + CLIP pipeline) ──────────────────
        if self.latest_region_classifications:
            for ri, region in enumerate(self.latest_region_classifications):
                bbox      = region['bbox']
                top_pred  = region['top_prediction']
                region_id = region['region_id']
                color     = COLORS[ri % len(COLORS)]
                x1, y1, x2, y2 = bbox

                # Corner-bracket bbox + thin outline
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)
                corner_bracket(vis_image, x1, y1, x2, y2, color, lw=2)

                # Label: "#id  class  conf%"
                conf_pct = top_pred['confidence']
                label = f"#{region_id}  {top_pred['label']}  {conf_pct:.0%}"

                # Place above bbox, clamp to image top
                ly = y1 - 5
                if ly < 16:
                    ly = y2 + 16
                draw_label(vis_image, label, x1, ly, color)

                # Confidence bar under bbox top edge
                bar_w = x2 - x1
                filled = max(2, int(bar_w * conf_pct))
                cv2.rectangle(vis_image, (x1, y1), (x1 + bar_w, y1 + 3), (40, 40, 40), -1)
                cv2.rectangle(vis_image, (x1, y1), (x1 + filled, y1 + 3), color, -1)

        # ── Full-image classification (no regions) ────────────────────────
        elif self.latest_classification:
            top_pred  = self.latest_classification['output']['top_prediction']
            all_preds = self.latest_classification['output']['all_predictions'][:5]

            # Bottom panel
            panel_h = 110
            semi_rect(vis_image, 0, h - panel_h, w, h, fill=(12, 12, 12), alpha=0.80)

            # Top prediction
            top_label = f"{top_pred['label']}"
            top_conf  = f"{top_pred['confidence']:.1%}"
            cv2.putText(vis_image, top_label, (14, h - panel_h + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 1)
            cv2.putText(vis_image, top_conf, (14, h - panel_h + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 255, 160), 1)

            # Confidence bar for top prediction
            bar_max = min(w // 2 - 20, 260)
            filled  = int(bar_max * top_pred['confidence'])
            cv2.rectangle(vis_image, (14, h - panel_h + 52), (14 + bar_max, h - panel_h + 56), (50, 50, 50), -1)
            cv2.rectangle(vis_image, (14, h - panel_h + 52), (14 + filled, h - panel_h + 56), (80, 255, 80), -1)

            # Top-5 list on the right
            col_x = w - 230
            cv2.putText(vis_image, "Top 5", (col_x, h - panel_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
            for i, pred in enumerate(all_preds):
                row_text  = f"{i+1}. {pred['label']}"
                row_conf  = f"{pred['confidence']:.0%}"
                row_y     = h - panel_h + 34 + i * 16
                row_color = (200, 200, 200) if i > 0 else (80, 255, 80)
                cv2.putText(vis_image, row_text, (col_x, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, row_color, 1)
                cv2.putText(vis_image, row_conf, (w - 42, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, row_color, 1)

        else:
            # Idle hint at bottom
            hint = "Run /vision/classify_all or /vision/classify_bb"
            semi_rect(vis_image, 0, h - 24, w, h, fill=(12, 12, 12), alpha=0.70)
            cv2.putText(vis_image, hint, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

        # ── Found-object highlight ────────────────────────────────────────
        if self.latest_found_object:
            found = self.latest_found_object
            bbox  = found['bbox']
            x1, y1, x2, y2 = bbox
            conf  = found['confidence']

            # Pulsing-style: thin outline + thick corner brackets
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), FOUND_COLOR, 1)
            clen = max(16, int(min(x2-x1, y2-y1) * 0.18))
            for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(vis_image, (px, py), (px + dx*clen, py), FOUND_COLOR, 3)
                cv2.line(vis_image, (px, py), (px, py + dy*clen), FOUND_COLOR, 3)

            # Center dot
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(vis_image, (cx, cy), 5, (15, 15, 15), -1)
            cv2.circle(vis_image, (cx, cy), 5, FOUND_COLOR, 2)
            cv2.circle(vis_image, (cx, cy), 2, (240, 240, 240), -1)

            # Label + confidence bar
            label = f"FOUND  {found['label']}  {conf:.0%}"
            ly = y1 - 5
            if ly < 16:
                ly = y2 + 16
            draw_label(vis_image, label, x1, ly, FOUND_COLOR, fs=0.40, pad=5, accent=4)

            bar_w  = x2 - x1
            filled = max(2, int(bar_w * conf))
            cv2.rectangle(vis_image, (x1, y2 - 4), (x1 + bar_w, y2), (40, 40, 40), -1)
            cv2.rectangle(vis_image, (x1, y2 - 4), (x1 + filled, y2), FOUND_COLOR, -1)

        # ── Top info bar ──────────────────────────────────────────────────
        bar_h = 22
        semi_rect(vis_image, 0, 0, w, bar_h, fill=(12, 12, 12), alpha=0.78)
        if self.latest_region_classifications:
            status = f"CLIP  |  Regions: {len(self.latest_region_classifications)}  |  Frame: {self.frame_counter}"
        elif self.latest_classification:
            status = f"CLIP  |  Full-image mode  |  Frame: {self.frame_counter}"
        else:
            status = f"CLIP Classifier  |  Frame: {self.frame_counter}  |  Idle"
        cv2.putText(vis_image, status, (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Parse command line arguments for custom labels
    candidate_labels = None
    if '--labels' in sys.argv:
        try:
            idx = sys.argv.index('--labels')
            labels_str = sys.argv[idx + 1]
            candidate_labels = [label.strip() for label in labels_str.split(',')]
        except (IndexError, ValueError):
            print("Invalid --labels format. Use: --labels 'cat,dog,car'")
    
    try:
        node = CLIPClassifier(candidate_labels=candidate_labels)
        
        # Use MultiThreadedExecutor for ReentrantCallbackGroup
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.03)
                node.visualization_callback()
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
