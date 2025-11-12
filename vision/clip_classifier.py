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
       Find bounding box by label name (runs CLIP classification automatically on SAM regions)
       ros2 service call /vision/find_object custom_interfaces/srv/FindObject "{label: 'piston_rod'}"

Setup:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 run vision clip_classifier
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple
import time

# Import custom interfaces
try:
    from custom_interfaces.srv import ClassifyBBox
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

# Try to import CLIP/transformers
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, AutoProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install: pip install torch transformers pillow")

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
        
        # Default labels if none provided
        self.candidate_labels = candidate_labels or [
            "cobot",
            "green_cube",
            "drill",
            "gear",
            "monkey_wrench",
            "piston_rod",
            "washer"
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
        self.processor = None
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
            '/camera/image_raw',
            self.rgb_callback,
            self.image_qos
        )
        
        # Classification services
        self.classification_all_service = self.create_service(
            Trigger,
            '/vision/classify_all',
            self.classify_all_callback
        )
        
        if CUSTOM_INTERFACES_AVAILABLE:
            self.classification_bb_service = self.create_service(
                ClassifyBBox,
                '/vision/classify_bb',
                self.classify_bb_callback
            )
        else:
            self.classification_bb_service = self.create_service(
                Trigger,
                '/vision/classify_bb',
                self.classify_bb_callback_fallback
            )
        
        # Find object by label service
        if CUSTOM_INTERFACES_AVAILABLE:
            try:
                from custom_interfaces.srv import FindObject
                self.find_object_service = self.create_service(
                    FindObject,
                    '/vision/find_object',
                    self.find_object_callback
                )
                self.get_logger().info("Service created: /vision/find_object")
            except ImportError:
                self.get_logger().warn("FindObject service not available. Add to custom_interfaces.")
        
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
        self.window_name = "CLIP Classifier - /camera/image_raw"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Timer for visualization (30 Hz)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("CLIP Classifier Started")
        self.get_logger().info(f"Subscribing to: /camera/image_raw")
        self.get_logger().info(f"Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"Labels: {', '.join(self.candidate_labels)}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Service: /vision/classify_all")
        self.get_logger().info(f"Service: /vision/classify_bb")
        self.get_logger().info(f"Service: /vision/find_object")
        self.get_logger().info(f"Subscriber: /vision/sam_detections (auto-classify on SAM publish)")
        self.get_logger().info(f"OpenCV Window: '{self.window_name}'")
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("CLIP not available! Install: pip install torch transformers pillow")
            return
        
        try:
            self.get_logger().info("Loading CLIP model...")
            model_name = "openai/clip-vit-base-patch32"
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.model.eval()  # Set to evaluation mode
            
            self.get_logger().info(f"CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None
    
    def rgb_callback(self, msg: Image):
        """Handle incoming RGB images from /camera/image_raw"""
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_counter += 1
            
            # Capture first frame for classification
            if not self.frame_captured:
                self.captured_frame = self.latest_rgb.copy()
                self.frame_captured = True
                self.get_logger().info(f"Captured frame {self.frame_counter} for classification")
                
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def classify_all_callback(self, request, response):
        """Service callback for /vision/classify_all - classify entire image"""
        try:
            if self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "single_clip",
                    "success": False,
                    "error": "No frame captured yet from /camera/image_raw",
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
        
        return response
    
    def classify_bb_callback(self, request, response):
        """Service callback for /vision/classify_bb with custom ClassifyBBox service"""
        try:
            if self.captured_frame is None:
                response.success = False
                response.label = ""
                response.confidence = 0.0
                response.all_predictions = json.dumps({
                    "error": "No frame captured yet from /camera/image_raw",
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
                    "error": "No frame captured yet from /camera/image_raw",
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

    def sam_detections_callback(self, msg: 'SAMDetections'):
        """Handle incoming SAM detections and automatically classify each region"""
        try:
            if not CLIP_AVAILABLE or self.model is None:
                self.get_logger().warn("CLIP model not available, ignoring SAM detections")
                return
            
            if self.captured_frame is None:
                self.get_logger().warn("No captured frame, waiting for /camera/image_raw")
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
    
    def find_object_callback(self, request, response):
        """
        Service callback for /vision/find_object - find bounding box by label
        Automatically runs CLIP classification on available SAM detections
        
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
            
            # Check if we have a captured frame
            if self.captured_frame is None:
                response.success = False
                response.message = "No frame captured yet from /camera/image_raw"
                response.bbox = []
                response.confidence = 0.0
                self.get_logger().warn("No captured frame available")
                return response
            
            # Check if CLIP is available
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = "CLIP model not available"
                response.bbox = []
                response.confidence = 0.0
                self.get_logger().error("CLIP model not available")
                return response
            
            # If no classified regions, try to get SAM detections and classify them
            if not self.latest_region_classifications:
                self.get_logger().info("No classified regions available. Checking for SAM detections...")
                
                # Try calling SAM detection service to get fresh detections
                try:
                    sam_client = self.create_client(Trigger, '/vision/sam_detect')
                    
                    if not sam_client.wait_for_service(timeout_sec=2.0):
                        response.success = False
                        response.message = "SAM detection service not available. Please run simple_sam_detector first."
                        response.bbox = []
                        response.confidence = 0.0
                        self.latest_found_object = None
                        self.get_logger().warn("SAM service not available")
                        return response
                    
                    # Call SAM service
                    self.get_logger().info("Calling SAM detection service...")
                    sam_request = Trigger.Request()
                    future = sam_client.call_async(sam_request)
                    rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                    
                    if not future.done() or not future.result().success:
                        response.success = False
                        response.message = "SAM detection failed. Please check simple_sam_detector."
                        response.bbox = []
                        response.confidence = 0.0
                        self.latest_found_object = None
                        return response
                    
                    # Wait for SAM detections message to arrive
                    self.get_logger().info("Waiting for SAM detections message...")
                    time.sleep(0.5)
                    
                    # Check if we received detections via subscriber
                    if not self.latest_region_classifications:
                        response.success = False
                        response.message = "No objects detected by SAM or classification failed."
                        response.bbox = []
                        response.confidence = 0.0
                        self.latest_found_object = None
                        return response
                    
                except Exception as e:
                    self.get_logger().error(f"Error calling SAM service: {e}")
                    response.success = False
                    response.message = f"SAM service error: {str(e)}"
                    response.bbox = []
                    response.confidence = 0.0
                    self.latest_found_object = None
                    return response
            
            # Now search for the target label in classified regions
            self.get_logger().info(f"Searching for '{target_label}' in {len(self.latest_region_classifications)} classified regions...")
            
            # Find all matching regions
            matching_regions = []
            for region in self.latest_region_classifications:
                top_pred = region['top_prediction']
                if top_pred['label'] == target_label:
                    matching_regions.append({
                        'bbox': region['bbox'],
                        'confidence': top_pred['confidence'],
                        'region_id': region['region_id']
                    })
            
            # Check if any matches found
            if not matching_regions:
                response.success = False
                response.message = f"Label '{target_label}' not found in classified regions"
                response.bbox = []
                response.confidence = 0.0
                self.latest_found_object = None
                self.get_logger().info(f"Label '{target_label}' not found")
                return response
            
            # Sort by confidence and get highest
            matching_regions.sort(key=lambda x: x['confidence'], reverse=True)
            best_match = matching_regions[0]
            
            # Check confidence threshold
            if best_match['confidence'] < 0.5:
                response.success = False
                response.message = f"Label '{target_label}' found but confidence too low ({best_match['confidence']:.2f} < 0.5)"
                response.bbox = []
                response.confidence = float(best_match['confidence'])
                self.latest_found_object = None
                self.get_logger().info(f"Label '{target_label}' confidence too low: {best_match['confidence']:.2f}")
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
            response.message = f"Found '{target_label}' with confidence {best_match['confidence']:.2f}"
            response.bbox = best_match['bbox']
            response.confidence = float(best_match['confidence'])
            
            self.get_logger().info(
                f"Found '{target_label}': bbox={best_match['bbox']}, "
                f"confidence={best_match['confidence']:.2f}, region_id={best_match['region_id']}"
            )
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            response.bbox = []
            response.confidence = 0.0
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
        
        # Prepare inputs
        inputs = self.processor(
            text=self.candidate_labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
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
        
        for region_id, bbox in enumerate(bboxes):
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
                continue
            
            # Crop region
            region_bgr = rgb_image[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(region_rgb)
            
            # Prepare inputs
            inputs = self.processor(
                text=self.candidate_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
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
            
            # Build region result
            region_result = {
                "region_id": region_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "top_prediction": {
                    "label": all_predictions[0]["label"],
                    "confidence": all_predictions[0]["confidence"]
                },
                "all_predictions": all_predictions[:10]  # Top 10
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
        if self.captured_frame is None:
            # Show waiting message
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank, 
                "Waiting to capture frame from /camera/image_raw...", 
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        # Create visualization image from captured frame
        vis_image = self.captured_frame.copy()
        h, w = vis_image.shape[:2]
        
        # Check if we have region classifications (from SAM auto-classification)
        if self.latest_region_classifications:
            # Draw each classified region with bounding box and label
            for region in self.latest_region_classifications:
                bbox = region['bbox']
                top_pred = region['top_prediction']
                region_id = region['region_id']
                
                # Draw bounding box
                cv2.rectangle(
                    vis_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 255),  # Yellow for classified regions
                    3
                )
                
                # Prepare label text
                label = f"#{region_id}: {top_pred['label']}"
                conf = f"{top_pred['confidence']:.1%}"
                
                # Calculate label position (above bbox)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw label background
                cv2.rectangle(
                    vis_image,
                    (bbox[0], bbox[1] - label_size[1] - 25),
                    (bbox[0] + max(label_size[0], 100), bbox[1]),
                    (0, 255, 255),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (bbox[0] + 5, bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
                
                # Draw confidence
                cv2.putText(
                    vis_image,
                    conf,
                    (bbox[0] + 5, bbox[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            # Add info text
            info_text = f"Classified Regions: {len(self.latest_region_classifications)}"
            cv2.putText(
                vis_image,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            
        # Draw full image classification overlay (if available and no regions)
        elif self.latest_classification:
            top_pred = self.latest_classification['output']['top_prediction']
            all_preds = self.latest_classification['output']['all_predictions'][:5]  # Top 5
            
            # Draw semi-transparent overlay at bottom
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Draw top prediction (large)
            label_text = f"Top: {top_pred['label']}"
            conf_text = f"{top_pred['confidence']:.1%}"
            
            cv2.putText(
                vis_image,
                label_text,
                (20, h-100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
            
            cv2.putText(
                vis_image,
                conf_text,
                (20, h-60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            
            # Draw top 5 predictions (smaller, on right)
            y_offset = h - 120
            for i, pred in enumerate(all_preds):
                text = f"{i+1}. {pred['label']}: {pred['confidence']:.1%}"
                cv2.putText(
                    vis_image,
                    text,
                    (w - 350, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        else:
            # Show "Call service to classify" message
            cv2.putText(
                vis_image,
                "Call /vision/classify_all or /vision/classify_bb",
                (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Draw found object highlight (if available)
        if self.latest_found_object:
            found = self.latest_found_object
            bbox = found['bbox']
            
            # Draw thick green bounding box for found object
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),  # Green for found object
                5
            )
            
            # Prepare label text
            label = f"FOUND: {found['label']}"
            conf = f"Conf: {found['confidence']:.1%}"
            
            # Calculate label position (above bbox)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            
            # Draw label background (green)
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1] - label_size[1] - 35),
                (bbox[0] + max(label_size[0], 150), bbox[1]),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (bbox[0] + 5, bbox[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2
            )
            
            # Draw confidence
            cv2.putText(
                vis_image,
                conf,
                (bbox[0] + 5, bbox[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            # Add corner markers
            corner_size = 15
            # Top-left
            cv2.line(vis_image, (bbox[0], bbox[1]), (bbox[0] + corner_size, bbox[1]), (0, 255, 0), 5)
            cv2.line(vis_image, (bbox[0], bbox[1]), (bbox[0], bbox[1] + corner_size), (0, 255, 0), 5)
            # Top-right
            cv2.line(vis_image, (bbox[2], bbox[1]), (bbox[2] - corner_size, bbox[1]), (0, 255, 0), 5)
            cv2.line(vis_image, (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_size), (0, 255, 0), 5)
            # Bottom-left
            cv2.line(vis_image, (bbox[0], bbox[3]), (bbox[0] + corner_size, bbox[3]), (0, 255, 0), 5)
            cv2.line(vis_image, (bbox[0], bbox[3]), (bbox[0], bbox[3] - corner_size), (0, 255, 0), 5)
            # Bottom-right
            cv2.line(vis_image, (bbox[2], bbox[3]), (bbox[2] - corner_size, bbox[3]), (0, 255, 0), 5)
            cv2.line(vis_image, (bbox[2], bbox[3]), (bbox[2], bbox[3] - corner_size), (0, 255, 0), 5)
        
        # Add title bar
        cv2.putText(
            vis_image,
            f"CLIP Classifier | Frame: {self.frame_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4
        )
        
        cv2.putText(
            vis_image,
            f"CLIP Classifier | Frame: {self.frame_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Show image
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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
