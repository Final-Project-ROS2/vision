#!/usr/bin/env python3
"""
CLIP Vision Classifier Node
Focused CLIP-based image classification from /camera/image_raw

Usage:
    ros2 run vision clip_classifier
    ros2 run vision clip_classifier --labels "cat,dog,car,airplane"
    python3 -m vision.clip_classifier
    
Services:
    /vision/classify_all - Classify entire image
        ros2 service call /vision/classify_all std_srvs/srv/Trigger
    
    /vision/classify_detect - Auto-detect objects with SAM, then classify each region
        ros2 service call /vision/classify_detect std_srvs/srv/Trigger
        Note: Requires simple_sam_detector to be running:
              ros2 run vision simple_sam_detector

// image vector embbeing fk this shit


# Terminal 1: SAM Detector
ros2 run vision simple_sam_detector

# Terminal 2: CLIP Classifier  
ros2 run vision clip_classifier


ros2 service call /vision/classify_detect std_srvs/srv/Trigger





You call: ros2 service call /vision/classify_detect std_srvs/srv/Trigger
CLIP's classify_detect_callback is triggered
CLIP internally calls: /vision/detect_objects service (SAM)
SAM's detect_service_callback returns JSON with bounding boxes
CLIP parses the bounding boxes from the JSON response
CLIP crops each region and classifies it
CLIP returns the final JSON with classifications


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

# Import custom service (will be generated after build)
try:
    from vision.srv import ClassifyRegions
    CUSTOM_SRV_AVAILABLE = True
except ImportError:
    CUSTOM_SRV_AVAILABLE = False
    print("⚠️ Custom ClassifyRegions service not available. Build the package first.")

# Try to import CLIP/transformers
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, AutoProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install: pip install torch transformers pillow")

# Try to import SAM custom messages (fallback to placeholder if not built yet)
try:
    from vision.msg import SAMDetections, SAMDetection  # type: ignore
    SAM_MSGS_AVAILABLE = True
except ImportError:
    SAM_MSGS_AVAILABLE = False
    # We'll subscribe to placeholder Image messages instead


class CLIPClassifier(Node):
    """
    CLIP-based image classifier for ROS2
    
    Subscribes to:
        - /camera/image_raw (RGB images from Gazebo camera)
    
    Services:
        - /vision/classify_all (Classify entire image with JSON output)
        - /vision/classify_detect (Auto-detect with SAM + classify regions)
    
    Service Clients:
        - /vision/detect_objects (Calls SAM detector for object detection)
    
    Display:
        - Shows live camera feed with top prediction in OpenCV window
        - Shows classified regions with labels when using classify_detect
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
        self.latest_region_classifications = []  # For classify_detect
        self.frame_counter = 0
        self.frame_captured = False
        
        # CLIP model
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Service client for SAM detector
        self.sam_detector_client = None
        self.sam_sub = None
        
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
        
        # Use Trigger service instead of custom ClassifyRegions
        self.classification_detect_service = self.create_service(
            Trigger,
            '/vision/classify_detect',
            self.classify_detect_callback
        )
        
        # Create service client to call SAM detector
        self.sam_detector_client = self.create_client(Trigger, '/vision/detect_objects')

        # Also subscribe to live SAM detections so CLIP reacts when SAM publishes
        if SAM_MSGS_AVAILABLE:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                10
            )
            self.get_logger().info("👂 Subscribing to: /vision/sam_detections (SAMDetections)")
        else:
            # Fallback: placeholder Image publisher used by simple_sam_detector before custom msgs are built
            self.sam_sub = self.create_subscription(
                Image,
                '/vision/sam_detections',
                self.sam_detections_placeholder_callback,
                10
            )
            self.get_logger().warn("👂 Subscribing to: /vision/sam_detections (placeholder Image). Build msgs for full integration.")
        
        # Check if SAM detector service is available
        self.get_logger().info("⏳ Waiting for /vision/detect_objects service...")
        if self.sam_detector_client.wait_for_service(timeout_sec=20.0):
            self.get_logger().info("✅ SAM detector service is ready!")
        else:
            self.get_logger().warn("⚠️ SAM detector service not available yet. Start with: ros2 run vision simple_sam_detector")
        
        # OpenCV window setup
        self.window_name = "CLIP Classifier - /camera/image_raw"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Timer for visualization (30 Hz)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("🚀 CLIP Classifier Started")
        self.get_logger().info(f"📡 Subscribing to: /camera/image_raw")
        self.get_logger().info(f"🤖 Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"🏷️  Labels: {', '.join(self.candidate_labels)}")
        self.get_logger().info(f"💻 Device: {self.device}")
        self.get_logger().info(f"🔧 Service: /vision/classify_all (classify entire image)")
        self.get_logger().info(f"🔧 Service: /vision/classify_detect (auto-detect + classify)")
        self.get_logger().info(f"   └─ Calls /vision/detect_objects → needs: ros2 run vision simple_sam_detector")
        self.get_logger().info(f"👁️  OpenCV Window: '{self.window_name}'")
        self.get_logger().info("💡 Usage: ros2 service call /vision/classify_all std_srvs/srv/Trigger")
        self.get_logger().info("💡 Usage: ros2 service call /vision/classify_detect std_srvs/srv/Trigger")
        self.get_logger().info("🔌 Live link: reacts to /vision/sam_detections by classifying regions automatically")
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("❌ CLIP not available! Install: pip install torch transformers pillow")
            return
        
        try:
            self.get_logger().info("🔧 Loading CLIP model...")
            model_name = "openai/clip-vit-base-patch32"
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.model.eval()  # Set to evaluation mode
            
            self.get_logger().info(f"✅ CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load CLIP model: {e}")
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
                self.get_logger().info(f"📸 Captured frame {self.frame_counter} for classification")
                
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
                self.get_logger().warn("⚠️ No frame captured yet")
                return response
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "single_clip",
                    "success": False,
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("❌ CLIP model not available")
                return response
            
            self.get_logger().info("🔍 Running CLIP classification on captured frame...")
            
            # Run classification on captured frame
            classification_data = self._classify_image(self.captured_frame)
            self.latest_classification = classification_data
            self.latest_region_classifications = []  # Clear region classifications
            
            response.success = True
            response.message = json.dumps(classification_data, indent=2)
            
            top_pred = classification_data['output']['top_prediction']
            self.get_logger().info(
                f"✅ Classification complete: {top_pred['label']} "
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
            self.get_logger().error(f"❌ Classification error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response

    def sam_detections_callback(self, msg: 'SAMDetections'):
        """Handle incoming SAM detections and classify regions automatically.

        This lets the CLIP node consume the SAM pipeline without needing to call the service.
        """
        try:
            if not CLIP_AVAILABLE or self.model is None:
                self.get_logger().warn("CLIP model not available yet; ignoring SAM detections")
                return
            if self.captured_frame is None:
                self.get_logger().warn("No captured frame available; waiting for /camera/image_raw")
                return

            # Extract bboxes from SAMDetections
            bboxes: List[List[int]] = []
            for det in msg.detections:
                bbox = list(det.bbox)
                if len(bbox) == 4:
                    bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

            if not bboxes:
                self.get_logger().warn("SAM detections message has no valid bounding boxes")
                return

            self.get_logger().info(f"🧩 Received {len(bboxes)} SAM regions → classifying with CLIP…")
            classification_data = self._classify_regions(self.captured_frame, bboxes)
            self.latest_region_classifications = classification_data['output']['classified_regions']
            self.latest_classification = None
        except Exception as e:
            self.get_logger().error(f"Error handling SAM detections: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def sam_detections_placeholder_callback(self, msg: Image):
        """Handle placeholder /vision/sam_detections messages (Image) from SimpleSAM.

        The placeholder encodes only counts; we can't get bboxes, so just log and wait
        for the service-based integration or rebuilt messages.
        """
        try:
            # The simple_sam_detector encodes counts in height/width for visibility only
            self.get_logger().info(
                f"📨 Placeholder SAM detections received: count_hint={msg.height}, frame={msg.width}"
            )
            self.get_logger().info("Build custom messages to enable auto-classification from topic events.")
        except Exception as e:
            self.get_logger().error(f"Error handling placeholder SAM detections: {e}")
    
    def classify_detect_callback(self, request, response):
        """Service callback for /vision/classify_detect - detect objects then classify regions"""
        try:
            if self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": "No frame captured yet from /camera/image_raw",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No frame captured yet")
                return response
            
            # Confirm using captured frame
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"📸 Using CAPTURED FRAME {self.frame_counter} for detection + classification")
            self.get_logger().info(f"   Frame shape: {self.captured_frame.shape}")
            self.get_logger().info("=" * 80)
            
            if not CLIP_AVAILABLE or self.model is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": "CLIP model not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("❌ CLIP model not available")
                return response
            
            # Step 1: Call SAM detector to get bounding boxes
            self.get_logger().info("🔍 Step 1/2: Calling /vision/detect_objects service...")
            
            # Wait for SAM detector service with shorter timeout
            if not self.sam_detector_client.wait_for_service(timeout_sec=2.0):
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": "SAM detector service /vision/detect_objects not available. Run: ros2 run vision simple_sam_detector",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("❌ /vision/detect_objects service not available")
                self.get_logger().error("   Make sure SAM detector is running: ros2 run vision simple_sam_detector")
                return response
            
            self.get_logger().info("✅ SAM detector service found, sending request...")
            
            # Call SAM detector
            sam_request = Trigger.Request()
            sam_future = self.sam_detector_client.call_async(sam_request)
            
            self.get_logger().info("⏳ Waiting for SAM detection to complete (this may take 10-15 seconds)...")
            
            # Wait for response with timeout and active spinning
            timeout_start = time.time()
            timeout_duration = 20.0  # Reduced from 30s
            
            while not sam_future.done():
                elapsed = time.time() - timeout_start
                if elapsed >= timeout_duration:
                    self.get_logger().error(f"❌ SAM detection timeout after {timeout_duration}s")
                    self.get_logger().error("   This usually means:")
                    self.get_logger().error("   1. SAM detector is not running → Start with: ros2 run vision simple_sam_detector")
                    self.get_logger().error("   2. SAM is processing but too slow → Check GPU/CPU usage")
                    self.get_logger().error("   3. SAM is not receiving images → Check /camera/image_raw topic")
                    
                    response.success = False
                    response.message = json.dumps({
                        "pipeline": "clip_with_detection",
                        "success": False,
                        "error": f"SAM detection timeout after {timeout_duration}s. Make sure simple_sam_detector is running and processing images.",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }, indent=2)
                    return response
                
                # Actively spin to process callbacks
                rclpy.spin_once(self, timeout_sec=0.1)
                
                # Show progress every 2 seconds
                if int(elapsed) % 2 == 0 and elapsed > 0:
                    self.get_logger().info(f"   Still waiting... ({elapsed:.0f}s / {timeout_duration}s)")
            
            try:
                sam_response = sam_future.result()
            except Exception as e:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": f"Failed to get SAM response: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error(f"❌ Failed to get SAM response: {e}")
                return response
            
            self.get_logger().info(f"✅ SAM response received in {time.time() - timeout_start:.1f}s")
            
            # Print full SAM JSON response for debugging
            self.get_logger().info("=" * 80)
            self.get_logger().info("📋 SAM JSON RESPONSE (FULL):")
            self.get_logger().info("=" * 80)
            self.get_logger().info(sam_response.message)
            self.get_logger().info("=" * 80)
            
            if not sam_response.success:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": f"SAM detection failed: {sam_response.message}",
                    "sam_message": sam_response.message[:500],  # First 500 chars for debugging
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error(f"❌ SAM detection failed")
                self.get_logger().error(f"   SAM message: {sam_response.message[:200]}")
                return response
            
            # Parse SAM detection results
            self.get_logger().info("📋 Parsing SAM detection results...")
            try:
                sam_data = json.loads(sam_response.message)
                self.get_logger().info(f"✅ SAM data parsed successfully")
                
                # Debug: Show SAM data structure
                self.get_logger().debug(f"   SAM data keys: {list(sam_data.keys())}")
                
                if not sam_data.get('success', False):
                    response.success = False
                    response.message = json.dumps({
                        "pipeline": "clip_with_detection",
                        "success": False,
                        "error": f"SAM detection unsuccessful: {sam_data.get('error', 'Unknown error')}",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }, indent=2)
                    return response
                
                # Extract bounding boxes from detection results
                # SAM detector returns: {"success": true, "detections": [{"detections": [...], ...}]}
                bboxes = []
                
                self.get_logger().info("🔍 Searching for bounding boxes in SAM response...")
                
                # Check for 'output' field (from SAM schema)
                if 'output' in sam_data:
                    detections_list = sam_data['output'].get('detections', [])
                    self.get_logger().info(f"📦 Found 'output' field with {len(detections_list)} detections")
                    for idx, det in enumerate(detections_list):
                        bbox = det.get('bbox')
                        if bbox and len(bbox) == 4:
                            self.get_logger().info(f"   ✓ Detection {idx}: bbox={bbox}")
                            bboxes.append(bbox)
                        else:
                            self.get_logger().warn(f"   ✗ Detection {idx}: invalid bbox={bbox}")
                            
                # Fallback: Check for 'detections' field
                elif 'detections' in sam_data and len(sam_data['detections']) > 0:
                    self.get_logger().info(f"📦 Found 'detections' field with {len(sam_data['detections'])} detection sets")
                    for set_idx, detection_set in enumerate(sam_data['detections']):
                        detections_list = detection_set.get('detections', [])
                        self.get_logger().info(f"   Set {set_idx}: {len(detections_list)} detections")
                        for det_idx, det in enumerate(detections_list):
                            bbox = det.get('bbox')
                            if bbox and len(bbox) == 4:
                                self.get_logger().info(f"      ✓ Detection {det_idx}: bbox={bbox}")
                                bboxes.append(bbox)
                            else:
                                self.get_logger().warn(f"      ✗ Detection {det_idx}: invalid bbox={bbox}")
                else:
                    self.get_logger().warn("⚠️ SAM response has neither 'output' nor 'detections' field")
                    self.get_logger().info(f"   Available keys: {list(sam_data.keys())}")
                
                if not bboxes:
                    self.get_logger().warn("⚠️ No bounding boxes extracted from SAM response")
                    self.get_logger().info(f"   SAM response structure (first 500 chars): {json.dumps(sam_data, indent=2)[:500]}...")
                    
                    response.success = False
                    response.message = json.dumps({
                        "pipeline": "clip_with_detection",
                        "success": False,
                        "error": "No objects detected by SAM detector",
                        "hint": "Try adjusting camera position or ensure objects are visible in scene",
                        "sam_response_sample": str(sam_data)[:300],
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }, indent=2)
                    return response
                
                self.get_logger().info("=" * 80)
                self.get_logger().info(f"✅ SAM detected {len(bboxes)} objects with valid bounding boxes")
                self.get_logger().info(f"   Bounding boxes: {bboxes}")
                self.get_logger().info("=" * 80)
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"❌ Failed to parse SAM response as JSON: {e}")
                self.get_logger().error(f"   Response (first 300 chars): {sam_response.message[:300]}")
                
                response.success = False
                response.message = json.dumps({
                    "pipeline": "clip_with_detection",
                    "success": False,
                    "error": f"Failed to parse SAM response as JSON: {e}",
                    "raw_response_sample": sam_response.message[:300],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                return response
            
            # Step 2: Classify each detected region with CLIP
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"🔍 Step 2/2: Classifying {len(bboxes)} detected regions with CLIP...")
            self.get_logger().info(f"📸 Using CAPTURED FRAME shape: {self.captured_frame.shape}")
            self.get_logger().info("=" * 80)
            
            classification_data = self._classify_regions(self.captured_frame, bboxes)
            self.latest_region_classifications = classification_data['output']['classified_regions']
            self.latest_classification = None  # Clear full image classification
            
            # Add SAM detection info to response
            classification_data['sam_detection'] = {
                'total_detections': len(bboxes),
                'bboxes': bboxes,
                'detection_time_ms': int((time.time() - timeout_start) * 1000)
            }
            
            response.success = True
            response.message = json.dumps(classification_data, indent=2)
            
            self.get_logger().info(f"✅ Pipeline complete: {len(bboxes)} regions detected and classified")
            
            # Log each region's top prediction
            for region in classification_data['output']['classified_regions']:
                top_pred = region['top_prediction']
                bbox = region['bbox']
                self.get_logger().info(
                    f"   Region {region['region_id']} {bbox}: {top_pred['label']} "
                    f"(confidence: {top_pred['confidence']:.2f})"
                )
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "pipeline": "clip_with_detection",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"❌ Classification error: {e}")
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
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(inputs.pixel_values)
            text_features = self.model.get_text_features(inputs.input_ids)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        image_vector = image_features[0].cpu().numpy().tolist()
        text_vectors = text_features.cpu().numpy()
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(probs_np)[::-1]
        
        # Build predictions list
        all_predictions = []
        for idx in sorted_indices:
            all_predictions.append({
                "label": self.candidate_labels[idx],
                "confidence": float(probs_np[idx])
            })
        
        # Build text vectors with labels
        text_vectors_list = []
        for i, label in enumerate(self.candidate_labels):
            text_vectors_list.append({
                "label": label,
                "vector": text_vectors[i].tolist()
            })
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build JSON schema
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
                    "confidence": round(all_predictions[0]["confidence"], 2)
                },
                "all_predictions": [
                    {
                        "label": pred["label"],
                        "confidence": round(pred["confidence"], 2)
                    }
                    for pred in all_predictions
                ],
                "embedding": {
                    "image_vector": image_vector,
                    "text_vectors": text_vectors_list,
                    "similarity_method": "cosine"
                },
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
                self.get_logger().warn(f"⚠️ Skipping invalid bbox: {bbox}")
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
        
        # Check if we have region classifications (from classify_detect)
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
                "Call /vision/classify_all or /vision/classify_detect",
                (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
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
            print("⚠️ Invalid --labels format. Use: --labels 'cat,dog,car'")
    
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
