#!/usr/bin/env python3
"""
SAM + CLIP Pipeline Node
Combines SAM object detection with CLIP classification

This node:
1. Subscribes to /camera/image_raw (captures ONE frame)
2. Calls /vision/detect_objects service (SAM) to get bounding boxes
3. Classifies each detected region using CLIP
4. Returns combined results

Usage:
    ros2 run vision sam_clip_pipeline
    
Service:
    ros2 service call /vision/classify_detect std_srvs/srv/Trigger

Dependencies:
    - simple_sam_detector must be running
    - Camera must be publishing to /camera/image_raw
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict
import time

# Try to import CLIP/transformers
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, AutoProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Install: pip install torch transformers pillow")


class SAMCLIPPipeline(Node):
    """
    SAM + CLIP Pipeline for object detection and classification
    
    Subscribes to:
        - /camera/image_raw (RGB images - captures ONE frame)
    
    Service Clients:
        - /vision/detect_objects (Calls SAM detector)
    
    Services:
        - /vision/classify_detect (Main pipeline service)
    """
    
    def __init__(self):
        super().__init__('sam_clip_pipeline')
        
        # Default classification labels
        self.candidate_labels = [
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
        
        # Image capture
        self.captured_frame = None
        self.frame_captured = False
        self.frame_counter = 0
        
        # SAM detection results
        self.sam_bounding_boxes = []
        self.sam_json_response = None
        
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
        
        # Subscribe to camera (will capture ONE frame)
        self.get_logger().info("📡 Subscribing to /camera/image_raw...")
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            self.image_qos
        )
        
        # Create service client to call SAM detector
        self.get_logger().info("🔧 Creating SAM detector service client...")
        self.sam_detector_client = self.create_client(Trigger, '/vision/detect_objects')
        
        # Check if SAM detector service is available
        self.get_logger().info("⏳ Waiting for /vision/detect_objects service...")
        if self.sam_detector_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("✅ SAM detector service is ready!")
        else:
            self.get_logger().warn("⚠️ SAM detector service not available yet")
            self.get_logger().warn("   Start with: ros2 run vision simple_sam_detector")
        
        # Create main pipeline service
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/classify_detect',
            self.pipeline_callback
        )
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 SAM + CLIP Pipeline Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 Subscribed to: /camera/image_raw (will capture ONE frame)")
        self.get_logger().info("🤖 CLIP Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"💻 Device: {self.device}")
        self.get_logger().info(f"🏷️  Labels: {', '.join(self.candidate_labels)}")
        self.get_logger().info("🔧 Service: /vision/classify_detect")
        self.get_logger().info("   └─ Calls /vision/detect_objects (SAM)")
        self.get_logger().info("=" * 80)
        self.get_logger().info("💡 Usage: ros2 service call /vision/classify_detect std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("❌ CLIP not available!")
            self.get_logger().error("   Install: pip install torch transformers pillow")
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
        """Handle incoming RGB images - CAPTURE ONLY ONE FRAME"""
        try:
            if not self.frame_captured:
                # Convert ROS Image message to OpenCV format (BGR8)
                self.captured_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.frame_counter += 1
                self.frame_captured = True
                
                self.get_logger().info("=" * 80)
                self.get_logger().info(f"📸 CAPTURED FRAME {self.frame_counter}")
                self.get_logger().info(f"   Shape: {self.captured_frame.shape}")
                self.get_logger().info(f"   Size: {self.captured_frame.shape[1]}x{self.captured_frame.shape[0]}")
                self.get_logger().info("   ✅ Ready for detection and classification")
                self.get_logger().info("=" * 80)
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to convert image: {e}")
    
    def pipeline_callback(self, request, response):
        """
        Main pipeline service callback
        
        Steps:
        1. Check if frame is captured
        2. Call SAM detector service to get bounding boxes
        3. Parse SAM JSON response
        4. Classify each region with CLIP
        5. Return combined results
        """
        try:
            # Step 0: Verify frame is captured
            if not self.frame_captured or self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "sam_clip",
                    "success": False,
                    "error": "No frame captured yet. Waiting for /camera/image_raw...",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No frame captured yet")
                return response
            
            # Step 1: Call SAM detector service
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 1/3: Calling SAM Detector Service")
            self.get_logger().info("=" * 80)
            
            sam_success, sam_json = self._call_sam_detector()
            
            if not sam_success:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "sam_clip",
                    "success": False,
                    "error": "SAM detection failed",
                    "sam_response": sam_json,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                return response
            
            # Step 2: Parse SAM response to extract bounding boxes
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 2/3: Parsing SAM Response")
            self.get_logger().info("=" * 80)
            
            bboxes = self._parse_sam_response(sam_json)
            
            if not bboxes:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "sam_clip",
                    "success": False,
                    "error": "No bounding boxes extracted from SAM response",
                    "sam_response": sam_json,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                return response
            
            # Step 3: Classify each region with CLIP
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"🔍 STEP 3/3: Classifying {len(bboxes)} Regions with CLIP")
            self.get_logger().info("=" * 80)
            
            classification_results = self._classify_regions_with_clip(bboxes)
            
            # Build final response
            response.success = True
            response.message = json.dumps(classification_results, indent=2)
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"✅ PIPELINE COMPLETE")
            self.get_logger().info(f"   Detected: {len(bboxes)} objects")
            self.get_logger().info(f"   Classified: {len(classification_results['output']['classified_regions'])} regions")
            self.get_logger().info("=" * 80)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "pipeline": "sam_clip",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"❌ Pipeline error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def _call_sam_detector(self) -> tuple:
        """
        Call SAM detector service and return (success, json_response)
        
        Returns:
            tuple: (success: bool, json_data: dict or str)
        """
        # Check if service is available
        if not self.sam_detector_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("❌ SAM detector service not available")
            self.get_logger().error("   Make sure to run: ros2 run vision simple_sam_detector")
            return False, {"error": "SAM service not available"}
        
        self.get_logger().info("📞 Calling /vision/detect_objects service...")
        
        # Create request
        sam_request = Trigger.Request()
        
        # Call service asynchronously
        sam_future = self.sam_detector_client.call_async(sam_request)
        
        # Wait for response with timeout
        timeout_start = time.time()
        timeout_duration = 20.0
        
        while not sam_future.done():
            elapsed = time.time() - timeout_start
            if elapsed >= timeout_duration:
                self.get_logger().error(f"❌ SAM service timeout after {timeout_duration}s")
                return False, {"error": "SAM service timeout"}
            
            # Actively spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Show progress
            if int(elapsed) % 2 == 0 and elapsed > 0:
                self.get_logger().info(f"   ⏳ Waiting... ({elapsed:.0f}s / {timeout_duration}s)")
        
        # Get result
        try:
            sam_response = sam_future.result()
            elapsed_time = time.time() - timeout_start
            
            self.get_logger().info(f"✅ SAM response received in {elapsed_time:.1f}s")
            
            # Print full SAM JSON response
            self.get_logger().info("=" * 80)
            self.get_logger().info("📋 SAM JSON RESPONSE (FULL):")
            self.get_logger().info("=" * 80)
            self.get_logger().info(sam_response.message)
            self.get_logger().info("=" * 80)
            
            # Parse JSON
            try:
                sam_json = json.loads(sam_response.message)
                return sam_response.success, sam_json
            except json.JSONDecodeError as e:
                self.get_logger().error(f"❌ Failed to parse SAM response as JSON: {e}")
                return False, {"error": f"JSON parse error: {e}", "raw": sam_response.message[:500]}
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to get SAM response: {e}")
            return False, {"error": str(e)}
    
    def _parse_sam_response(self, sam_json: dict) -> List[List[int]]:
        """
        Parse SAM JSON response to extract bounding boxes
        
        Args:
            sam_json: SAM detector response as dictionary
            
        Returns:
            List of bounding boxes [[x1, y1, x2, y2], ...]
        """
        bboxes = []
        
        self.get_logger().info(f"📦 SAM response keys: {list(sam_json.keys())}")
        
        # Check for 'detections' field (current SAM format)
        if 'detections' in sam_json and isinstance(sam_json['detections'], list):
            self.get_logger().info(f"   Found 'detections' field with {len(sam_json['detections'])} sets")
            
            for set_idx, detection_set in enumerate(sam_json['detections']):
                detections_list = detection_set.get('detections', [])
                self.get_logger().info(f"   Set {set_idx}: {len(detections_list)} detections")
                
                for det_idx, det in enumerate(detections_list):
                    bbox = det.get('bbox')
                    if bbox and len(bbox) == 4:
                        self.get_logger().info(f"      ✓ Detection {det_idx}: bbox={bbox}")
                        bboxes.append(bbox)
                    else:
                        self.get_logger().warn(f"      ✗ Detection {det_idx}: invalid bbox={bbox}")
        
        # Alternative: Check for 'output' field
        elif 'output' in sam_json:
            detections_list = sam_json['output'].get('detections', [])
            self.get_logger().info(f"   Found 'output' field with {len(detections_list)} detections")
            
            for det_idx, det in enumerate(detections_list):
                bbox = det.get('bbox')
                if bbox and len(bbox) == 4:
                    self.get_logger().info(f"      ✓ Detection {det_idx}: bbox={bbox}")
                    bboxes.append(bbox)
                else:
                    self.get_logger().warn(f"      ✗ Detection {det_idx}: invalid bbox={bbox}")
        
        else:
            self.get_logger().warn("⚠️ SAM response has neither 'detections' nor 'output' field")
            self.get_logger().info(f"   Available keys: {list(sam_json.keys())}")
        
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"✅ Extracted {len(bboxes)} bounding boxes")
        if bboxes:
            self.get_logger().info(f"   Bounding boxes: {bboxes}")
        self.get_logger().info("=" * 80)
        
        return bboxes
    
    def _classify_regions_with_clip(self, bboxes: List[List[int]]) -> Dict:
        """
        Classify each detected region using CLIP
        
        Args:
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Dictionary with classification results
        """
        if not CLIP_AVAILABLE or self.model is None:
            self.get_logger().error("❌ CLIP model not available")
            return {
                "pipeline": "sam_clip",
                "success": False,
                "error": "CLIP model not available",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        start_time = time.time()
        classified_regions = []
        
        h, w = self.captured_frame.shape[:2]
        
        for region_id, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            # Clamp bbox to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                self.get_logger().warn(f"⚠️ Skipping invalid bbox {region_id}: {bbox}")
                continue
            
            self.get_logger().info(f"   Classifying region {region_id}: {[x1, y1, x2, y2]}")
            
            # Crop region from captured frame
            region_bgr = self.captured_frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB for CLIP
            region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(region_rgb)
            
            # Prepare CLIP inputs
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
            
            # Log top prediction
            top_pred = all_predictions[0]
            self.get_logger().info(f"      ✓ Top: {top_pred['label']} ({top_pred['confidence']:.2f})")
            
            # Build region result
            region_result = {
                "region_id": region_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "top_prediction": {
                    "label": top_pred["label"],
                    "confidence": top_pred["confidence"]
                },
                "all_predictions": all_predictions[:5]  # Top 5
            }
            
            classified_regions.append(region_result)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build final schema
        schema = {
            "pipeline": "sam_clip",
            "success": True,
            "model": "openai/clip-vit-base-patch32",
            "input": {
                "image_id": f"frame_{self.frame_counter:06d}",
                "image_shape": list(self.captured_frame.shape),
                "num_regions": len(bboxes),
                "candidate_labels": self.candidate_labels
            },
            "output": {
                "classified_regions": classified_regions,
                "summary": {
                    "total_regions": len(classified_regions),
                    "processing_time_ms": processing_time_ms
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "device": self.device
            }
        }
        
        return schema
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = SAMCLIPPipeline()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
