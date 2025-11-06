#!/usr/bin/env python3
"""
SAM + CLIP Pipeline Node (Integrated)
Combines SAM object detection with CLIP classification - ALL IN ONE NODE

This node:
1. Subscribes to /camera/image_raw (captures ONE frame)
2. Implements SAM detection directly (OpenCV-based segmentation)
3. Classifies each detected region using CLIP
4. Saves results as JSON file
5. Returns combined results

Usage:
    ros2 run vision sam_clip_pipeline
    
Service:
    ros2 service call /vision/process_pipeline std_srvs/srv/Trigger

Dependencies:
    - Camera must be publishing to /camera/image_raw
    - No external SAM detector needed (integrated)
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
from typing import List, Dict, Tuple
import time
from pathlib import Path

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
    Integrated SAM + CLIP Pipeline for object detection and classification
    
    Subscribes to:
        - /camera/image_raw (RGB images - captures ONE frame)
    
    Services:
        - /vision/process_pipeline (Main integrated pipeline service)
    
    Output:
        - Saves JSON file with detection and classification results
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
        
        # CLIP model
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output directory for JSON results
        self.output_dir = Path.home() / "sam_clip_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # OpenCV window for visualization
        self.window_name = "SAM + CLIP Pipeline Results"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 750)
        
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
        
        # Create main integrated pipeline service
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/process_pipeline',
            self.pipeline_callback
        )
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 Integrated SAM + CLIP Pipeline Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 Subscribed to: /camera/image_raw (will capture ONE frame)")
        self.get_logger().info("🤖 CLIP Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"💻 Device: {self.device}")
        self.get_logger().info(f"🏷️  Labels: {', '.join(self.candidate_labels)}")
        self.get_logger().info(f"📁 Output Directory: {self.output_dir}")
        self.get_logger().info(f"�️  OpenCV Window: '{self.window_name}'")
        self.get_logger().info("�🔧 Service: /vision/process_pipeline")
        self.get_logger().info("   └─ Runs SAM detection + CLIP classification + JSON output")
        self.get_logger().info("=" * 80)
        self.get_logger().info("💡 Usage: ros2 service call /vision/process_pipeline std_srvs/srv/Trigger")
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
        Main integrated pipeline service callback
        
        Steps:
        1. Check if frame is captured
        2. Run SAM detection directly (OpenCV-based segmentation)
        3. Classify each region with CLIP
        4. Save results as JSON file
        5. Return combined results
        """
        try:
            # Step 0: Verify frame is captured
            if not self.frame_captured or self.captured_frame is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "sam_clip_integrated",
                    "success": False,
                    "error": "No frame captured yet. Waiting for /camera/image_raw...",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No frame captured yet")
                return response
            
            # Step 1: Run SAM detection directly
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 1/3: Running SAM Detection (Integrated)")
            self.get_logger().info("=" * 80)
            
            detections, bboxes = self._detect_objects_sam(self.captured_frame)
            
            if not bboxes:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "sam_clip_integrated",
                    "success": False,
                    "error": "No objects detected in image",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No objects detected")
                return response
            
            self.get_logger().info(f"✅ Detected {len(bboxes)} objects")
            
            # Step 1.5: Display SAM detection results (before CLIP)
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 1.5/4: Displaying SAM Detection Results")
            self.get_logger().info("=" * 80)
            
            self._display_sam_detections(detections, bboxes)
            
            # Step 2: Classify each region with CLIP
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"🔍 STEP 2/4: Classifying {len(bboxes)} Regions with CLIP")
            self.get_logger().info("=" * 80)
            
            classification_results = self._classify_regions_with_clip(bboxes, detections)
            
            # Step 3: Save results as JSON file
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 3/5: Saving Results as JSON")
            self.get_logger().info("=" * 80)
            
            json_path = self._save_json_output(classification_results)
            
            # Add file path to response
            classification_results['output']['json_file'] = str(json_path)
            
            # Step 4: Display visualization
            self.get_logger().info("=" * 80)
            self.get_logger().info("🔍 STEP 4/5: Displaying Final Classification Results")
            self.get_logger().info("=" * 80)
            
            self._display_results(detections, bboxes, classification_results)
            
            # Build final response
            response.success = True
            response.message = json.dumps(classification_results, indent=2)
            
            # Step 5: Print JSON to terminal
            self.get_logger().info("=" * 80)
            self.get_logger().info("📋 STEP 5/5: JSON OUTPUT")
            self.get_logger().info("=" * 80)
            self.get_logger().info(response.message)
            self.get_logger().info("=" * 80)
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"✅ PIPELINE COMPLETE")
            self.get_logger().info(f"   Detected: {len(bboxes)} objects")
            self.get_logger().info(f"   Classified: {len(classification_results['output']['classified_regions'])} regions")
            self.get_logger().info(f"   JSON saved to: {json_path}")
            self.get_logger().info(f"   Visualization displayed in window: '{self.window_name}'")
            self.get_logger().info("=" * 80)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "pipeline": "sam_clip_integrated",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"❌ Pipeline error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def _detect_objects_sam(self, rgb_image: np.ndarray) -> Tuple[List[Dict], List[List[int]]]:
        """
        Detect objects using OpenCV contour detection (SAM-style segmentation)
        Integrated directly in the pipeline - no external service needed
        
        Args:
            rgb_image: BGR image from OpenCV
            
        Returns:
            Tuple of (detections_list, bboxes_list)
            - detections: List of detection dictionaries with full info
            - bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        """
        if rgb_image is None:
            return [], []
        
        h, w = rgb_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better object separation
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours (external only)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter parameters
        min_area = (w * h) * 0.001  # Minimum 0.1% of image
        max_area = (w * h) * 0.8    # Maximum 80% of image
        min_box_size = 20           # Minimum bounding box dimension
        
        detections = []
        bboxes = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter small boxes
            if w_box < min_box_size or h_box < min_box_size:
                continue
            
            # Create binary mask for this object
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            confidence = min(0.95, 0.60 + circularity * 0.35)
            
            bbox = [x, y, x + w_box, y + h_box]
            
            detection = {
                "id": f"obj_{i}",
                "class_name": "object",  # Will be classified by CLIP
                "confidence": float(confidence),
                "bbox": bbox,
                "center": [x + w_box // 2, y + h_box // 2],
                "area": int(area),
                "mask": mask,
                "contour": contour
            }
            
            detections.append(detection)
            bboxes.append(bbox)
        
        self.get_logger().info(f"   SAM Detection: Found {len(detections)} objects")
        for i, det in enumerate(detections):
            self.get_logger().info(f"      [{i}] bbox={det['bbox']}, confidence={det['confidence']:.2f}")
        
        return detections, bboxes
    
    def _display_sam_detections(self, detections: List[Dict], bboxes: List[List[int]]):
        """
        Display SAM detection results (before CLIP classification)
        
        Args:
            detections: List of detection dictionaries
            bboxes: List of bounding boxes
        """
        # Create visualization image
        vis_image = self.captured_frame.copy()
        h, w = vis_image.shape[:2]
        
        # Color for SAM detections (green)
        sam_color = (0, 255, 0)
        
        # Draw each detection
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), sam_color, 3)
            
            # Draw filled mask with transparency
            if 'mask' in det:
                mask = det['mask']
                colored_mask = np.zeros_like(vis_image)
                colored_mask[:, :] = sam_color
                vis_image = np.where(
                    mask[..., None] > 0,
                    cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0),
                    vis_image
                )
            
            # Prepare label text
            label = f"Detection #{idx}"
            conf_text = f"Conf: {det.get('confidence', 0.0):.2f}"
            area_text = f"Area: {det.get('area', 0)}"
            
            # Calculate label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            area_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            max_width = max(label_size[0], conf_size[0], area_size[0])
            total_height = label_size[1] + conf_size[1] + area_size[1] + 25
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - total_height - 10),
                (x1 + max_width + 20, y1),
                sam_color,
                -1
            )
            
            # Draw label text
            y_offset = y1 - total_height + 5
            
            # Detection ID
            cv2.putText(vis_image, label, (x1 + 10, y_offset + label_size[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += label_size[1] + 8
            
            # Confidence
            cv2.putText(vis_image, conf_text, (x1 + 10, y_offset + conf_size[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += conf_size[1] + 5
            
            # Area
            cv2.putText(vis_image, area_text, (x1 + 10, y_offset + area_size[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add title bar
        title_text = f"SAM Detection Results | Objects: {len(detections)} | BEFORE CLIP Classification"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw title background
        cv2.rectangle(vis_image, (0, 0), (min(w, title_size[0] + 30), 50), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (0, 0), (min(w, title_size[0] + 30), 50), sam_color, 3)
        
        # Draw title text
        cv2.putText(vis_image, title_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add instruction at bottom
        instruction = "SAM detected these objects. Press any key to continue to CLIP classification..."
        inst_y = h - 30
        cv2.rectangle(vis_image, (0, inst_y - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(vis_image, instruction, (20, inst_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save SAM detection image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sam_vis_path = self.output_dir / f"sam_detection_{timestamp}.jpg"
        cv2.imwrite(str(sam_vis_path), vis_image)
        self.get_logger().info(f"   ✅ SAM detection visualization saved: {sam_vis_path}")
        
        # Display in OpenCV window
        cv2.imshow(self.window_name, vis_image)
        self.get_logger().info(f"   ✅ SAM detections displayed. Press any key to continue to CLIP...")
        cv2.waitKey(0)  # Wait for key press before continuing to CLIP
        
        self.get_logger().info(f"   ✅ Proceeding to CLIP classification...")
    
    def _save_json_output(self, results: Dict) -> Path:
        """
        Save pipeline results as JSON file
        
        Args:
            results: Dictionary with pipeline results
            
        Returns:
            Path to saved JSON file
        """
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"sam_clip_results_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        # Save JSON file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.get_logger().info(f"   ✅ JSON saved: {json_path}")
        
        return json_path
    
    def _display_results(self, detections: List[Dict], bboxes: List[List[int]], classification_results: Dict):
        """
        Display detection and classification results on image
        
        Args:
            detections: List of detection dictionaries
            bboxes: List of bounding boxes
            classification_results: Classification results dictionary
        """
        # Create visualization image
        vis_image = self.captured_frame.copy()
        h, w = vis_image.shape[:2]
        
        # Get classified regions from results
        classified_regions = classification_results.get('output', {}).get('classified_regions', [])
        
        # Color palette for different objects
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light Green
            (255, 128, 0),  # Orange
        ]
        
        # Draw each detection with classification
        for idx, (det, region) in enumerate(zip(detections, classified_regions)):
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get color for this object
            color = colors[idx % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw filled mask with transparency
            if 'mask' in det:
                mask = det['mask']
                colored_mask = np.zeros_like(vis_image)
                colored_mask[:, :] = color
                vis_image = np.where(
                    mask[..., None] > 0,
                    cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0),
                    vis_image
                )
            
            # Prepare label text
            top_pred = region['top_prediction']
            label = f"#{idx}: {top_pred['label']}"
            conf_text = f"CLIP: {top_pred['confidence']:.2f}"
            sam_conf_text = f"SAM: {region.get('sam_confidence', 0.0):.2f}"
            
            # Calculate label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            sam_size = cv2.getTextSize(sam_conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            max_width = max(label_size[0], conf_size[0], sam_size[0])
            total_height = label_size[1] + conf_size[1] + sam_size[1] + 30
            
            # Draw label background (semi-transparent)
            overlay = vis_image.copy()
            cv2.rectangle(
                overlay,
                (x1, y1 - total_height - 10),
                (x1 + max_width + 20, y1),
                color,
                -1
            )
            vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
            
            # Draw border for label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - total_height - 10),
                (x1 + max_width + 20, y1),
                color,
                2
            )
            
            # Draw label text (white with black outline)
            y_offset = y1 - total_height + 5
            
            # Object ID and class label
            for thickness, text_color in [(4, (0, 0, 0)), (2, (255, 255, 255))]:
                cv2.putText(
                    vis_image,
                    label,
                    (x1 + 10, y_offset + label_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    thickness
                )
            
            y_offset += label_size[1] + 10
            
            # CLIP confidence
            for thickness, text_color in [(4, (0, 0, 0)), (2, (255, 255, 255))]:
                cv2.putText(
                    vis_image,
                    conf_text,
                    (x1 + 10, y_offset + conf_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    thickness
                )
            
            y_offset += conf_size[1] + 5
            
            # SAM confidence
            for thickness, text_color in [(4, (0, 0, 0)), (2, (255, 255, 255))]:
                cv2.putText(
                    vis_image,
                    sam_conf_text,
                    (x1 + 10, y_offset + sam_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    thickness
                )
        
        # Add title bar
        title_text = f"SAM + CLIP Pipeline | Frame: {self.frame_counter} | Objects: {len(detections)}"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        
        # Draw title background
        cv2.rectangle(vis_image, (0, 0), (title_size[0] + 30, 50), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (0, 0), (title_size[0] + 30, 50), (0, 255, 0), 3)
        
        # Draw title text
        cv2.putText(
            vis_image,
            title_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add legend at bottom
        legend_y = h - 100
        legend_text = "Legend:"
        cv2.putText(vis_image, legend_text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        legend_y += 30
        for idx, region in enumerate(classified_regions[:5]):  # Show first 5 in legend
            color = colors[idx % len(colors)]
            legend_item = f"#{idx}: {region['top_prediction']['label']} ({region['top_prediction']['confidence']:.2f})"
            
            # Draw color box
            cv2.rectangle(vis_image, (10, legend_y - 15), (30, legend_y), color, -1)
            cv2.rectangle(vis_image, (10, legend_y - 15), (30, legend_y), (255, 255, 255), 1)
            
            # Draw legend text
            cv2.putText(vis_image, legend_item, (40, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            legend_y += 25
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = self.output_dir / f"visualization_{timestamp}.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        self.get_logger().info(f"   ✅ Visualization saved: {vis_path}")
        
        # Display in OpenCV window
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)  # Refresh window
        
        self.get_logger().info(f"   ✅ Visualization displayed in window: '{self.window_name}'")
        self.get_logger().info(f"   💡 Press any key in the window to close it")
    
    def _classify_regions_with_clip(self, bboxes: List[List[int]], detections: List[Dict]) -> Dict:
        """
        Classify each detected region using CLIP
        
        Args:
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            detections: List of detection dictionaries with additional info
            
        Returns:
            Dictionary with classification results
        """
        if not CLIP_AVAILABLE or self.model is None:
            self.get_logger().error("❌ CLIP model not available")
            return {
                "pipeline": "sam_clip_integrated",
                "success": False,
                "error": "CLIP model not available",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        start_time = time.time()
        classified_regions = []
        
        h, w = self.captured_frame.shape[:2]
        
        for region_id, (bbox, det) in enumerate(zip(bboxes, detections)):
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
            
            # Build region result with SAM detection info
            region_result = {
                "region_id": region_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "sam_confidence": round(det.get("confidence", 0.0), 2),
                "area": det.get("area", 0),
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
            "pipeline": "sam_clip_integrated",
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
                "device": self.device,
                "output_directory": str(self.output_dir)
            }
        }
        
        return schema
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
