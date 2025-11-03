#!/usr/bin/env python3
"""
CLIP Vision Classifier Node
Focused CLIP-based image classification from /camera/image_raw

Usage:
    ros2 run vision clip_classifier
    ros2 run vision clip_classifier --labels "cat,dog,car,airplane"
    
Service:
    ros2 service call /vision/classify_image std_srvs/srv/Trigger
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

# Try to import CLIP/transformers
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Install: pip install torch transformers pillow")


class CLIPClassifier(Node):
    """
    CLIP-based image classifier for ROS2
    
    Subscribes to:
        - /camera/image_raw (RGB images from Gazebo camera)
    
    Services:
        - /vision/classify_image (Trigger classification with JSON output)
    
    Display:
        - Shows live camera feed with top prediction in OpenCV window
    """
    
    def __init__(self, candidate_labels: List[str] = None):
        super().__init__('clip_classifier')
        
        # Default labels if none provided
        self.candidate_labels = candidate_labels or [
            "robot", "tool", "part", "container", "table", 
            "box", "cube", "cylinder", "person", "hand"
        ]
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest image from camera
        self.latest_rgb = None
        self.latest_classification = None
        self.frame_counter = 0
        
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
        
        # Classification service
        self.classification_service = self.create_service(
            Trigger,
            '/vision/classify_image',
            self.classify_service_callback
        )
        
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
        self.get_logger().info(f"🔧 Service: /vision/classify_image")
        self.get_logger().info(f"👁️  OpenCV Window: '{self.window_name}'")
        self.get_logger().info("💡 Call service: ros2 service call /vision/classify_image std_srvs/srv/Trigger")
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("❌ CLIP not available! Install: pip install torch transformers pillow")
            return
        
        try:
            self.get_logger().info("🔧 Loading CLIP model...")
            model_name = "openai/clip-vit-base-patch32"
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
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
                
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def classify_service_callback(self, request, response):
        """Service callback for /vision/classify_image"""
        try:
            if self.latest_rgb is None:
                response.success = False
                response.message = json.dumps({
                    "pipeline": "single_clip",
                    "success": False,
                    "error": "No image available from /camera/image_raw",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("⚠️ No image received yet")
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
            
            self.get_logger().info("🔍 Running CLIP classification...")
            
            # Run classification
            classification_data = self._classify_image(self.latest_rgb)
            self.latest_classification = classification_data
            
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
    
    def visualization_callback(self):
        """Display camera feed with classification in OpenCV window"""
        if self.latest_rgb is None:
            # Show waiting message
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank, 
                "Waiting for /camera/image_raw...", 
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        # Create visualization image
        vis_image = self.latest_rgb.copy()
        h, w = vis_image.shape[:2]
        
        # Draw classification overlay
        if self.latest_classification:
            top_pred = self.latest_classification['output']['top_prediction']
            all_preds = self.latest_classification['output']['all_predictions'][:3]  # Top 3
            
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
            
            # Draw top 3 predictions (smaller, on right)
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
                "Call /vision/classify_image to classify",
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
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            vis_image,
            f"CLIP Classifier | Frame: {self.frame_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4
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
