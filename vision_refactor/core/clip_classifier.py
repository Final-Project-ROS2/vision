#!/usr/bin/env python3
"""
Refactored CLIP Classifier - Clean and Focused

Provides image classification using CLIP models.
Automatically classifies regions detected by SAM detector.

Services:
    /vision/classify_all - Classify entire image
    /vision/classify_bb - Classify specific bounding box
    /vision/classify_bbox_filtered - Get filtered high-confidence results
    /vision/find_object - Find objects by label name

Subscriber:
    /vision/sam_detections - Automatic classification of detected regions
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
import cv2
import numpy as np
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# CLIP imports
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, AutoProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install: pip install torch transformers pillow")

# Custom interfaces
try:
    from custom_interfaces.srv import ClassifyBBox, FindObject
    from custom_interfaces.msg import SAMDetections, CLIPClassification
    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False
    print("Custom interfaces not available. Limited functionality.")

from vision_refactor.utils.common import VisionNodeBase, OpenCVWindow, draw_bbox


class CLIPClassifier(VisionNodeBase):
    """
    Simplified CLIP-based image classifier
    
    Classifies images and image regions using OpenAI's CLIP model.
    Integrates with SAM detector for automatic region classification.
    """
    
    def __init__(self):
        super().__init__('clip_classifier')
        
        # Default classification labels
        self.candidate_labels = [
            "object", "tool", "part", "component", "device",
            "cup", "bottle", "can", "bowl", "plate",
            "box", "container", "package", "bag",
            "book", "paper", "document", "folder",
            "phone", "remote", "keyboard", "mouse",
            "screwdriver", "wrench", "hammer", "pliers",
            "bolt", "screw", "nut", "washer",
            "wire", "cable", "connector", "plug",
            "battery", "charger", "adapter",
            "ball", "toy", "game", "controller",
            "apple", "orange", "banana", "food",
            "piston_rod", "gear", "bearing", "spring",
            "motor", "sensor", "circuit", "board"
        ]
        
        # Classification state
        self.latest_classification: Optional[Dict] = None
        self.latest_region_classifications: List[Dict] = []
        self.frame_counter = 0
        
        # CLIP model
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Visualization
        self.window = OpenCVWindow("CLIP Classifier", 1000, 700)
        
        # Initialize CLIP
        self.init_clip_model()
        
        # Setup ROS components
        self.setup_camera_subscriptions()
        self.setup_services()
        self.setup_subscribers()
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("CLIP Classifier initialized")
        self.get_logger().info(f"Model: openai/clip-vit-base-patch32")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Labels: {len(self.candidate_labels)} categories")
    
    def init_clip_model(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            self.get_logger().error("CLIP not available. Classification disabled.")
            return
        
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.get_logger().info(f"CLIP model loaded on {self.device}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CLIP model: {e}")
            self.model = None
    
    def setup_services(self):
        """Create classification services"""
        self.classify_all_service = self.create_service(
            Trigger,
            '/vision/classify_all',
            self.classify_all_callback,
            callback_group=self.callback_group
        )
        
        if CUSTOM_INTERFACES:
            self.classify_bb_service = self.create_service(
                ClassifyBBox,
                '/vision/classify_bb',
                self.classify_bb_callback,
                callback_group=self.callback_group
            )
            
            self.find_object_service = self.create_service(
                FindObject,
                '/vision/find_object',
                self.find_object_callback,
                callback_group=self.callback_group
            )
        
        self.classify_filtered_service = self.create_service(
            Trigger,
            '/vision/classify_bbox_filtered',
            self.classify_filtered_callback,
            callback_group=self.callback_group
        )
    
    def setup_subscribers(self):
        """Subscribe to SAM detections for automatic classification"""
        if CUSTOM_INTERFACES:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                self.service_qos
            )
    
    def classify_all_callback(self, request, response):
        """Classify entire image"""
        try:
            if not self.has_camera_data() or self.model is None:
                response.success = False
                response.message = "No camera data or CLIP model unavailable"
                return response
            
            # Classify image
            result = self.classify_image(self.latest_rgb)
            self.latest_classification = result
            
            # Format response
            top_pred = result['output']['top_prediction']
            response.success = True
            response.message = json.dumps({
                'label': top_pred['label'],
                'confidence': top_pred['confidence'],
                'processing_time_ms': result['output']['metadata']['processing_time_ms']
            })
            
            self.get_logger().info(f"Image classification: {top_pred['label']} ({top_pred['confidence']:.3f})")
            
        except Exception as e:
            response.success = False
            response.message = f"Classification failed: {str(e)}"
            self.get_logger().error(f"Classification error: {e}")
        
        return response
    
    def classify_bb_callback(self, request, response):
        """Classify specific bounding box region"""
        try:
            if not self.has_camera_data() or self.model is None:
                response.success = False
                response.message = "No camera data or CLIP model unavailable"
                return response
            
            bbox = [request.x1, request.y1, request.x2, request.y2]
            result = self.classify_region(self.latest_rgb, bbox)
            
            # Format response
            top_pred = result['top_prediction']
            response.success = True
            response.label = top_pred['label']
            response.confidence = top_pred['confidence']
            response.message = json.dumps(result)
            
        except Exception as e:
            response.success = False
            response.message = f"Classification failed: {str(e)}"
            
        return response
    
    def classify_filtered_callback(self, request, response):
        """Return high-confidence classifications from recent SAM detections"""
        try:
            filtered_results = []
            confidence_threshold = 0.5
            
            for result in self.latest_region_classifications:
                if result.get('confidence', 0) > confidence_threshold:
                    filtered_results.append(result)
            
            response.success = True
            response.message = json.dumps({
                'filtered_classifications': filtered_results,
                'total_regions': len(self.latest_region_classifications),
                'high_confidence_count': len(filtered_results),
                'confidence_threshold': confidence_threshold
            })
            
        except Exception as e:
            response.success = False
            response.message = f"Filter failed: {str(e)}"
        
        return response
    
    def find_object_callback(self, request, response):
        """Find object by label name using embedding similarity"""
        try:
            if not self.has_camera_data() or self.model is None:
                response.success = False
                response.message = "No camera data or CLIP model unavailable"
                return response
            
            target_label = request.label.lower().strip()
            best_bbox = None
            best_confidence = 0.0
            
            # Search through recent classifications
            for result in self.latest_region_classifications:
                if target_label in result.get('label', '').lower():
                    if result.get('confidence', 0) > best_confidence:
                        best_confidence = result['confidence']
                        best_bbox = result.get('bbox')
            
            if best_bbox:
                response.success = True
                response.bbox = best_bbox
                response.confidence = best_confidence
                response.message = f"Found {target_label} with confidence {best_confidence:.3f}"
            else:
                response.success = False
                response.message = f"Object '{target_label}' not found"
            
        except Exception as e:
            response.success = False
            response.message = f"Search failed: {str(e)}"
        
        return response
    
    def sam_detections_callback(self, msg: SAMDetections):
        """Automatically classify SAM detections"""
        try:
            if not self.has_camera_data() or self.model is None:
                return
            
            self.latest_region_classifications = []
            
            for detection in msg.detections:
                # Classify each detected region
                result = self.classify_region(self.latest_rgb, detection.bbox)
                
                # Store result with detection info
                classification = {
                    'object_id': detection.object_id,
                    'bbox': detection.bbox,
                    'label': result['top_prediction']['label'],
                    'confidence': result['top_prediction']['confidence'],
                    'detection_confidence': detection.confidence,
                    'area': detection.area,
                    'center': detection.center
                }
                
                self.latest_region_classifications.append(classification)
            
            self.get_logger().info(f"Classified {len(msg.detections)} detected regions")
            
        except Exception as e:
            self.get_logger().error(f"SAM classification error: {e}")
    
    def classify_image(self, image: np.ndarray) -> Dict:
        """Classify entire image using CLIP"""
        if self.model is None:
            return {'error': 'CLIP model not available'}
        
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        
        # Process with CLIP
        inputs = self.processor(
            text=self.candidate_labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        # Convert to numpy and sort
        probs_np = probs.cpu().numpy()[0]
        sorted_indices = np.argsort(probs_np)[::-1]
        
        # Build predictions
        all_predictions = []
        for idx in sorted_indices[:10]:  # Top 10
            all_predictions.append({
                'label': self.candidate_labels[idx],
                'confidence': float(probs_np[idx])
            })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'pipeline': 'clip_full_image',
            'model': 'openai/clip-vit-base-patch32',
            'output': {
                'top_prediction': all_predictions[0],
                'all_predictions': all_predictions,
                'metadata': {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'processing_time_ms': processing_time,
                    'device': self.device
                }
            }
        }
    
    def classify_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """Classify specific image region"""
        if self.model is None:
            return {'error': 'CLIP model not available'}
        
        try:
            # Extract region
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Clamp to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return {'error': 'Invalid bounding box'}
            
            # Crop region
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                return {'error': 'Empty region'}
            
            # Classify region
            rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_region)
            
            inputs = self.processor(
                text=self.candidate_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            probs_np = probs.cpu().numpy()[0]
            sorted_indices = np.argsort(probs_np)[::-1]
            
            # Build result
            predictions = []
            for idx in sorted_indices[:5]:  # Top 5
                predictions.append({
                    'label': self.candidate_labels[idx],
                    'confidence': float(probs_np[idx])
                })
            
            return {
                'bbox': bbox,
                'top_prediction': predictions[0],
                'all_predictions': predictions
            }
            
        except Exception as e:
            return {'error': f'Classification failed: {str(e)}'}
    
    def visualization_callback(self):
        """Display classification results"""
        if not self.has_camera_data():
            return
        
        vis_image = self.latest_rgb.copy()
        
        # Draw region classifications
        for result in self.latest_region_classifications:
            bbox = result.get('bbox')
            if bbox:
                confidence = result.get('confidence', 0)
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)  # Green/Orange
                
                label = f"{result.get('label', 'unknown')}"
                vis_image = draw_bbox(vis_image, bbox, label, confidence, color)
        
        # Add overall classification info
        if self.latest_classification:
            top_pred = self.latest_classification['output']['top_prediction']
            info_text = f"Scene: {top_pred['label']} ({top_pred['confidence']:.3f})"
            cv2.putText(vis_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add region count
        region_text = f"Regions: {len(self.latest_region_classifications)}"
        cv2.putText(vis_image, region_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.window.show(vis_image)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.window.close()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = CLIPClassifier()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            node.destroy_node()
    
    except Exception as e:
        print(f"Failed to start CLIP Classifier: {e}")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()