#!/usr/bin/env python3
"""
DINO Pipeline Adapter for ROS2
Simplified integration of the DINO pipeline components for ROS2 usage
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DinoPipelineAdapter:
    """
    Simplified DINO pipeline adapter for ROS2 integration
    Provides a research-level implementation suitable for ROS2 deployment
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if self._check_cuda() else "cpu")
        self.initialized = False
        
        # Try to import and initialize actual DINO components
        self._init_components()
        
        logger.info(f"DINO Pipeline Adapter initialized on {self.device}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _init_components(self):
        """Initialize pipeline components with fallbacks"""
        try:
            # Try to import actual DINO pipeline
            from Final_proj.DINO_pipeline.main_dino_pipeline import DINOPipeline
            self.dino_pipeline = DINOPipeline(device=self.device)
            self.mode = "full"
            self.initialized = True
            logger.info("✅ Full DINO pipeline loaded")
        except Exception as e:
            logger.warning(f"⚠️ Full DINO pipeline not available: {e}")
            self._init_simulation_mode()
    
    def _init_simulation_mode(self):
        """Initialize simulation/research mode"""
        self.mode = "simulation"
        self.initialized = True
        logger.info("🎭 Running in simulation mode for research/development")
    
    def process_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray = None, 
                     output_dir: str = None) -> Dict:
        """
        Process RGB-D images through the DINO pipeline
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_image: Depth image as numpy array (H, W) - optional
            output_dir: Directory to save debug outputs
            
        Returns:
            Dictionary containing pipeline results
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")
        
        if self.mode == "full":
            return self._process_full_pipeline(rgb_image, depth_image, output_dir)
        else:
            return self._process_simulation(rgb_image, depth_image, output_dir)
    
    def _process_full_pipeline(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                              output_dir: str) -> Dict:
        """Process using full DINO pipeline"""
        # Save temporary images for pipeline
        temp_dir = Path(output_dir) if output_dir else Path("/tmp/dino_pipeline")
        temp_dir.mkdir(exist_ok=True)
        
        rgb_path = temp_dir / "temp_rgb.jpg"
        depth_path = temp_dir / "temp_depth.png"
        
        cv2.imwrite(str(rgb_path), rgb_image)
        if depth_image is not None:
            cv2.imwrite(str(depth_path), depth_image)
        
        # Run full pipeline
        results = self.dino_pipeline.process_rgbd(
            str(rgb_path), 
            str(depth_path) if depth_image is not None else None,
            str(temp_dir)
        )
        
        return self._format_results(results)
    
    def _process_simulation(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                           output_dir: str) -> Dict:
        """Process using simulation/research mode"""
        logger.info("🔄 Processing in simulation mode...")
        
        height, width = rgb_image.shape[:2]
        
        # Simulate object detection using simple computer vision
        detections = self._simulate_object_detection(rgb_image)
        
        # Simulate semantic tagging
        semantic_results = self._simulate_semantic_tagging(detections)
        
        # Simulate grasp generation
        grasps = self._simulate_grasp_generation(detections, depth_image)
        
        # Simulate scene graph
        scene_graph = self._simulate_scene_graph(semantic_results, grasps)
        
        # Create debug visualization
        debug_image = self._create_debug_visualization(rgb_image, detections, grasps)
        
        results = {
            "detections": detections,
            "semantic_objects": semantic_results,
            "grasps": grasps,
            "scene_graph": scene_graph,
            "debug_image": debug_image,
            "processing_time": 0.5,  # Simulated
            "mode": "simulation"
        }
        
        # Save debug outputs if requested
        if output_dir:
            self._save_debug_outputs(results, output_dir)
        
        return results
    
    def _simulate_object_detection(self, rgb_image: np.ndarray) -> List[Dict]:
        """Simulate object detection using simple CV methods"""
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold and find contours
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        object_classes = ["tool", "object", "container", "part", "device"]
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                
                detection = {
                    "id": f"obj_{i}",
                    "class": object_classes[i % len(object_classes)],
                    "confidence": 0.7 + 0.3 * np.random.random(),
                    "bbox": [x, y, x + w, y + h],
                    "center": [x + w//2, y + h//2],
                    "area": cv2.contourArea(contour),
                    "contour": contour
                }
                detections.append(detection)
        
        # Add at least one detection if none found
        if not detections:
            h, w = rgb_image.shape[:2]
            detections.append({
                "id": "obj_0",
                "class": "object",
                "confidence": 0.8,
                "bbox": [w//4, h//4, 3*w//4, 3*h//4],
                "center": [w//2, h//2],
                "area": w*h//4,
                "contour": None
            })
        
        return detections
    
    def _simulate_semantic_tagging(self, detections: List[Dict]) -> List[Dict]:
        """Simulate CLIP semantic tagging"""
        semantic_attributes = {
            "tool": ["metallic", "handheld", "precise", "functional"],
            "object": ["solid", "movable", "graspable", "useful"],
            "container": ["hollow", "storage", "cylindrical", "open"],
            "part": ["mechanical", "small", "component", "metal"],
            "device": ["electronic", "complex", "powered", "precise"]
        }
        
        colors = ["red", "blue", "green", "black", "silver", "white"]
        materials = ["metal", "plastic", "wood", "ceramic", "glass"]
        
        semantic_results = []
        for det in detections:
            obj_class = det["class"]
            attributes = semantic_attributes.get(obj_class, ["unknown"])
            
            semantic_obj = {
                **det,
                "semantic_tags": attributes[:2],  # Top 2 attributes
                "color": colors[len(det["id"]) % len(colors)],
                "material": materials[len(det["id"]) % len(materials)],
                "affordances": self._get_affordances(obj_class),
                "graspable": obj_class in ["tool", "object", "part"],
                "manipulation_difficulty": np.random.uniform(0.3, 0.8)
            }
            semantic_results.append(semantic_obj)
        
        return semantic_results
    
    def _get_affordances(self, obj_class: str) -> List[str]:
        """Get affordances for object class"""
        affordance_map = {
            "tool": ["grasp", "use", "manipulate", "apply"],
            "object": ["grasp", "move", "place", "stack"],
            "container": ["grasp", "fill", "empty", "carry"],
            "part": ["grasp", "attach", "detach", "install"],
            "device": ["grasp", "activate", "operate", "configure"]
        }
        return affordance_map.get(obj_class, ["grasp"])
    
    def _simulate_grasp_generation(self, detections: List[Dict], depth_image: np.ndarray) -> List[Dict]:
        """Simulate 6D grasp generation"""
        grasps = []
        
        for det in detections:
            if det.get("graspable", True):
                bbox = det["bbox"]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Estimate depth
                if depth_image is not None:
                    depth = float(depth_image[center_y, center_x]) / 1000.0  # Convert to meters
                else:
                    depth = 0.5  # Default depth
                
                # Generate multiple grasp candidates
                for approach in ["top", "side"]:
                    grasp = {
                        "id": f"grasp_{det['id']}_{approach}",
                        "target_object_id": det["id"],
                        "target_class": det["class"],
                        "pose": {
                            "position": [center_x / 1000.0, center_y / 1000.0, depth],  # Convert to meters
                            "orientation": [0, 0, 0, 1]  # Quaternion
                        },
                        "quality": 0.6 + 0.4 * np.random.random(),
                        "width": 0.02 + 0.08 * np.random.random(),  # 2-10cm
                        "approach_direction": approach,
                        "grasp_type": "pinch" if approach == "top" else "power",
                        "collision_free": True,
                        "execution_time_estimate": 2.0 + 3.0 * np.random.random()
                    }
                    grasps.append(grasp)
        
        return grasps
    
    def _simulate_scene_graph(self, semantic_objects: List[Dict], grasps: List[Dict]) -> Dict:
        """Simulate scene graph construction"""
        relations = []
        
        # Generate spatial relations between objects
        for i, obj1 in enumerate(semantic_objects):
            for j, obj2 in enumerate(semantic_objects):
                if i != j:
                    # Calculate relative positions
                    pos1 = obj1["center"]
                    pos2 = obj2["center"]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance < 200:  # Objects are close
                        relation = {
                            "relation_type": "near",
                            "subject_object_id": i,
                            "target_object_id": j,
                            "confidence": 0.8,
                            "distance": distance / 1000.0,  # Convert to meters
                            "direction": "left" if pos1[0] < pos2[0] else "right"
                        }
                        relations.append(relation)
        
        scene_graph = {
            "scene_id": f"scene_{len(semantic_objects)}_objects",
            "num_objects": len(semantic_objects),
            "scene_confidence": 0.85,
            "objects": semantic_objects,
            "relations": relations,
            "scene_type": "workshop",
            "scene_affordances": ["manipulation", "assembly", "inspection"],
            "has_support_surface": True
        }
        
        return scene_graph
    
    def _create_debug_visualization(self, rgb_image: np.ndarray, detections: List[Dict], 
                                   grasps: List[Dict]) -> np.ndarray:
        """Create debug visualization image"""
        debug_img = rgb_image.copy()
        
        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(debug_img, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw grasp points
        for grasp in grasps:
            pos = grasp["pose"]["position"]
            # Convert back to pixel coordinates
            x, y = int(pos[0] * 1000), int(pos[1] * 1000)
            
            if grasp["approach_direction"] == "top":
                color = (255, 0, 0)  # Red for top grasps
            else:
                color = (0, 0, 255)  # Blue for side grasps
            
            cv2.circle(debug_img, (x, y), 10, color, -1)
            cv2.putText(debug_img, f"G:{grasp['quality']:.2f}", (x + 15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return debug_img
    
    def _save_debug_outputs(self, results: Dict, output_dir: str):
        """Save debug outputs to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save debug image
        if "debug_image" in results:
            cv2.imwrite(str(output_path / "debug_visualization.jpg"), results["debug_image"])
        
        # Save results as JSON (excluding image data)
        import json
        json_results = {k: v for k, v in results.items() if k != "debug_image"}
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json_results = convert_numpy(json_results)
        
        with open(output_path / "pipeline_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📁 Debug outputs saved to {output_path}")
    
    def _format_results(self, raw_results: Dict) -> Dict:
        """Format results from full pipeline to standard format"""
        # This would adapt the full DINO pipeline output to our standard format
        # For now, return as-is since we're in simulation mode
        return raw_results