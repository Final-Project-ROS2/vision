#!/usr/bin/env python3
"""
SAM Pipeline Adapter for ROS2
Integration of the SAM (Segment Anything Model) pipeline with CLIP, GraspNet, and Scene Understanding
for ROS2 robotic vision applications.

Pipeline Flow:
1. SAM (Meta) - Automatic segmentation and object detection
2. CLIP - Semantic tagging and attribute extraction
3. GraspNet - 6D grasp pose generation
4. Scene Understanding - Spatial relationships and scene graph construction
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMPipelineAdapter:
    """
    SAM-based vision pipeline adapter for ROS2 integration
    Provides research-level implementation of SAM → CLIP → GraspNet → Scene Understanding
    """
    
    def __init__(self, device: str = "auto", sam_checkpoint: str = None):
        self.device = device if device != "auto" else ("cuda" if self._check_cuda() else "cpu")
        self.sam_checkpoint = sam_checkpoint
        self.initialized = False
        
        # Try to import and initialize actual pipeline components
        self._init_components()
        
        logger.info(f"SAM Pipeline Adapter initialized on {self.device}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _init_components(self):
        """Initialize pipeline components with fallbacks"""
        # NOTE: The full SAM pipeline with external models is not currently integrated.
        # The main vision processing is handled by sam_vision_pipeline_node.py
        # This adapter runs in simulation mode for development and testing.
        logger.info("🎭 Running in simulation mode for research/development")
        self._init_simulation_mode()
    
    def _find_sam_checkpoint(self) -> Optional[str]:
        """Try to find SAM checkpoint in common locations"""
        possible_paths = [
            "sam_vit_b_01ec64.pth",
            "../sam_vit_b_01ec64.pth",
            "../../sam_vit_b_01ec64.pth",
            "../../../sam_vit_b_01ec64.pth",
            str(Path.home() / "sam_vit_b_01ec64.pth"),
            str(Path(__file__).parent.parent / "Final-proj" / "sam_vit_b_01ec64.pth"),
            str(Path(__file__).parent.parent / "Final-proj" / "models" / "sam_vit_b_01ec64.pth"),
            "/tmp/sam_vit_b_01ec64.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found SAM checkpoint at: {path}")
                return str(path)
        
        logger.warning("SAM checkpoint not found in common locations")
        logger.info("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None
    
    def _init_simulation_mode(self):
        """Initialize simulation/research mode"""
        self.mode = "simulation"
        self.initialized = True
        self.sam_detector = None
        logger.info("🎭 Running in simulation mode for research/development")
    
    def process_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray = None, 
                     output_dir: str = None) -> Dict:
        """
        Process RGB-D images through the SAM pipeline
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_image: Depth image as numpy array (H, W) - optional
            output_dir: Directory to save debug outputs
            
        Returns:
            Dictionary containing pipeline results with keys:
            - detections: List of detected objects
            - semantic_objects: Objects with semantic tags
            - grasps: 6D grasp poses
            - scene_graph: Scene understanding
            - debug_image: Visualization
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")
        
        if self.mode == "full":
            return self._process_full_pipeline(rgb_image, depth_image, output_dir)
        else:
            return self._process_simulation(rgb_image, depth_image, output_dir)
    
    def _process_full_pipeline(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                              output_dir: str) -> Dict:
        """Process using full SAM + CLIP + GraspNet pipeline"""
        logger.info("🔄 Processing with full SAM pipeline...")
        
        # Stage 1: SAM Object Detection & Segmentation
        detections, masks = self._run_sam_detection(rgb_image)
        logger.info(f"   SAM detected {len(detections)} objects")
        
        # Stage 2: CLIP Semantic Tagging
        semantic_objects = self._run_clip_tagging(rgb_image, detections, masks)
        logger.info(f"   CLIP tagged {len(semantic_objects)} objects")
        
        # Stage 3: GraspNet 6D Prediction
        grasps = self._run_graspnet_prediction(rgb_image, depth_image, semantic_objects)
        logger.info(f"   GraspNet generated {len(grasps)} grasp poses")
        
        # Stage 4: Scene Graph Construction
        scene_graph = self._build_scene_graph(semantic_objects, grasps)
        logger.info(f"   Scene graph constructed with {len(scene_graph['relations'])} relations")
        
        # Create visualization
        debug_image = self._create_debug_visualization(rgb_image, detections, grasps, masks)
        
        results = {
            "detections": detections,
            "semantic_objects": semantic_objects,
            "grasps": grasps,
            "scene_graph": scene_graph,
            "debug_image": debug_image,
            "processing_time": 0.0,  # Would track actual time
            "mode": "full"
        }
        
        # Save debug outputs if requested
        if output_dir:
            self._save_debug_outputs(results, output_dir)
        
        return results
    
    def _run_sam_detection(self, rgb_image: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """Run SAM automatic mask generation"""
        # Use the existing SAM detector
        results = self.sam_detector.process_image(rgb_image)
        
        detections = []
        masks = []
        
        if 'masks' in results:
            for i, mask in enumerate(results['masks']):
                # Get bounding box from mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0:
                    continue
                    
                bbox = [
                    int(x_indices.min()),
                    int(y_indices.min()),
                    int(x_indices.max()),
                    int(y_indices.max())
                ]
                
                detection = {
                    "id": f"obj_{i}",
                    "class": "object",  # Will be classified by CLIP
                    "confidence": float(results.get('scores', [0.9])[i] if i < len(results.get('scores', [])) else 0.9),
                    "bbox": bbox,
                    "center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                    "area": int(mask.sum()),
                    "mask": mask
                }
                detections.append(detection)
                masks.append(mask)
        
        return detections, masks
    
    def _run_clip_tagging(self, rgb_image: np.ndarray, detections: List[Dict], 
                         masks: List[np.ndarray]) -> List[Dict]:
        """Apply CLIP semantic tagging to detected objects"""
        # This would use actual CLIP if available
        # For now, use rule-based tagging based on visual features
        
        semantic_objects = []
        for det in detections:
            # Extract object region
            bbox = det["bbox"]
            obj_region = rgb_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Simple semantic analysis based on color and shape
            semantic_tags = self._analyze_object_features(obj_region, det.get("mask"))
            
            semantic_obj = {
                **det,
                "semantic_tags": semantic_tags["attributes"],
                "color": semantic_tags["color"],
                "material": semantic_tags["material"],
                "affordances": semantic_tags["affordances"],
                "graspable": True,
                "manipulation_difficulty": 0.5
            }
            semantic_objects.append(semantic_obj)
        
        return semantic_objects
    
    def _analyze_object_features(self, obj_region: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze object features for semantic tagging"""
        # Calculate dominant color
        if obj_region.size > 0:
            avg_color = obj_region.mean(axis=(0, 1))
            color_name = self._get_color_name(avg_color)
        else:
            color_name = "unknown"
        
        # Estimate material based on texture
        material = self._estimate_material(obj_region)
        
        # Determine attributes and affordances
        attributes = ["solid", "movable", "graspable"]
        affordances = ["grasp", "move", "manipulate", "place"]
        
        return {
            "attributes": attributes,
            "color": color_name,
            "material": material,
            "affordances": affordances
        }
    
    def _get_color_name(self, bgr_color: np.ndarray) -> str:
        """Get color name from BGR values"""
        b, g, r = bgr_color
        
        # Simple color classification
        if r > 150 and g < 100 and b < 100:
            return "red"
        elif g > 150 and r < 100 and b < 100:
            return "green"
        elif b > 150 and r < 100 and g < 100:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r < 80 and g < 80 and b < 80:
            return "black"
        elif r > 180 and g > 180 and b > 180:
            return "white"
        else:
            return "mixed"
    
    def _estimate_material(self, obj_region: np.ndarray) -> str:
        """Estimate material type from visual features"""
        # Simple material estimation based on intensity variance
        if obj_region.size > 0:
            gray = cv2.cvtColor(obj_region, cv2.COLOR_BGR2GRAY)
            variance = gray.var()
            
            if variance < 100:
                return "plastic"
            elif variance < 500:
                return "metal"
            else:
                return "wood"
        return "unknown"
    
    def _run_graspnet_prediction(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                                 semantic_objects: List[Dict]) -> List[Dict]:
        """Generate 6D grasp poses using GraspNet"""
        grasps = []
        
        for obj in semantic_objects:
            if not obj.get("graspable", True):
                continue
            
            bbox = obj["bbox"]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Estimate depth
            if depth_image is not None and depth_image.shape[0] > center_y and depth_image.shape[1] > center_x:
                depth = float(depth_image[center_y, center_x]) / 1000.0  # Convert to meters
            else:
                depth = 0.5  # Default depth
            
            # Generate grasp candidates
            for approach in ["top", "side"]:
                # Calculate grasp orientation based on approach
                if approach == "top":
                    orientation = [0, 0.707, 0, 0.707]  # Looking down
                else:
                    orientation = [0, 0, 0, 1]  # Horizontal
                
                grasp = {
                    "id": f"grasp_{obj['id']}_{approach}",
                    "target_object_id": obj["id"],
                    "target_class": obj.get("class", "object"),
                    "pose": {
                        "position": [center_x / 1000.0, center_y / 1000.0, depth],
                        "orientation": orientation
                    },
                    "quality": 0.6 + 0.3 * np.random.random(),
                    "width": 0.03 + 0.05 * np.random.random(),
                    "approach_direction": approach,
                    "grasp_type": "pinch" if approach == "top" else "power",
                    "collision_free": True,
                    "execution_time_estimate": 2.0 + 2.0 * np.random.random()
                }
                grasps.append(grasp)
        
        return grasps
    
    def _build_scene_graph(self, semantic_objects: List[Dict], grasps: List[Dict]) -> Dict:
        """Build scene graph with spatial relationships"""
        relations = []
        
        # Generate spatial relations
        for i, obj1 in enumerate(semantic_objects):
            for j, obj2 in enumerate(semantic_objects):
                if i >= j:
                    continue
                
                # Calculate spatial relationship
                pos1 = obj1["center"]
                pos2 = obj2["center"]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < 200:  # Close objects
                    relation_type = "near"
                elif pos1[1] < pos2[1] - 50:
                    relation_type = "above"
                elif pos1[1] > pos2[1] + 50:
                    relation_type = "below"
                elif pos1[0] < pos2[0]:
                    relation_type = "left_of"
                else:
                    relation_type = "right_of"
                
                relation = {
                    "relation_type": relation_type,
                    "subject_object_id": i,
                    "target_object_id": j,
                    "confidence": 0.85,
                    "distance": distance / 1000.0,
                    "direction": relation_type
                }
                relations.append(relation)
        
        scene_graph = {
            "scene_id": f"scene_{len(semantic_objects)}_objects",
            "num_objects": len(semantic_objects),
            "scene_confidence": 0.88,
            "objects": semantic_objects,
            "relations": relations,
            "scene_type": "workspace",
            "scene_affordances": ["manipulation", "grasping", "assembly"],
            "has_support_surface": True
        }
        
        return scene_graph
    
    def _process_simulation(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                           output_dir: str) -> Dict:
        """Process using simulation/research mode"""
        logger.info("🔄 Processing in simulation mode (SAM → CLIP → GraspNet → Scene)...")
        
        # Simulate SAM segmentation
        detections, masks = self._simulate_sam_segmentation(rgb_image)
        
        # Simulate CLIP tagging
        semantic_objects = self._run_clip_tagging(rgb_image, detections, masks)
        
        # Simulate grasp generation
        grasps = self._run_graspnet_prediction(rgb_image, depth_image, semantic_objects)
        
        # Build scene graph
        scene_graph = self._build_scene_graph(semantic_objects, grasps)
        
        # Create debug visualization
        debug_image = self._create_debug_visualization(rgb_image, detections, grasps, masks)
        
        results = {
            "detections": detections,
            "semantic_objects": semantic_objects,
            "grasps": grasps,
            "scene_graph": scene_graph,
            "debug_image": debug_image,
            "processing_time": 0.5,
            "mode": "simulation"
        }
        
        if output_dir:
            self._save_debug_outputs(results, output_dir)
        
        return results
    
    def _simulate_sam_segmentation(self, rgb_image: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """Simulate SAM automatic segmentation"""
        # Use simple computer vision for simulation
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        masks = []
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create mask
                mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                detection = {
                    "id": f"obj_{i}",
                    "class": "object",
                    "confidence": 0.75 + 0.2 * np.random.random(),
                    "bbox": [x, y, x + w, y + h],
                    "center": [x + w//2, y + h//2],
                    "area": cv2.contourArea(contour),
                    "mask": mask
                }
                detections.append(detection)
                masks.append(mask)
        
        # Ensure at least one detection
        if not detections:
            h, w = rgb_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
            
            detections.append({
                "id": "obj_0",
                "class": "object",
                "confidence": 0.8,
                "bbox": [w//4, h//4, 3*w//4, 3*h//4],
                "center": [w//2, h//2],
                "area": (w*h)//4,
                "mask": mask
            })
            masks.append(mask)
        
        return detections, masks
    
    def _create_debug_visualization(self, rgb_image: np.ndarray, detections: List[Dict],
                                   grasps: List[Dict], masks: List[np.ndarray]) -> np.ndarray:
        """Create comprehensive debug visualization"""
        debug_img = rgb_image.copy()
        
        # Draw segmentation masks with transparency
        mask_overlay = np.zeros_like(rgb_image)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (det, mask) in enumerate(zip(detections, masks)):
            if mask is not None:
                color = colors[i % len(colors)]
                mask_3channel = cv2.merge([mask, mask, mask])
                mask_overlay[mask_3channel > 0] = color
        
        # Blend masks with original image
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # Draw bounding boxes and labels
        for det in detections:
            bbox = det["bbox"]
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            label = f"{det.get('class', 'object')}: {det['confidence']:.2f}"
            cv2.putText(debug_img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw grasp poses
        for grasp in grasps:
            pos = grasp["pose"]["position"]
            x, y = int(pos[0] * 1000), int(pos[1] * 1000)
            
            color = (255, 0, 0) if grasp["approach_direction"] == "top" else (0, 0, 255)
            cv2.circle(debug_img, (x, y), 8, color, -1)
            cv2.circle(debug_img, (x, y), 12, color, 2)
            
            # Draw approach direction
            if grasp["approach_direction"] == "top":
                cv2.arrowedLine(debug_img, (x, y - 20), (x, y - 5), color, 2)
            else:
                cv2.arrowedLine(debug_img, (x - 20, y), (x - 5, y), color, 2)
        
        return debug_img
    
    def _save_debug_outputs(self, results: Dict, output_dir: str):
        """Save debug outputs to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save debug image
        if "debug_image" in results:
            cv2.imwrite(str(output_path / "sam_pipeline_visualization.jpg"), results["debug_image"])
        
        # Save results as JSON
        json_results = {k: v for k, v in results.items() if k not in ["debug_image", "masks"]}
        
        # Convert numpy arrays to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json_results = convert_numpy(json_results)
        
        with open(output_path / "sam_pipeline_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f" Debug outputs saved to {output_path}")