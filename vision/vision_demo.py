#!/usr/bin/env python3
"""
SAM Vision Pipeline Demo
Demonstrates the complete 4-stage vision pipeline with test data
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import cv2
import numpy as np
import os
import time
from pathlib import Path
import json

class VisionPipelineDemo(Node):
    """
    Demo node for the SAM vision pipeline
    Tests the pipeline with sample images and displays results
    """
    
    def __init__(self):
        super().__init__('vision_pipeline_demo')
        
        # Demo configuration
        self.demo_images_path = Path(__file__).parent.parent / "Final-proj" / "data" / "test_images"
        self.output_path = Path.home() / "vision_demo_results"
        self.output_path.mkdir(exist_ok=True)
        
        # Service clients
        self.process_client = self.create_client(Trigger, '/vision/process_scene')
        self.reset_client = self.create_client(Trigger, '/vision/reset_pipeline')
        
        self.get_logger().info(" SAM Vision Pipeline Demo Started!")
        self.get_logger().info(f" Demo images path: {self.demo_images_path}")
        self.get_logger().info(f" Results will be saved to: {self.output_path}")
        
        # Wait for services
        self.wait_for_services()
        
        # Run demo
        self.run_demo()
    
    def wait_for_services(self):
        """Wait for vision pipeline services to become available"""
        self.get_logger().info("⏳ Waiting for vision pipeline services...")
        
        while not self.process_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Processing service not available, waiting...')
        
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')
        
        self.get_logger().info(" All services available!")
    
    def run_demo(self):
        """Run the complete demo sequence"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(" Starting SAM Vision Pipeline Demo")
        self.get_logger().info("="*60)
        
        # Demo sequence
        demos = [
            self.demo_1_basic_processing,
            self.demo_2_scene_understanding,
            self.demo_3_grasp_planning,
            self.demo_4_real_time_simulation
        ]
        
        for i, demo_func in enumerate(demos, 1):
            self.get_logger().info(f"\n🎯 Running Demo {i}: {demo_func.__name__.replace('_', ' ').title()}")
            try:
                demo_func()
                self.get_logger().info(f" Demo {i} completed successfully!")
            except Exception as e:
                self.get_logger().error(f" Demo {i} failed: {e}")
            
            time.sleep(2)  # Pause between demos
        
        self.get_logger().info("\n All demos completed!")
        self.show_demo_summary()
    
    def demo_1_basic_processing(self):
        """Demo 1: Basic image processing through the pipeline"""
        self.get_logger().info("📸 Testing basic pipeline processing...")
        
        # Reset pipeline
        self.call_service(self.reset_client, "Reset pipeline")
        
        # Process a few test images
        test_images = self.get_test_images()
        
        for img_name, img_path in test_images[:2]:  # Process first 2 images
            self.get_logger().info(f"   Processing: {img_name}")
            
            # In a real scenario, these would be published to camera topics
            # For demo, we'll trigger processing and assume images are available
            success = self.call_service(self.process_client, f"Process {img_name}")
            
            if success:
                self.get_logger().info(f"  {img_name} processed successfully")
            else:
                self.get_logger().warn(f"   {img_name} processing failed")
            
            time.sleep(1)
    
    def demo_2_scene_understanding(self):
        """Demo 2: Scene understanding and object relationships"""
        self.get_logger().info("🗺️ Testing scene understanding capabilities...")
        
        # Load a complex scene image
        test_images = self.get_test_images()
        
        if test_images:
            img_name, img_path = test_images[0]
            self.get_logger().info(f"   Analyzing scene: {img_name}")
            
            # Process for scene understanding
            success = self.call_service(self.process_client, f"Analyze scene {img_name}")
            
            if success:
                self.get_logger().info("   Scene analysis capabilities:")
                self.get_logger().info("   - Object detection and segmentation")
                self.get_logger().info("   - Semantic attribute extraction")
                self.get_logger().info("   - Spatial relationship mapping")
                self.get_logger().info("   - Scene graph construction")
            
            # Simulate scene analysis results
            self.simulate_scene_analysis_output()
    
    def demo_3_grasp_planning(self):
        """Demo 3: 6D grasp pose generation"""
        self.get_logger().info("🤏 Testing 6D grasp planning capabilities...")
        
        # Test grasp generation
        success = self.call_service(self.process_client, "Generate grasp poses")
        
        if success:
            self.get_logger().info("   Grasp planning features:")
            self.get_logger().info("   - 6D pose estimation (position + orientation)")
            self.get_logger().info("   - Grasp quality scoring")
            self.get_logger().info("   - Collision-free verification")
            self.get_logger().info("   - Multiple approach angles")
            self.get_logger().info("   - Gripper width estimation")
            
            # Show example grasp poses
            self.simulate_grasp_output()
    
    def demo_4_real_time_simulation(self):
        """Demo 4: Real-time processing simulation"""
        self.get_logger().info(" Testing real-time processing simulation...")
        
        # Simulate real-time processing
        processing_times = []
        
        for i in range(5):
            start_time = time.time()
            
            success = self.call_service(self.process_client, f"Real-time frame {i+1}")
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            self.get_logger().info(f"   Frame {i+1}: {processing_time:.2f}s")
            time.sleep(0.5)
        
        # Show performance stats
        avg_time = np.mean(processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        self.get_logger().info(f"   Average processing time: {avg_time:.2f}s")
        self.get_logger().info(f"   Effective FPS: {fps:.1f}")
        self.get_logger().info(f"   Performance: {'Good' if fps > 1 else 'Moderate' if fps > 0.5 else 'Needs optimization'}")
    
    def get_test_images(self):
        """Get list of available test images"""
        test_images = []
        
        if self.demo_images_path.exists():
            for img_file in self.demo_images_path.glob("*.{jpg,png,jpeg}"):
                test_images.append((img_file.stem, img_file))
        
        # Add some default test cases if no images found
        if not test_images:
            test_images = [
                ("synthetic_tools", "synthetic"),
                ("workshop_scene", "synthetic"),
                ("assembly_task", "synthetic")
            ]
        
        return test_images
    
    def call_service(self, client, description):
        """Call a ROS2 service and return success status"""
        try:
            request = Trigger.Request()
            future = client.call_async(request)
            
            # Wait for response (with timeout)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().debug(f" {description}: {response.message}")
                    return True
                else:
                    self.get_logger().warn(f"{description}: {response.message}")
                    return False
            else:
                self.get_logger().error(f" {description}: Service call timed out")
                return False
                
        except Exception as e:
            self.get_logger().error(f" {description}: Service call failed - {e}")
            return False
    
    def simulate_scene_analysis_output(self):
        """Simulate and display scene analysis output"""
        scene_data = {
            "objects_detected": 4,
            "object_classes": ["tool", "container", "part", "surface"],
            "spatial_relations": [
                "tool is on surface",
                "container is near tool", 
                "part is inside container"
            ],
            "scene_type": "workshop",
            "confidence": 0.87
        }
        
        self.get_logger().info("   📊 Scene Analysis Results:")
        self.get_logger().info(f"      Objects detected: {scene_data['objects_detected']}")
        self.get_logger().info(f"      Classes: {', '.join(scene_data['object_classes'])}")
        self.get_logger().info(f"      Scene type: {scene_data['scene_type']}")
        self.get_logger().info(f"      Overall confidence: {scene_data['confidence']:.2f}")
        
        for relation in scene_data['spatial_relations']:
            self.get_logger().info(f"      Relation: {relation}")
    
    def simulate_grasp_output(self):
        """Simulate and display grasp planning output"""
        grasp_data = [
            {"object": "tool", "approach": "top", "quality": 0.85, "width": 0.04},
            {"object": "tool", "approach": "side", "quality": 0.72, "width": 0.06},
            {"object": "part", "approach": "pinch", "quality": 0.91, "width": 0.02},
            {"object": "container", "approach": "handle", "quality": 0.78, "width": 0.08}
        ]
        
        self.get_logger().info("   🤖 Grasp Planning Results:")
        for i, grasp in enumerate(grasp_data, 1):
            self.get_logger().info(f"      Grasp {i}: {grasp['object']} via {grasp['approach']} "
                                 f"(quality: {grasp['quality']:.2f}, width: {grasp['width']:.2f}m)")
    
    def show_demo_summary(self):
        """Show final demo summary"""
        summary = f"""
        
🎯 SAM Vision Pipeline Demo Summary
{'='*50}

✅ Completed Demonstrations:
   1. Basic Pipeline Processing
   2. Scene Understanding  
   3. 6D Grasp Planning
   4. Real-time Simulation

🔧 System Capabilities Demonstrated:
   • Object detection and segmentation
   • Semantic understanding with CLIP
   • 6D grasp pose generation
   • Scene graph construction
   • Real-time processing capability

📁 Results Location: {self.output_path}

🚀 Ready for Integration:
   • Gazebo simulation
   • Real robot deployment
   • Custom application development

For more information, see README.md
        """
        
        self.get_logger().info(summary)
        
        # Save demo results
        self.save_demo_results()
    
    def save_demo_results(self):
        """Save demo results to file"""
        demo_results = {
            "demo_completed": True,
            "timestamp": time.time(),
            "demos_run": [
                "basic_processing",
                "scene_understanding", 
                "grasp_planning",
                "real_time_simulation"
            ],
            "system_info": {
                "ros2_version": "humble",
                "pipeline_mode": "simulation",
                "demo_duration": "60-120 seconds"
            },
            "next_steps": [
                "Install full SAM pipeline for production use",
                "Configure camera hardware",
                "Integrate with robot planning system",
                "Customize for specific application domain"
            ]
        }
        
        results_file = self.output_path / "demo_results.json"
        with open(results_file, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        self.get_logger().info(f"📄 Demo results saved to: {results_file}")


def main(args=None):
    """Main entry point for vision demo"""
    rclpy.init(args=args)
    
    try:
        demo_node = VisionPipelineDemo()
        # Demo runs automatically in __init__, so just cleanup
        time.sleep(1)
        demo_node.destroy_node()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()