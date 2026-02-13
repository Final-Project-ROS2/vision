#!/usr/bin/env python3
"""
Camera Intrinsic Calibration Script
Uses OpenCV to calibrate camera intrinsic parameters using a chessboard pattern.
Chessboard: 6x7 corners (7x8 squares), 55mm square size

Features:
- Progressive distortion model selection (k1_only, k1_k2, k1_k2_k3)
- Optimal new camera matrix with configurable alpha parameter
- Real-time chessboard detection during capture
- Comprehensive undistortion testing

Prerequisites:
Launch the camera first:
  ros2 launch ur_yt_sim final_project.launch.py real_camera:=true

Subscribes to RealSense depth camera topics:
- /camera/color/image_raw (RGB stream from RealSense)
- /camera/depth/image_rect_raw (Depth stream from RealSense)

Distortion Model Selection:
1. k1_only: Simplest model, fixes k2 and k3 to zero (RECOMMENDED START)
   - Use when: Initial calibration or fisheye distortion is minimal
   - Flags: cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3

2. k1_k2: Moderate model, fixes k3 to zero
   - Use when: k1_only shows large reprojection error (>0.5 pixels)
   - Flags: cv2.CALIB_FIX_K3

3. k1_k2_k3: Full model, all coefficients free
   - Use when: k1_k2 still has large error AND you have high-quality images
   - Warning: Can overfit with poor calibration images

Alpha Parameter for Undistortion:
- alpha=0: Crop to remove all black pixels (maximize useful area)
- alpha=1: Keep all pixels, may have black borders (DEFAULT)
- alpha=0.5: Balance between the two

Calibration Tips:
- Quality over quantity: 15-20 good images > 40 poor images
- Cover all image corners with chessboard
- Include extreme tilts (rotate along X and Y axes)
- Vary depth: closer and farther from camera
- Ensure sharp focus, good lighting, no motion blur


"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import glob
import os
import json
from datetime import datetime
import threading


class CameraSubscriber(Node):
    """ROS2 node to subscribe to RealSense camera topics and capture images."""
    
    def __init__(self):
        super().__init__('camera_calibrator_subscriber')
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()
        
        # RealSense topics when launched with real_camera:=true
        rgb_topic = '/camera/color/image_raw'
        depth_topic = '/camera/depth/image_rect_raw'
        
        # Subscribe to camera topics
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10
        )
        
        self.get_logger().info('='*70)
        self.get_logger().info('Camera Calibrator - Waiting for RealSense topics')
        self.get_logger().info('='*70)
        self.get_logger().info(f'RGB Topic:   {rgb_topic}')
        self.get_logger().info(f'Depth Topic: {depth_topic}')
        self.get_logger().info('='*70)
        self.get_logger().info('Make sure camera is launched with:')
        self.get_logger().info('  ros2 launch ur_yt_sim final_project.launch.py real_camera:=true')
        self.get_logger().info('='*70)
    
    def rgb_callback(self, msg):
        """Store latest RGB image."""
        try:
            # Use 'passthrough' to get the image as-is, then convert if needed
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # If image is RGB, convert to BGR for OpenCV
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                if msg.encoding == 'rgb8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                # If it's already bgr8 or 8UC3 (which is BGR), keep as is
            
            with self.lock:
                self.latest_rgb = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def get_latest_rgb(self):
        """Get the latest RGB frame."""
        with self.lock:
            return self.latest_rgb.copy() if self.latest_rgb is not None else None
    
    def get_latest_depth(self):
        """Get the latest depth frame."""
        with self.lock:
            return self.latest_depth.copy() if self.latest_depth is not None else None


class CameraCalibrator:
    def __init__(self, chessboard_size=(8, 7), square_size=55.0, distortion_model='k1_only'):
        """
        Initialize the camera calibrator.
        
        Args:
            chessboard_size: Tuple of (width, height) representing inner corners
            square_size: Size of each chessboard square in millimeters
            distortion_model: Distortion model to use:
                - 'k1_only': Only radial distortion k1 (simplest)
                - 'k1_k2': Radial distortion k1 and k2
                - 'k1_k2_k3': Full radial distortion (use only if necessary)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.distortion_model = distortion_model
        
        # Set calibration flags based on distortion model
        if distortion_model == 'k1_only':
            self.calib_flags = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
            print("Using distortion model: k1 only (k2 and k3 fixed to zero)")
        elif distortion_model == 'k1_k2':
            self.calib_flags = cv2.CALIB_FIX_K3
            print("Using distortion model: k1 + k2 (k3 fixed to zero)")
        elif distortion_model == 'k1_k2_k3':
            self.calib_flags = 0  # No fixed parameters
            print("Using distortion model: k1 + k2 + k3 (full model)")
        else:
            raise ValueError(f"Unknown distortion model: {distortion_model}")
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (0,0,0), (55,0,0), (110,0,0) ... (385,330,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
    def capture_images(self, camera_node, num_images=20, output_dir='calibration_images'):
        """
        Capture images from ROS2 camera topic for calibration.
        
        Args:
            camera_node: CameraSubscriber ROS2 node
            num_images: Number of images to capture
            output_dir: Directory to save captured images
        """
        # Use absolute path in calibration folder
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving images to: {output_dir}")
        
        print(f"\n{'='*70}")
        print(f"CALIBRATION IMAGE CAPTURE")
        print(f"{'='*70}")
        print(f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]} inner corners")
        print(f"Square size: {self.square_size}mm")
        print(f"Target images: {num_images}")
        print(f"{'='*70}")
        print(f"Instructions:")
        print(f"  1. Hold chessboard in view of camera")
        print(f"  2. Move it to different positions/angles")
        print(f"  3. Press SPACE when chessboard is detected (green overlay)")
        print(f"  4. Press ESC to exit early")
        print(f"{'='*70}\n")
        print("Waiting for camera data...")
        
        # Wait for first image
        while rclpy.ok():
            rclpy.spin_once(camera_node, timeout_sec=0.1)
            frame = camera_node.get_latest_rgb()
            if frame is not None:
                print("Camera data received! Starting capture...\n")
                break
        
        count = 0
        
        while count < num_images and rclpy.ok():
            rclpy.spin_once(camera_node, timeout_sec=0.01)
            frame = camera_node.get_latest_rgb()
            
            if frame is None:
                continue
            
            # Display the frame
            display_frame = frame.copy()
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            
            # Try to detect chessboard in real-time
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
            
            # Draw corners if detected
            if ret:
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                status_text = "CHESSBOARD DETECTED - Press SPACE to capture"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "Searching for chessboard..."
                status_color = (0, 0, 255)  # Red
            
            # Display status
            cv2.putText(display_frame, f"Captured: {count}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, status_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display_frame, "ESC to exit", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Calibration - Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                print("\nCapture cancelled by user")
                break
            elif key == 32:  # SPACE key
                if ret:  # Only save if chessboard is detected
                    filename = os.path.join(output_dir, f'calibration_{count:03d}.jpg')
                    cv2.imwrite(filename, frame)
                    print(f"✓ Saved: {filename}")
                    count += 1
                else:
                    print("✗ Cannot capture - chessboard not detected!")
        
        cv2.destroyAllWindows()
        print(f"\n{'='*70}")
        print(f"Capture complete: {count}/{num_images} images saved")
        print(f"{'='*70}\n")
        
        return count > 0
    
    def load_images(self, image_path='calibration_images/*.jpg'):
        """
        Load and process calibration images.
        
        Args:
            image_path: Glob pattern for calibration images
            
        Returns:
            Number of successfully processed images
        """
        images = glob.glob(image_path)
        
        if not images:
            print(f"No images found matching pattern: {image_path}")
            return 0
        
        print(f"\n{'='*70}")
        print(f"PROCESSING CALIBRATION IMAGES")
        print(f"{'='*70}")
        print(f"Found {len(images)} images")
        print(f"Chessboard pattern: {self.chessboard_size[0]}x{self.chessboard_size[1]} inner corners")
        print(f"{'='*70}\n")
        
        successful = 0
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            if img is None:
                print(f"[{idx+1}/{len(images)}] ✗ Could not read {os.path.basename(fname)}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners with multiple flags for better detection
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
            
            if ret:
                self.objpoints.append(self.objp)
                
                # Refine corner locations to sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                
                # Draw and display the corners
                cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv2.imshow('Chessboard Detection', img)
                cv2.waitKey(100)
                
                successful += 1
                print(f"[{idx+1}/{len(images)}] ✓ {os.path.basename(fname)}")
            else:
                print(f"[{idx+1}/{len(images)}] ✗ {os.path.basename(fname)} - Chessboard not detected")
        
        cv2.destroyAllWindows()
        print(f"\n{'='*70}")
        print(f"Successfully processed: {successful}/{len(images)} images")
        print(f"{'='*70}\n")
        
        return successful
    
    def calibrate(self, image_shape):
        """
        Perform camera calibration.
        
        Args:
            image_shape: Shape of the calibration images (height, width)
            
        Returns:
            Dictionary containing calibration results
        """
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("Error: No calibration data available")
            return None
        
        print("\nPerforming camera calibration...")
        print(f"Distortion model: {self.distortion_model}")
        
        # imageSize expects (width, height), not full shape
        image_size = (image_shape[1], image_shape[0])  # (width, height)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None, flags=self.calib_flags
        )
        
        if not ret:
            print("Error: Calibration failed")
            return None
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], 
                                             camera_matrix, dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        results = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chessboard_size': self.chessboard_size,
            'square_size_mm': self.square_size,
            'distortion_model': self.distortion_model,
            'num_images': len(self.objpoints),
            'image_size': image_shape[:2],
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'mean_reprojection_error': mean_error,
            'rvecs': [r.tolist() for r in rvecs],
            'tvecs': [t.tolist() for t in tvecs]
        }
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Distortion model: {self.distortion_model}")
        print(f"Number of images used: {results['num_images']}")
        print(f"Image size: {results['image_size']}")
        print(f"Mean reprojection error: {mean_error:.4f} pixels")
        print("\nCamera Matrix (K):")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(f"  k1={dist_coeffs[0][0]:.6f}")
        print(f"  k2={dist_coeffs[0][1]:.6f}")
        print(f"  p1={dist_coeffs[0][2]:.6f}")
        print(f"  p2={dist_coeffs[0][3]:.6f}")
        print(f"  k3={dist_coeffs[0][4]:.6f}")
        print("\nFocal Length:")
        print(f"  fx={camera_matrix[0][0]:.2f} pixels")
        print(f"  fy={camera_matrix[1][1]:.2f} pixels")
        print("\nPrincipal Point:")
        print(f"  cx={camera_matrix[0][2]:.2f} pixels")
        print(f"  cy={camera_matrix[1][2]:.2f} pixels")
        print("="*60)
        
        return results
    
    def save_calibration(self, results, output_file='camera_calibration.json'):
        """
        Save calibration results to a JSON file.
        
        Args:
            results: Calibration results dictionary
            output_file: Output filename
        """
        if results is None:
            print("Error: No results to save")
            return False
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nCalibration data saved to: {output_file}")
        return True
    
    def test_undistortion(self, test_image_path, calibration_results, alpha=1.0):
        """
        Test the calibration by undistorting an image.
        
        Args:
            test_image_path: Path to test image
            calibration_results: Calibration results dictionary
            alpha: Free scaling parameter (0=no black pixels, 1=all pixels visible)
                   0: Maximize useful area, may lose some pixels
                   1: Keep all original pixels, may have black borders
        """
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"Error: Could not read test image: {test_image_path}")
            return
        
        camera_matrix = np.array(calibration_results['camera_matrix'])
        dist_coeffs = np.array(calibration_results['distortion_coefficients'])
        
        h, w = img.shape[:2]
        
        print(f"\nUndistorting with alpha={alpha}...")
        print(f"  alpha=0: Maximize useful area (crop black borders)")
        print(f"  alpha=1: Keep all pixels (may have black borders)")
        
        # Get optimal new camera matrix with specified alpha
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )
        
        print(f"\nNew camera matrix (alpha={alpha}):")
        print(new_camera_matrix)
        print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        
        # Undistort using optimal camera matrix
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Optional: Crop to ROI (remove black borders)
        x, y, w_roi, h_roi = roi
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi] if w_roi > 0 and h_roi > 0 else undistorted
        
        # Display comparison
        cv2.imshow('Original', img)
        cv2.imshow('Undistorted (Full)', undistorted)
        cv2.imshow('Undistorted (Cropped to ROI)', undistorted_cropped)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main calibration workflow."""
    print("="*70)
    print("CAMERA INTRINSIC CALIBRATION - RealSense Depth Camera")
    print("="*70)
    print("Chessboard configuration:")
    print("  - Inner corners: 6x7")
    print("  - Square size: 55mm")
    print("="*70)
    print("Distortion Model Selection (Progressive Approach):")
    print("  1. Start with k1_only (simplest)")
    print("  2. If error > 0.5px, try k1_k2")
    print("  3. If still high error AND good images, try k1_k2_k3")
    print("="*70)
    print("\nPREREQUISITE: Launch camera first with:")
    print("  ros2 launch ur_yt_sim final_project.launch.py real_camera:=true")
    print("="*70)
    
    # Initialize ROS2
    rclpy.init()
    
    # Create ROS2 camera subscriber
    camera_node = CameraSubscriber()
    
    # Default to k1_only for simplicity (can be changed via menu)
    calibrator = None
    distortion_model = 'k1_only'  # Start with simplest model
    
    # Menu
    try:
        while True:
            print("\nOptions:")
            print("1. Capture new calibration images from ROS2 camera")
            print("2. Load existing images and calibrate")
            print("3. Change distortion model (current: {})".format(distortion_model))
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                # Create calibrator with current distortion model if not exists
                if calibrator is None:
                    calibrator = CameraCalibrator(chessboard_size=(6, 7), square_size=55.0, distortion_model=distortion_model)
                
                num_images = input("Number of images to capture (default 20): ").strip()
                num_images = int(num_images) if num_images else 20
                
                output_dir = input("Output directory (default 'calibration_images'): ").strip()
                output_dir = output_dir if output_dir else 'calibration_images'
                
                if calibrator.capture_images(camera_node, num_images, output_dir):
                    print("\nImage capture complete!")
                    
                    # Ask if user wants to proceed with calibration
                    proceed = input("\nProceed with calibration? (y/n): ").strip().lower()
                    if proceed == 'y':
                        image_pattern = os.path.join(output_dir, '*.jpg')
                        num_processed = calibrator.load_images(image_pattern)
                        
                        if num_processed >= 10:
                            # Get image shape from first image
                            test_img = cv2.imread(glob.glob(image_pattern)[0])
                            results = calibrator.calibrate(test_img.shape)
                            
                            if results:
                                calibrator.save_calibration(results)
                                
                                # Test undistortion
                                test = input("\nTest undistortion on an image? (y/n): ").strip().lower()
                                if test == 'y':
                                    alpha_input = input("Enter alpha value (0-1, default 1.0): ").strip()
                                    alpha = float(alpha_input) if alpha_input else 1.0
                                    test_path = glob.glob(image_pattern)[0]
                                    calibrator.test_undistortion(test_path, results, alpha)
                        else:
                            print("Error: Need at least 10 good images for calibration")
            
            elif choice == '2':
                # Create calibrator with current distortion model if not exists
                if calibrator is None:
                    calibrator = CameraCalibrator(chessboard_size=(6, 7), square_size=55.0, distortion_model=distortion_model)
                
                image_pattern = input("Enter image pattern (default 'calibration_images/*.jpg'): ").strip()
                image_pattern = image_pattern if image_pattern else 'calibration_images/*.jpg'
                
                num_processed = calibrator.load_images(image_pattern)
                
                if num_processed >= 10:
                    # Get image shape from first image
                    images = glob.glob(image_pattern)
                    test_img = cv2.imread(images[0])
                    results = calibrator.calibrate(test_img.shape)
                    
                    if results:
                        output_file = input("Output file (default 'camera_calibration.json'): ").strip()
                        output_file = output_file if output_file else 'camera_calibration.json'
                        calibrator.save_calibration(results, output_file)
                        
                        # Test undistortion
                        test = input("\nTest undistortion on an image? (y/n): ").strip().lower()
                        if test == 'y':
                            alpha_input = input("Enter alpha value (0-1, default 1.0): ").strip()
                            alpha = float(alpha_input) if alpha_input else 1.0
                            test_path = images[0]
                            calibrator.test_undistortion(test_path, results, alpha)
                else:
                    print("Error: Need at least 10 good images for calibration")
            
            elif choice == '3':
                print("\nDistortion Model Selection:")
                print("1. k1_only - Only k1 (simplest, recommended to start)")
                print("2. k1_k2 - k1 and k2 (if k1_only has large error)")
                print("3. k1_k2_k3 - Full model (use only if necessary)")
                model_choice = input("Enter choice (1-3): ").strip()
                
                if model_choice == '1':
                    distortion_model = 'k1_only'
                elif model_choice == '2':
                    distortion_model = 'k1_k2'
                elif model_choice == '3':
                    distortion_model = 'k1_k2_k3'
                else:
                    print("Invalid choice, keeping current model")
                    continue
                
                print(f"\nDistortion model set to: {distortion_model}")
                print("Note: This will be used for next calibration. Re-run calibration to apply.")
                
                # Reset calibrator so it uses new model
                calibrator = None
            
            elif choice == '4':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    finally:
        # Cleanup
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
