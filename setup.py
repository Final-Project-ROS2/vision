from setuptools import find_packages, setup
import os
from glob import glob

import os
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py') + glob('launch/*.sh')),
        # Include config files
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.rviz') + glob('config/*.yaml') + glob('config/*.json')),
        # Include message files
        (os.path.join('share', package_name, 'msg'), 
            glob('msg/*.msg')), 
        # Include DINO pipeline components
        (os.path.join('share', package_name, 'Final-proj'), 
            glob('Final-proj/**/*.py', recursive=True)),
        # Include model weights (if present)
        (os.path.join('share', package_name, 'models'), 
            glob('Final-proj/**/*.pth', recursive=True)),
    ],
    install_requires=[
        'setuptools',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.35.0',
        'accelerate>=0.24.0',
        'pillow>=10.0.0',
        'matplotlib>=3.7.0',
        'scipy>=1.11.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'tqdm>=4.66.0',
        'jsonschema>=4.19.0',
    ],
    zip_safe=True,
    maintainer='final-project',
    maintainer_email='karamahati@gmail.com',
    description='ROS2 SAM Vision Pipeline - 4-stage robotic vision system with SAM (Meta), CLIP, GraspNet, and Scene Understanding',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_sam_detector = vision.simple_sam_detector:main',
            'clip_classifier = vision.clip_classifier:main',
            'sam_clip_pipeline = vision.sam_clip_pipeline:main',
            'graspnet_detector = vision.graspnet_detector:main',
            'scene_understanding = vision.scene_understanding:main',
            'pixel_to_real_service = vision.pixel_to_real:main',
            'unified_pipeline = vision.unified_pipeline:main',
            'find_object_service = vision.find_object_service_node:main',
            'find_object_grasp_service = vision.find_object_grasp_service_node:main',
            # 'show_rgb_image = vision.show_rgb_image_node:main',
            # 'show_depth_image = vision.show_depth_image_node:main',
            # 'camera_service = vision.camera_service_node:main',
            # 'sam_vision_pipeline = vision.sam_vision_pipeline_node:main',
            # 'vision_demo = vision.vision_demo:main',
            # 'view_detections = scripts.view_detections:main',
            # 'test_services = scripts.test_services:main',
            # 'test_pipeline_images = scripts.test_pipeline_images:main',
            # 'integration_test = scripts.integration_test:main',
        ],
    },
)
