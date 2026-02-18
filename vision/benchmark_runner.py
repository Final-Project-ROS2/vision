#!/usr/bin/env python3
"""
Vision Pipeline Benchmarking Tool

Measures speed, latency, and performance metrics for:
- SAM Object Detection
- CLIP Classification
- GraspNet Grasp Detection

Sends results to benchmark dashboard for visualization.

Usage:
    # Run all benchmarks
    ros2 run vision benchmark_runner
    
    # Run specific benchmark
    ros2 run vision benchmark_runner --test sam
    ros2 run vision benchmark_runner --test clip
    ros2 run vision benchmark_runner --test graspnet
    
    # Custom iterations
    ros2 run vision benchmark_runner --iterations 50

Requirements:
    - All vision nodes must be running:
      - simple_sam_detector
      - clip_classifier
      - graspnet_detector
      - benchmark_dashboard
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from std_msgs.msg import String
import json
import time
import statistics
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# Import custom interfaces
try:
    from custom_interfaces.srv import (
        DetectObjects, ClassifyBBox, DetectGrasps, 
        DetectGraspBBox, FindObject
    )
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    CUSTOM_INTERFACES_AVAILABLE = False
    print("ERROR: Custom interfaces not available. Build custom_interfaces package first.")


class BenchmarkRunner(Node):
    """
    Benchmarking tool for vision pipeline components
    """
    
    def __init__(self, test_type='all', iterations=30):
        super().__init__('benchmark_runner')
        
        self.test_type = test_type
        self.iterations = iterations
        
        # Callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service clients
        if CUSTOM_INTERFACES_AVAILABLE:
            # SAM detector services
            self.run_pipeline_client = self.create_client(
                Trigger,
                '/vision/run_pipeline',
                callback_group=self.callback_group
            )
            
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects',
                callback_group=self.callback_group
            )
            
            # CLIP classifier services
            self.classify_all_client = self.create_client(
                Trigger,
                '/vision/classify_all',
                callback_group=self.callback_group
            )
            
            self.classify_bbox_client = self.create_client(
                ClassifyBBox,
                '/vision/classify_bb',
                callback_group=self.callback_group
            )
            
            self.find_object_client = self.create_client(
                FindObject,
                '/vision/find_object',
                callback_group=self.callback_group
            )
            
            # GraspNet detector services
            self.detect_grasp_client = self.create_client(
                Trigger,
                '/vision/detect_grasp',
                callback_group=self.callback_group
            )
            
            self.detect_grasp_bbox_client = self.create_client(
                DetectGraspBBox,
                '/vision/detect_grasp_bb',
                callback_group=self.callback_group
            )
        
        # Publisher for benchmark results
        self.results_publisher = self.create_publisher(
            String,
            '/benchmark/results',
            10
        )
        
        # Storage for results
        self.benchmark_results = {
            'sam': {},
            'clip': {},
            'graspnet': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'iterations': iterations,
                'test_type': test_type
            }
        }
        
        self.get_logger().info('Benchmark Runner initialized')
        self.get_logger().info(f'Test type: {test_type}')
        self.get_logger().info(f'Iterations: {iterations}')
    
    def wait_for_services(self, timeout_sec=10.0):
        """Wait for all required services to be available"""
        self.get_logger().info('Waiting for services...')
        
        services = []
        
        if self.test_type in ['all', 'sam']:
            services.extend([
                (self.run_pipeline_client, '/vision/run_pipeline'),
                (self.detect_objects_client, '/vision/detect_objects')
            ])
        
        if self.test_type in ['all', 'clip']:
            services.extend([
                (self.classify_all_client, '/vision/classify_all'),
                (self.classify_bbox_client, '/vision/classify_bb'),
                (self.find_object_client, '/vision/find_object')
            ])
        
        if self.test_type in ['all', 'graspnet']:
            services.extend([
                (self.detect_grasp_client, '/vision/detect_grasp'),
                (self.detect_grasp_bbox_client, '/vision/detect_grasp_bb')
            ])
        
        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f'Service {name} not available!')
                return False
            self.get_logger().info(f'  ✓ {name} available')
        
        self.get_logger().info('All required services available')
        return True
    
    def benchmark_sam_detector(self):
        """Benchmark SAM object detection"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('BENCHMARKING SAM OBJECT DETECTION')
        self.get_logger().info('=' * 60)
        
        results = {
            'run_pipeline': {'latencies': [], 'successes': 0, 'failures': 0},
            'detect_objects': {'latencies': [], 'successes': 0, 'failures': 0, 'object_counts': []}
        }
        
        # Test 1: run_pipeline (publishes to topic)
        self.get_logger().info(f'\nTest 1: /vision/run_pipeline ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = Trigger.Request()
            start_time = time.perf_counter()
            
            try:
                response = self.run_pipeline_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['run_pipeline']['latencies'].append(latency_ms)
                
                if response.success:
                    results['run_pipeline']['successes'] += 1
                else:
                    results['run_pipeline']['failures'] += 1
                
                self.get_logger().info(f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms')
                
                # Small delay between iterations
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['run_pipeline']['failures'] += 1
        
        # Test 2: detect_objects (returns in service response)
        self.get_logger().info(f'\nTest 2: /vision/detect_objects ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = DetectObjects.Request()
            start_time = time.perf_counter()
            
            try:
                response = self.detect_objects_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['detect_objects']['latencies'].append(latency_ms)
                
                if response.success:
                    results['detect_objects']['successes'] += 1
                    num_objects = len(response.object_ids)
                    results['detect_objects']['object_counts'].append(num_objects)
                    self.get_logger().info(
                        f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms, {num_objects} objects'
                    )
                else:
                    results['detect_objects']['failures'] += 1
                    self.get_logger().warn(f'  Iteration {i+1}/{self.iterations}: FAILED')
                
                # Small delay between iterations
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['detect_objects']['failures'] += 1
        
        # Calculate statistics
        self.benchmark_results['sam'] = self._calculate_stats(results)
        self._print_sam_results(self.benchmark_results['sam'])
    
    def benchmark_clip_classifier(self):
        """Benchmark CLIP classification"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('BENCHMARKING CLIP CLASSIFIER')
        self.get_logger().info('=' * 60)
        
        results = {
            'classify_all': {'latencies': [], 'successes': 0, 'failures': 0},
            'classify_bbox': {'latencies': [], 'successes': 0, 'failures': 0},
            'find_object': {'latencies': [], 'successes': 0, 'failures': 0, 'found_counts': []}
        }
        
        # Test 1: classify_all (entire image)
        self.get_logger().info(f'\nTest 1: /vision/classify_all ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = Trigger.Request()
            start_time = time.perf_counter()
            
            try:
                response = self.classify_all_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['classify_all']['latencies'].append(latency_ms)
                
                if response.success:
                    results['classify_all']['successes'] += 1
                else:
                    results['classify_all']['failures'] += 1
                
                self.get_logger().info(f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms')
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['classify_all']['failures'] += 1
        
        # Test 2: classify_bbox (specific region - test bbox 100,100,300,300)
        self.get_logger().info(f'\nTest 2: /vision/classify_bb ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = ClassifyBBox.Request()
            request.x1 = 100
            request.y1 = 100
            request.x2 = 300
            request.y2 = 300
            
            start_time = time.perf_counter()
            
            try:
                response = self.classify_bbox_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['classify_bbox']['latencies'].append(latency_ms)
                
                if response.success:
                    results['classify_bbox']['successes'] += 1
                else:
                    results['classify_bbox']['failures'] += 1
                
                self.get_logger().info(f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms')
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['classify_bbox']['failures'] += 1
        
        # Test 3: find_object (search by label)
        self.get_logger().info(f'\nTest 3: /vision/find_object ({self.iterations} iterations)')
        test_labels = ['bottle', 'cup', 'box', 'can', 'object']  # Cycle through labels
        
        for i in range(self.iterations):
            request = FindObject.Request()
            request.label = test_labels[i % len(test_labels)]
            
            start_time = time.perf_counter()
            
            try:
                response = self.find_object_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['find_object']['latencies'].append(latency_ms)
                
                if response.success:
                    results['find_object']['successes'] += 1
                    # Count how many objects found
                    try:
                        data = json.loads(response.message)
                        found_count = len(data.get('objects', []))
                        results['find_object']['found_counts'].append(found_count)
                        self.get_logger().info(
                            f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms, {found_count} found'
                        )
                    except:
                        results['find_object']['found_counts'].append(0)
                else:
                    results['find_object']['failures'] += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['find_object']['failures'] += 1
        
        # Calculate statistics
        self.benchmark_results['clip'] = self._calculate_stats(results)
        self._print_clip_results(self.benchmark_results['clip'])
    
    def benchmark_graspnet_detector(self):
        """Benchmark GraspNet grasp detection"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('BENCHMARKING GRASPNET GRASP DETECTION')
        self.get_logger().info('=' * 60)
        
        results = {
            'detect_grasp': {'latencies': [], 'successes': 0, 'failures': 0, 'grasp_counts': []},
            'detect_grasp_bbox': {'latencies': [], 'successes': 0, 'failures': 0, 'grasp_counts': []}
        }
        
        # Test 1: detect_grasp (all objects)
        self.get_logger().info(f'\nTest 1: /vision/detect_grasp ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = Trigger.Request()
            start_time = time.perf_counter()
            
            try:
                response = self.detect_grasp_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['detect_grasp']['latencies'].append(latency_ms)
                
                if response.success:
                    results['detect_grasp']['successes'] += 1
                    # Parse response to count grasps
                    try:
                        data = json.loads(response.message)
                        grasp_count = len(data.get('grasps', []))
                        results['detect_grasp']['grasp_counts'].append(grasp_count)
                        self.get_logger().info(
                            f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms, {grasp_count} grasps'
                        )
                    except:
                        results['detect_grasp']['grasp_counts'].append(0)
                else:
                    results['detect_grasp']['failures'] += 1
                    self.get_logger().warn(f'  Iteration {i+1}/{self.iterations}: FAILED')
                
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['detect_grasp']['failures'] += 1
        
        # Test 2: detect_grasp_bb (specific bounding box)
        self.get_logger().info(f'\nTest 2: /vision/detect_grasp_bb ({self.iterations} iterations)')
        for i in range(self.iterations):
            request = DetectGraspBBox.Request()
            request.x1 = 150
            request.y1 = 150
            request.x2 = 350
            request.y2 = 350
            
            start_time = time.perf_counter()
            
            try:
                response = self.detect_grasp_bbox_client.call(request)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                results['detect_grasp_bbox']['latencies'].append(latency_ms)
                
                if response.success:
                    results['detect_grasp_bbox']['successes'] += 1
                    # Parse response to count grasps
                    try:
                        data = json.loads(response.message)
                        grasp_count = len(data.get('grasps', []))
                        results['detect_grasp_bbox']['grasp_counts'].append(grasp_count)
                        self.get_logger().info(
                            f'  Iteration {i+1}/{self.iterations}: {latency_ms:.2f} ms, {grasp_count} grasps'
                        )
                    except:
                        results['detect_grasp_bbox']['grasp_counts'].append(0)
                else:
                    results['detect_grasp_bbox']['failures'] += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'  Iteration {i+1} failed: {e}')
                results['detect_grasp_bbox']['failures'] += 1
        
        # Calculate statistics
        self.benchmark_results['graspnet'] = self._calculate_stats(results)
        self._print_graspnet_results(self.benchmark_results['graspnet'])
    
    def _calculate_stats(self, results: Dict) -> Dict:
        """Calculate statistics from raw benchmark data"""
        stats = {}
        
        for test_name, test_data in results.items():
            latencies = test_data['latencies']
            
            if latencies:
                stats[test_name] = {
                    'avg_latency_ms': round(statistics.mean(latencies), 2),
                    'min_latency_ms': round(min(latencies), 2),
                    'max_latency_ms': round(max(latencies), 2),
                    'std_dev_ms': round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
                    'median_latency_ms': round(statistics.median(latencies), 2),
                    'throughput_hz': round(1000.0 / statistics.mean(latencies), 2),
                    'successes': test_data['successes'],
                    'failures': test_data['failures'],
                    'success_rate': round(
                        test_data['successes'] / (test_data['successes'] + test_data['failures']) * 100, 2
                    ) if (test_data['successes'] + test_data['failures']) > 0 else 0.0
                }
                
                # Add extra metrics if available
                if 'object_counts' in test_data and test_data['object_counts']:
                    stats[test_name]['avg_objects'] = round(
                        statistics.mean(test_data['object_counts']), 2
                    )
                
                if 'grasp_counts' in test_data and test_data['grasp_counts']:
                    stats[test_name]['avg_grasps'] = round(
                        statistics.mean(test_data['grasp_counts']), 2
                    )
                
                if 'found_counts' in test_data and test_data['found_counts']:
                    stats[test_name]['avg_found'] = round(
                        statistics.mean(test_data['found_counts']), 2
                    )
            else:
                stats[test_name] = {
                    'error': 'No data collected',
                    'successes': 0,
                    'failures': test_data['failures']
                }
        
        return stats
    
    def _print_sam_results(self, stats: Dict):
        """Print SAM benchmark results"""
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('SAM DETECTION BENCHMARK RESULTS')
        self.get_logger().info('=' * 60)
        
        for test_name, test_stats in stats.items():
            self.get_logger().info(f'\n{test_name}:')
            for key, value in test_stats.items():
                self.get_logger().info(f'  {key}: {value}')
    
    def _print_clip_results(self, stats: Dict):
        """Print CLIP benchmark results"""
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('CLIP CLASSIFICATION BENCHMARK RESULTS')
        self.get_logger().info('=' * 60)
        
        for test_name, test_stats in stats.items():
            self.get_logger().info(f'\n{test_name}:')
            for key, value in test_stats.items():
                self.get_logger().info(f'  {key}: {value}')
    
    def _print_graspnet_results(self, stats: Dict):
        """Print GraspNet benchmark results"""
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('GRASPNET DETECTION BENCHMARK RESULTS')
        self.get_logger().info('=' * 60)
        
        for test_name, test_stats in stats.items():
            self.get_logger().info(f'\n{test_name}:')
            for key, value in test_stats.items():
                self.get_logger().info(f'  {key}: {value}')
    
    def publish_results(self):
        """Publish benchmark results to dashboard"""
        self.benchmark_results['metadata']['end_time'] = datetime.now().isoformat()
        
        msg = String()
        msg.data = json.dumps(self.benchmark_results, indent=2)
        self.results_publisher.publish(msg)
        
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('BENCHMARK COMPLETE')
        self.get_logger().info('Results published to /benchmark/results')
        self.get_logger().info('=' * 60)
    
    def run_benchmarks(self):
        """Run all requested benchmarks"""
        if not self.wait_for_services():
            self.get_logger().error('Failed to connect to services. Exiting.')
            return False
        
        self.get_logger().info('\nStarting benchmarks...\n')
        
        try:
            if self.test_type in ['all', 'sam']:
                self.benchmark_sam_detector()
            
            if self.test_type in ['all', 'clip']:
                self.benchmark_clip_classifier()
            
            if self.test_type in ['all', 'graspnet']:
                self.benchmark_graspnet_detector()
            
            self.publish_results()
            return True
            
        except Exception as e:
            self.get_logger().error(f'Benchmark failed: {e}')
            return False


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vision Pipeline Benchmark Runner')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'sam', 'clip', 'graspnet'],
                        help='Type of benchmark to run')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of iterations per test')
    
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    benchmark = BenchmarkRunner(
        test_type=parsed_args.test,
        iterations=parsed_args.iterations
    )
    
    success = benchmark.run_benchmarks()
    
    benchmark.destroy_node()
    rclpy.shutdown()
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
