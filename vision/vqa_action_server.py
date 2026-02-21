#!/usr/bin/env python3
"""
VQA Action Server

Exposes a ROS2 action server `/vqa` using `custom_interfaces.action.Prompt`.
The goal carries a natural-language question, the server grabs the latest RGB
frame from the configured camera topic, and answers using an Ollama
vision-language model via HTTP (default: `qwen3-vl:8b`).

Uses only standard library (urllib, json) for HTTP calls - no external dependencies.

Action interface (custom_interfaces/action/Prompt):
  # Goal
  string prompt
  ---
  # Result
  bool success
  string final_response
  ---
  # Feedback
  string[] tools_called

Usage example:
  ros2 run vision vqa_action_server
  ros2 action send_goal /vqa custom_interfaces/action/Prompt "{prompt: 'What objects are visible?'}"

Parameters:
    - real_hardware (bool): switch camera topics between hardware and sim (default: false)
    - vlm_model (string): Ollama model name to query (default: qwen3-vl:8b)
    - ollama_host (string): Base URL for Ollama HTTP API (default: http://localhost:11434)
    - image_timeout_sec (double): max allowed age of the cached frame in seconds (default: 2.0)
    - system_prompt (string): system prompt to guide model behavior (default: concise answering)
    - include_metadata (bool): include model name and timing in response (default: false)
    - image_reliability (string): image QoS reliability ('best_effort' or 'reliable', default: best_effort)
"""

import base64
import json
import time
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from custom_interfaces.action import Prompt

MODEL_NAME = 'qwen3-vl:8b'
OLLAMA_HOST = 'http://localhost:11434'
IMAGE_TIMEOUT = 2.0
SYSTEM_PROMPT = (
    'You are a concise visual question answering assistant. '
    'Answer questions about images directly and briefly without showing your reasoning process. '
    'Provide only the final answer in 1-2 sentences maximum.'
)


class VQAActionServer(Node):
    """ROS2 node hosting the /vqa action server."""

    def __init__(self) -> None:
        super().__init__('vqa_action_server')

        # Parameters
        self.declare_parameter('real_hardware', False)
        self.declare_parameter('image_reliability', 'best_effort')

        self.real_hardware = bool(self.get_parameter('real_hardware').value)
        reliability_param = str(self.get_parameter('image_reliability').value).lower()
        if reliability_param == 'reliable':
            reliability = QoSReliabilityPolicy.RELIABLE
        elif reliability_param in ('best_effort', 'besteffort', 'best-effort'):
            reliability = QoSReliabilityPolicy.BEST_EFFORT
        else:
            self.get_logger().warn(
                f"Unknown image_reliability '{reliability_param}', defaulting to best_effort."
            )
            reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.model_name = MODEL_NAME
        self.ollama_host = OLLAMA_HOST
        self.image_timeout = IMAGE_TIMEOUT
        self.system_prompt = SYSTEM_PROMPT
        self.image_reliability = reliability

        # Camera topics
        if self.real_hardware:
            self.rgb_topic = '/camera/color/image_raw'
            self.desired_encoding = 'passthrough'
        else:
            self.rgb_topic = '/camera/image_raw'
            self.desired_encoding = 'bgr8'

        # QoS profile
        self.image_qos = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # State
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_image_stamp = None

        self.cb_group = ReentrantCallbackGroup()

        # Subscriber
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            self.image_qos,
            callback_group=self.cb_group,
        )

        # Action server
        self.action_server = ActionServer(
            self,
            Prompt,
            '/vqa',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info('=' * 80)
        self.get_logger().info('VQA Action Server ready')
        self.get_logger().info(f"  Topic: {self.rgb_topic}")
        self.get_logger().info(f"  Model: {self.model_name}")
        self.get_logger().info(f"  Ollama API: {self.ollama_host}")
        self.get_logger().info(f"  Image QoS reliability: {self.image_reliability.name}")
        self.get_logger().info('  Action: /vqa (custom_interfaces/action/Prompt)')
        self.get_logger().info('=' * 80)

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------
    def rgb_callback(self, msg: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.desired_encoding
            )
            self.latest_image_stamp = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert incoming image: {exc}')

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    def goal_callback(self, goal_request: Prompt.Goal) -> GoalResponse:
        self.get_logger().info(f'Received VQA goal: {goal_request.prompt!r}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Execute VQA
    # ------------------------------------------------------------------
    def execute_callback(self, goal_handle) -> Prompt.Result:
        prompt_text = goal_handle.request.prompt

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return self._build_result(False, 'Goal canceled before execution.')

        # Ensure we have an image
        frame = self.latest_rgb.copy() if self.latest_rgb is not None else None
        if frame is None:
            goal_handle.abort()
            return self._build_result(False, f'No image available from {self.rgb_topic}.')

        # Check staleness
        if self.latest_image_stamp is not None:
            age = (self.get_clock().now() - self.latest_image_stamp).nanoseconds / 1e9
            if age > self.image_timeout:
                self.get_logger().warn(
                    f'Latest frame is stale ({age:.2f}s old, threshold {self.image_timeout:.2f}s).'
                )

        # Encode image to base64
        try:
            success, buffer = cv2.imencode('.png', frame)
            if not success:
                raise RuntimeError('cv2.imencode returned False')

            image_bytes = buffer.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode('ascii')

        except Exception as exc:
            goal_handle.abort()
            return self._build_result(False, f'Failed to encode image: {exc}')

        # Feedback
        feedback = Prompt.Feedback()
        feedback.tools_called = ['ollama_http']
        goal_handle.publish_feedback(feedback)

        # Call Ollama via HTTP (standard library only)
        start = time.perf_counter()

        try:
            image_data_url = f"data:image/png;base64,{image_b64}"

            # Build request payload matching Ollama API format
            # Include system prompt to constrain output
            payload = {
                'model': self.model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': self.system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': prompt_text,
                        'images': [image_b64],
                    }
                ],
                'stream': False,
            }

            # Make HTTP POST request using stdlib
            url = f"{self.ollama_host}/api/chat"
            req = Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urlopen(req, timeout=120) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            generation_time = time.perf_counter() - start

            # Extract text from response
            text = ''
            if isinstance(response_data, dict):
                # Ollama /api/chat returns message.content
                msg = response_data.get('message', {})
                text = msg.get('content', '') if isinstance(msg, dict) else ''
                # Fallback to 'response' field if present
                if not text:
                    text = response_data.get('response', '')

            if not text:
                text = 'No content returned from model.'

        except HTTPError as exc:
            goal_handle.abort()
            error_body = exc.read().decode('utf-8') if exc.fp else str(exc)
            return self._build_result(False, f'Ollama HTTP {exc.code}: {error_body}')
        except URLError as exc:
            goal_handle.abort()
            return self._build_result(False, f'Ollama connection failed: {exc.reason}')
        except Exception as exc:
            goal_handle.abort()
            return self._build_result(False, f'Ollama HTTP call failed: {exc}')

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return self._build_result(False, 'Goal canceled during execution.')

        goal_handle.succeed()

        final_msg = text

        self.get_logger().info(f"Final response: {final_msg}")

        return self._build_result(True, final_msg)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(success: bool, message: str) -> Prompt.Result:
        result = Prompt.Result()
        result.success = bool(success)
        result.final_response = str(message)
        return result

    def destroy_node(self) -> None:
        self.action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VQAActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
