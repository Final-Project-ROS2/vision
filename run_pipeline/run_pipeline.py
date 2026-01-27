#!/usr/bin/env python3
import subprocess
import time

# Step 1: Build the workspace
subprocess.run(['colcon', 'build'], check=True)

# Step 2: Launch the ROS2 simulation
launch_process = subprocess.Popen(['ros2', 'launch', 'ur_yt_sim', 'final_project.launch.py'])

# Allow some time for the launch to initialize
# Adjust the sleep time as necessary
time.sleep(5)

# Step 3: Initialize a list to store runtimes
runtimes = []

# Step 4: Run the service call 10 times
for _ in range(5): #ros2 service call /vision/understand_scene std_srvs/srv/Trigger
    start_time = time.time()
    subprocess.run(['ros2', 'service', 'call', '/vision/understand_scene', 'std_srvs/srv/Trigger'], check=True)
    end_time = time.time()

    # Step 5: Calculate and round runtime
    runtime = round(end_time - start_time, 3)
    runtimes.append(runtime)
    print(f'Runtime for service call: {runtime} seconds')

# Step 6: Quit the ROS2 launch
launch_process.terminate()