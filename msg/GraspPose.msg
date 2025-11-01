# GraspPose.msg 
# Custom message for 6D grasp poses from GraspNet

std_msgs/Header header

# Grasp pose
geometry_msgs/PoseStamped pose    # 6D grasp pose (position + orientation)
float32 quality                   # Grasp quality score [0-1]
float32 width                     # Gripper width needed
string approach_direction         # "top", "side", "angled"

# Associated object
string target_object_id           # ID of target object
string target_class              # Class of target object
geometry_msgs/Point contact_point # Primary contact point

# Grasp type and parameters
string grasp_type                # "pinch", "power", "lateral"
float32 force_estimate           # Estimated required force
bool collision_free              # Whether grasp is collision-free

# Execution metadata
float32 execution_time_estimate  # Estimated time to execute
string[] prerequisites           # Required pre-conditions