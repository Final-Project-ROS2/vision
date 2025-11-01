# DetectionResult.msg
# Custom message for object detection results from DINO pipeline

std_msgs/Header header

# Object information
string class_name          # Detected object class
float32 confidence         # Detection confidence [0-1]
geometry_msgs/Polygon bbox # Bounding box as polygon
sensor_msgs/Image mask     # Segmentation mask (optional)

# Semantic information from CLIP
string[] semantic_tags     # Additional semantic descriptors
float32[] tag_confidences  # Confidence for each semantic tag

# 3D information (if depth available)
geometry_msgs/Point center_3d    # 3D center point
float32 depth_estimate          # Estimated depth at center
bool has_depth_info            # Whether 3D info is valid

# Additional attributes
string[] attributes        # Object attributes (color, material, etc.)
float32[] attribute_confidences