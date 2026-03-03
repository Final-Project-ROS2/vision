"""
MiniGrasp Configuration
Simplified parameters for grasp detection
"""

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================

# Depth filtering (meters)
MIN_VALID_DEPTH = 0.1  # Minimum reliable depth (10cm)
MAX_VALID_DEPTH = 2.0  # Maximum workspace depth (2m) - very permissive

# ==============================================================================
# PLANE REMOVAL (Remove table/floor background)
# ==============================================================================
# RANSAC parameters for detecting and removing dominant plane (table/floor)
RANSAC_DISTANCE_THRESHOLD = 0.01  # Max distance point can be from plane (1cm)
RANSAC_NUM_ITERATIONS = 1000      # RANSAC iterations for plane detection
RANSAC_MIN_POINTS = 3             # Minimum points to fit plane

# Plane removal settings
REMOVE_PLANE = True               # Enable automatic plane removal
PLANE_REMOVAL_AGGRESSIVE = True   # Remove ALL points close to plane (recommended)

# ==============================================================================
# TOP-VIEW OCCLUSION HANDLING
# ==============================================================================
# When camera only sees top of object, we need to estimate 3D volume
EXTRUDE_TOP_SURFACE = True        # Create synthetic 3D volume from 2D top view
EXTRUSION_DEPTH = 0.08            # How far to extrude downward (8cm for boxes)
EXTRUSION_METHOD = 'uniform'      # 'uniform' or 'adaptive'

# Object shape assumptions (for better extrusion)
ASSUMED_OBJECT_HEIGHT = 0.08      # Default object height if unknown (8cm)
MIN_OBJECT_HEIGHT = 0.02          # Minimum reasonable height (2cm)
MAX_OBJECT_HEIGHT = 0.15          # Maximum reasonable height (15cm)

# ==============================================================================
# WORKSPACE BOUNDS
# ==============================================================================
# Define the 3D region where objects can be grasped
# Coordinates are relative to camera frame:
#   X: Right (+) / Left (-)
#   Y: Down (+) / Up (-)
#   Z: Away from camera (+)

WORKSPACE_BOUNDS = {
    'x_min': -1.0,    # 1m to the left (very permissive)
    'x_max': 1.0,     # 1m to the right
    'y_min': -1.0,    # 1m up
    'y_max': 1.0,     # 1m down
    'z_min': 0.2,     # Start at 20cm from camera
    'z_max': 1.5,     # End at 1.5m from camera
}

# ==============================================================================
# GRIPPER PARAMETERS
# ==============================================================================

# Physical gripper dimensions (meters)
GRIPPER_WIDTH = 0.05      # Maximum opening width (5cm)
GRIPPER_MIN_WIDTH = 0.001   # Minimum opening width (1cm)
GRIPPER_DEPTH = 0.05       # Finger depth (5cm)
GRIPPER_HEIGHT = 0.02      # Finger height (2cm)

# Gripper force/stability parameters
GRIPPER_FORCE = 50.0       # Maximum gripping force (N) - adjust to your gripper
FRICTION_COEFFICIENT = 0.5 # Friction coefficient with objects

# Width optimization
PREFERRED_WIDTH_RATIO = 0.6  # Prefer using 60% of gripper range
WIDTH_SAFETY_MARGIN = 1.1    # Add 10% margin to calculated width

# ==============================================================================
# GRASP REFINEMENT & ALIGNMENT
# ==============================================================================
# Refine grasp poses to align better with actual point cloud
ENABLE_GRASP_REFINEMENT = True    # Enable ICP-like refinement
REFINEMENT_MAX_ITERATIONS = 5     # Max refinement steps
REFINEMENT_DISTANCE = 0.002       # Convergence threshold (2mm)

# Point-to-grasp alignment
CENTER_ON_POINTS = True           # Move grasp to actual point cluster center
MIN_POINTS_FOR_GRASP = 5          # Minimum points near grasp for validity (very permissive)

# ==============================================================================
# GRASP GENERATION
# ==============================================================================

# Number of grasp candidates to generate
NUM_GRASP_CANDIDATES = 300  # Balance between coverage and speed

# Minimum grasp quality score (0-1)
MIN_GRASP_SCORE = 0.4  # Very permissive - almost all grasps pass

# ==============================================================================
# GRASP FILTERING
# ==============================================================================

# Approach angle filtering
# Target vector (typically pointing down toward table)
APPROACH_TARGET_VECTOR = [0, 0, -1]  # Downward in camera frame

# Maximum deviation from target approach (degrees)
MAX_APPROACH_ANGLE = 360
# Collision checking
COLLISION_FINGER_WIDTH = 0.01   # Finger thickness
COLLISION_BASE_DEPTH = 0.03     # Gripper base depth

# ==============================================================================
# QUICK PRESETS
# ==============================================================================

def get_table_50cm_config():
    """Preset for table at 50cm from camera"""
    global WORKSPACE_BOUNDS, MAX_VALID_DEPTH
    WORKSPACE_BOUNDS = {
        'x_min': -0.25, 'x_max': 0.25,
        'y_min': -0.25, 'y_max': 0.25,
        'z_min': 0.2, 'z_max': 0.45,
    }
    MAX_VALID_DEPTH = 0.5

def get_table_80cm_config():
    """Preset for table at 80cm from camera"""
    global WORKSPACE_BOUNDS, MAX_VALID_DEPTH
    WORKSPACE_BOUNDS = {
        'x_min': -0.35, 'x_max': 0.35,
        'y_min': -0.35, 'y_max': 0.35,
        'z_min': 0.4, 'z_max': 0.75,
    }
    MAX_VALID_DEPTH = 0.8

def get_bin_picking_config():
    """Preset for bin picking"""
    global WORKSPACE_BOUNDS, MAX_VALID_DEPTH, APPROACH_TARGET_VECTOR
    WORKSPACE_BOUNDS = {
        'x_min': -0.2, 'x_max': 0.2,
        'y_min': -0.2, 'y_max': 0.2,
        'z_min': 0.2, 'z_max': 0.6,
    }
    MAX_VALID_DEPTH = 0.7
    APPROACH_TARGET_VECTOR = [0, 0, -1]  # Looking down into bin
