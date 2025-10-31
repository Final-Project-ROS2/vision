# SemanticObject.msg
# Represents a semantically understood object in the scene

string object_id                 # Unique identifier
string class_name               # Object class from detection
float32 confidence             # Overall confidence

# Geometric properties
geometry_msgs/PoseStamped pose  # Object pose in 3D space
geometry_msgs/Vector3 size     # Object dimensions (x, y, z)
geometry_msgs/Polygon footprint # 2D footprint on support surface

# Semantic properties
string[] attributes            # Physical attributes (color, material, shape)
string[] affordances          # What can be done with this object
string primary_function       # Main purpose/function of object

# Relationships (indices into SceneGraph.objects array)
int32[] supports              # Objects this object supports
int32[] supported_by          # Objects supporting this object
int32[] near_objects          # Nearby objects
int32[] contains              # Objects contained within this object

# Grasp information
GraspPose[] available_grasps   # Possible grasp poses for this object
bool graspable                # Whether object can be grasped
float32 manipulation_difficulty # How difficult to manipulate [0-1]