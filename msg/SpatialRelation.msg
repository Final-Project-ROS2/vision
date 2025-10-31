# SpatialRelation.msg
# Represents spatial relationships between objects

string relation_type          # "on", "in", "near", "left_of", "right_of", etc.
int32 subject_object_id      # Index of subject object
int32 target_object_id       # Index of target object  
float32 confidence          # Confidence in this relation [0-1]

# Geometric details
float32 distance            # Distance between objects (meters)
geometry_msgs/Vector3 relative_position  # Relative position vector
string direction           # Cardinal direction if applicable

# Contextual information
bool is_functional         # Whether relation is functionally important
string[] implications      # What this relation implies for manipulation