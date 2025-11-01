# SceneGraph.msg
# Custom message for scene graph representation

std_msgs/Header header

# Scene metadata
string scene_id                   # Unique scene identifier
int32 num_objects                # Total number of objects
float32 scene_confidence         # Overall scene understanding confidence

# Objects in scene
SemanticObject[] objects         # List of all detected objects

# Spatial relationships
SpatialRelation[] relations      # Relationships between objects

# Global scene properties
string scene_type               # "workshop", "kitchen", "office", etc.
string[] scene_affordances      # What actions are possible in this scene
geometry_msgs/Point table_plane # Main work surface (if detected)
bool has_support_surface       # Whether a support surface exists

# Temporal information
builtin_interfaces/Time timestamp     # When scene was captured
float32 processing_time_ms            # Time taken to process scene