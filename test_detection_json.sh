#!/bin/bash
# Test script to call detection service and show JSON output

echo "================================================"
echo "🔍 Calling SAM Detection Service..."
echo "================================================"
echo ""

# Call the service and capture output
output=$(ros2 service call /vision/detect_objects std_srvs/srv/Trigger 2>&1)

# Extract just the message part (the JSON)
json_part=$(echo "$output" | sed -n '/message:/,$ p' | sed '1d')

echo "📊 Detection Results (JSON Schema):"
echo "================================================"
echo "$json_part"
echo ""
echo "================================================"
echo "✅ Detection complete!"
echo ""
echo "💡 Tip: Pipe to jq for pretty formatting:"
echo "   ros2 service call /vision/detect_objects std_srvs/srv/Trigger | grep -A 1000 'message:' | sed '1d' | jq ."
