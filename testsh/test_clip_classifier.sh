#!/bin/bash
# Test script for CLIP classifier

echo "================================================"
echo "🤖 CLIP Image Classifier Test"
echo "================================================"
echo ""

# Check if service exists
echo "🔍 Checking if CLIP classifier is running..."
if ! ros2 service list | grep -q "/vision/classify_image"; then
    echo "❌ Service /vision/classify_image not found!"
    echo ""
    echo "Start the classifier first:"
    echo "  ros2 run vision clip_classifier"
    exit 1
fi

echo "✅ CLIP classifier service found"
echo ""

# Call the service
echo "📸 Calling classification service..."
echo "================================================"
output=$(ros2 service call /vision/classify_image std_srvs/srv/Trigger 2>&1)

# Extract JSON part
json_part=$(echo "$output" | sed -n '/message:/,$ p' | sed '1d')

echo "$json_part"
echo ""
echo "================================================"

# Parse and display key info
if command -v jq &> /dev/null; then
    echo ""
    echo "📊 Quick Summary:"
    echo "================================================"
    
    top_label=$(echo "$json_part" | jq -r '.output.top_prediction.label')
    top_conf=$(echo "$json_part" | jq -r '.output.top_prediction.confidence')
    proc_time=$(echo "$json_part" | jq -r '.output.metadata.processing_time_ms')
    device=$(echo "$json_part" | jq -r '.output.metadata.device')
    
    echo "🏆 Top Prediction: $top_label ($top_conf)"
    echo "⏱️  Processing Time: ${proc_time}ms"
    echo "💻 Device: $device"
    echo ""
    echo "📋 All Predictions:"
    echo "$json_part" | jq -r '.output.all_predictions[] | "   \(.label): \(.confidence)"'
    
    echo ""
    echo "🧮 Embedding Dimensions:"
    img_vec_len=$(echo "$json_part" | jq '.output.embedding.image_vector | length')
    echo "   Image vector: ${img_vec_len}D"
    echo "   Text vectors: 512D each"
else
    echo ""
    echo "💡 Tip: Install jq for better JSON formatting:"
    echo "   sudo apt install jq"
fi

echo ""
echo "================================================"
echo "✅ Classification complete!"
echo ""
echo "📝 Save to file:"
echo "   ros2 service call /vision/classify_image std_srvs/srv/Trigger | \\"
echo "     grep -A 1000 'message:' | sed '1d' > clip_result.json"
