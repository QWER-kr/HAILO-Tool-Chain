import json
import os
# Updated NMS layer configuration dictionary for YOLOv8
nms_layer_config = {
    "nms_scores_th": 0.3,
    "nms_iou_th": 0.7,
    "image_dims": [640, 640],
    "max_proposals_per_class": 25,
    "classes": 1,  # Updated number of classes
    "regression_length": 16,
    "background_removal": False,
    "background_removal_index": 0,
    "bbox_decoders": [
        {
            "name": "bbox_decoder41",
            "stride": 8,
            "reg_layer": "conv41",
            "cls_layer": "conv42"
        },
        {
            "name": "bbox_decoder52",
            "stride": 16,
            "reg_layer": "conv52",
            "cls_layer": "conv53"
        },
        {
            "name": "bbox_decoder62",
            "stride": 32,
            "reg_layer": "conv62",
            "cls_layer": "conv63"
        }
    ]
}

# Path to save the updated JSON configuration
output_dir = "/nms_configs/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "yolov8_nms_layer_config.json")

# Save the updated configuration as a JSON file
with open(output_path, "w") as json_file:
    json.dump(nms_layer_config, json_file, indent=4)

print(f"NMS layer configuration saved to {output_path}")
