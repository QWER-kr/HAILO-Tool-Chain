
from hailo_sdk_client import ClientRunner

# Define the ONNX model path and configuration
onnx_path = "/content/best.onnx"
onnx_model_name = "yolov8n_renamed"
chosen_hw_arch = "hailo8l"  # Specify the target hardware architecture

# Initialize the ClientRunner
runner = ClientRunner(hw_arch=chosen_hw_arch)

# Use the recommended end node names for YOLOv8n
end_node_names = [
    '/model.22/cv2.0/cv2.0.2/Conv',  # P3 regression_layer
    '/model.22/cv3.0/cv3.0.2/Conv',  # P3 cls_layer
    '/model.22/cv2.1/cv2.1.2/Conv',  # P4 regression_layer
    '/model.22/cv3.1/cv3.1.2/Conv',  # P4 cls_layer
    '/model.22/cv2.2/cv2.2.2/Conv',  # P5 regression_layer
    '/model.22/cv3.2/cv3.2.2/Conv',  # P5 cls_layer
]
net_input_shapes = {"images": [1, 3, 640, 640]}  # Adjust input shapes if needed

try:
    # Translate the ONNX model to Hailo's format
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        end_node_names=end_node_names,
        net_input_shapes=net_input_shapes,  # Adjust input shapes if needed
    )
    print("Model translation successful.")
except Exception as e:
    print(f"Error during model translation: {e}")
    raise

# Save the Hailo model HAR file
hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
try:
    runner.save_har(hailo_model_har_name)
    print(f"HAR file saved as: {hailo_model_har_name}")
except Exception as e:
    print(f"Error saving HAR file: {e}")
    