
import os
from hailo_sdk_client import ClientRunner

# Define your model's HAR file name
model_name = "yolov8n_renamed"
hailo_model_har_name = f"{model_name}_hailo_model.har"

# Ensure the HAR file exists
assert os.path.isfile(hailo_model_har_name), "Please provide a valid path for the HAR file"

# Initialize the ClientRunner with the HAR file
runner = ClientRunner(har=hailo_model_har_name)

# Define the model script to add a normalization layer
# Normalization for [0, 1] range
alls = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("/content/drive/MyDrive/nms_configs/yolov8_nms_layer_config.json", meta_arch=yolov8, engine=cpu)
performance_param(compiler_optimization_level=max)
"""

# Load the model script into the ClientRunner
runner.load_model_script(alls)

# Define a calibration dataset
# Replace 'calib_dataset' with the actual dataset you're using for calibration
# For example, if it's a directory of images, prepare the dataset accordingly
calib_dataset = "./yolov8_cal/processed_calibration_data.npy"

# Perform optimization with the calibration dataset
runner.optimize(calib_dataset)

# Save the optimized model to a new Quantized HAR file
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)

print(f"Quantized HAR file saved to: {quantized_model_har_path}")
    