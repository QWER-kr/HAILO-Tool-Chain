
from hailo_sdk_client import ClientRunner
import os

# Define the quantized model HAR file
model_name = "yolov8n_renamed"
quantized_model_har_path = f"{model_name}_quantized_model.har"
output_directory = "/content/drive/MyDrive/yolov8_cal"

os.makedirs(output_directory, exist_ok=True)

# Initialize the ClientRunner with the HAR file
runner = ClientRunner(har=quantized_model_har_path)
print("[info] ClientRunner initialized successfully.")

# Compile the model
try:
    hef = runner.compile()
    print("[info] Compilation completed successfully.")
except Exception as e:
    print(f"[error] Failed to compile the model: {e}")
    raise

# Save the compiled model to the specified directory
output_file_path = os.path.join(output_directory, f"{model_name}.hef")
with open(output_file_path, "wb") as f:
    f.write(hef)

print(f"[info] Compiled model saved successfully to {output_file_path}")
