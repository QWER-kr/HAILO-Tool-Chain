# HAILO-Tool-Chain

## Environments
1. YOLOv8 Nano
2. Roboflow Car Dataset (universitas-diponegoro-mb10s)
3. Calibration on Roboflow Car Dataset

### Implementation
```
import os
import numpy as np
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

model_name = "yolov8"
image_dir = "../datasets/Car-detection-1/train/images" # Edit path
output_dir = "./" + model_name + "_calibration"
os.makedirs(output_dir, exist_ok=True)

calibration_path = os.path.join(output_dir, "calibration_data.npy")
processed_path = os.path.join(output_dir, "processed_calibration_data.npy")

# dataset prepare
rf = Roboflow(api_key="nScixiYjRzng10mSk5IB") # Edit API Key
project = rf.workspace("universitas-diponegoro-mb10s").project("car-detection-nxsxm")
version = project.version(1)
dataset = version.download("yolov8")

# YOLO load, train, and export
model = YOLO("yolov8n.pt")
model.train(data="../datasets/Car-detection-1/data.yaml", epochs=3)
model.export(format="onnx")

# Calibration and Save to the .npz file
calib_data = []
for img in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img)
    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(img_path).resize((640, 640))
        img_np = np.array(img) / 255.0
        calib_data.append(img_np)

calib_data = np.array(calib_data)

np.save(calibration_path, calib_data)
print(f"Normalized calibration dataset saved with shape: {calib_data.shape} to {calibration_path}")

processed_calibration_data = calib_data * 255.0
np.save(processed_path, processed_calibration_data)
print(f"Processed calibration dataset saved with shape: {processed_calibration_data.shape} to {processed_path}")
```

## Convert to HAILO ".har" Format

### Linux Setting
```
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-tk libfuse2 graphviz libgraphviz-dev

# Update alternatives to point to Python 3.10 if version does not match the 3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --config python3  # You'll need to select Python 3.10 manually here.

# Upgrade pip and install virtualenv
python3.10 -m pip install --upgrade pip virtualenv
```

### Create Python Virtual Environment
```
# Create a virtual environment using Python 3.10
python3.10 -m virtualenv my_env

# Install the Hailo Dataflow Compiler WHL file
my_env/bin/pip install MyDrive/hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl

# Making sure it's installed properly
my_env/bin/hailo --version
```

### Optimization YOLO
```
my_env/bin/python translate_model.py
my_env/bin/python inspect_dict.py
my_env/bin/python nms.py
my_env/bin/python optimize_model.py
my_env/bin/python resource_monitor.py
```

### Check the Quantization YOLO
```
my_env/bin/hailo profiler yolov8n_quantized_model.har
```
