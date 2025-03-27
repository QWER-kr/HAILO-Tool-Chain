# HAILO-Tool-Chain

## Preparation
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

### Overall Flow
1. ONNX or TFLite Model의 파일, 입력 및 출력 Node 정보를 HAILO Compile에 넘겨주어 **.har** format으로 변환한다.
```
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l")

end_node_names = [
    '/model.22/cv2.0/cv2.0.2/Conv',  # P3 regression_layer
    '/model.22/cv3.0/cv3.0.2/Conv',  # P3 cls_layer
    '/model.22/cv2.1/cv2.1.2/Conv',  # P4 regression_layer
    '/model.22/cv3.1/cv3.1.2/Conv',  # P4 cls_layer
    '/model.22/cv2.2/cv2.2.2/Conv',  # P5 regression_layer
    '/model.22/cv3.2/cv3.2.2/Conv',  # P5 cls_layer
]

net_input_shape = {"images": [1, 3, 640, 640]}

hn, npz = runner.translate_onnx_model(
    "model.onnx", # onnx path
    "yolov8", # model name
    end_node_names, # output node name
    input_shape, # model input node shape
)

runner.save_har = "translate_model.har"

```
2. (Optional) converted **.har** model 정보를 dictionary 형태로 확인한다.
```
from pprint import pprint
from hailo_sdk_client import ClientRunner

runner = ClientRunner(har="translate_model.har")

hn_dict = runner.get_hn()

for key, value in hn_dict.items()
    print(f"Key: {key}")
    pprint(value)
```
3. (Detection model 한정) NMS(Non-Maximum Suppression)을 통해서 중복 bounding boxes를 제거한다.
(Hailo Dataflow Compile 공식 문서 설명)
- NMS는 겹치는 여러 entities 중에서 최종 entities(예: 바운딩 박스)를 선택하여 object detector의 예측을 필터링하는 데 사용되는 기술입니다. 이는 점수 threshold와 IoU(Intersection over Union, 겹치는 박스 필터링)의 두 단계로 구성됩니다. 
- NMS 알고리즘에는 네트워크 출력에서 계산된 bounding box를 제공해야 합니다. 이 프로세스를 “bbox decoding”이라고 하며, 네트워크 출력을 박스 좌표로 수학적으로 변환하는 과정으로 구성됩니다. 
- bbox decoding 계산은 구현마다 크게 다를 수 있으며, 다양한 유형의 수학 연산(pow, exp, log 등)이 포함됩니다.
```
import json
import os

# NMS layer configuration 정의
nms_layer_config = {
    "nms_scores_th": 0.3,                     # 객체로 인식할 최소 confidence score threshold (중복 bounding box 제거)
    "nms_iou_th": 0.7,                        # Threshold 보다 높은 IoU는 제거 (중복 bounding box 제거)
    "image_dims": [640, 640],                 # Model이 처리하는 image resolution
    "max_proposals_per_class": 25,            # 한 class 당 NMS 후 남길 bounding box 수
    "classes": 1,                             # Number of classes
    "regression_length": 16,                  # Bounding box를 표현하는 regression vector 길 (여기서는 anchor box 당 16개의 값)
    "background_removal": False,              # 배경 제거 유무 (False 시 배경 고려)
    "background_removal_index": 0,            # 배경 class index (배경을 제거할 경우, 해당 index를 가진 box 제)
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

output_dir = "/nms_configs/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "yolov8_nms_layer_config.json")

with open(output_path, "w") as json_file:
    json.dump(nms_layer_config, json_file, indent=4)

```
4. **.har** model 4-bit precision으로 quantization

```
import os
from hailo_sdk_client import ClientRunner

model_name = "yolov8n_renamed"
hailo_model_har_name = f"{model_name}_hailo_model.har"

assert os.path.isfile(hailo_model_har_name), "Please provide a valid path for the HAR file"

runner = ClientRunner(har=hailo_model_har_name)

# Hailo Operation Graph를 수정하는 부분
alls = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])                        # Input data normalization
change_output_activation(conv42, sigmoid)                                                     # Ouput node에 Sigmoid activation 연결 (Class Categoty Score에 대한 연산 및 NMS 단계와 연결을 위한 Sigmoid 연결)
change_output_activation(conv53, sigmoid)                                                     # Ouput node에 Sigmoid activation 연결 (Class Categoty Score에 대한 연산 및 NMS 단계와 연결을 위한 Sigmoid 연결)
change_output_activation(conv63, sigmoid)                                                     # Ouput node에 Sigmoid activation 연결 (Class Categoty Score에 대한 연산 및 NMS 단계와 연결을 위한 Sigmoid 연결)
nms_postprocess("./nms_configs/yolov8_nms_layer_config.json", meta_arch=yolov8, engine=cpu)   # 앞서 설정한 Non-Maximum Suppression(NMS) configuration JSON 파일에서 불러와 적용.
performance_param(compiler_optimization_level=max)                                            # Hailo compiler optimization lavel을  max로 설정
"""

runner.load_model_script(alls)  # 저의 alls script 기반으로 model에 적용

calib_dataset = "./yolov8_cal/processed_calibration_data.npy" # 학습에 사용된 dataset

runner.optimize(calib_dataset)    # Calibration dataset을 기반으로 8-bit quantization 수행

quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)
```
5. Hailo NPU에서 실행 가능한 **.hef** format 파일 생성 (30분정도 시간 소요)
```
from hailo_sdk_client import ClientRunner
import os

model_name = "yolov8n_renamed"
quantized_model_har_path = f"{model_name}_quantized_model.har"

runner = ClientRunner(har=quantized_model_har_path)

hef = runner.compile()

file_name = f"{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)
```

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

## Error

### CASE : FP32 Model Convert to hef
**optimzied_model.py**를 통해서 모델을 8-bit로 quantization하여야 **.hef** format으로 변환 가능한 것 같음
```
[error] Failed to compile the model: Model requires quantized weights in order to run on HW, but none were given. Did you forget to quantize?
Traceback (most recent call last):
  File "/home/hailo/compile_hef.py", line 15, in <module>
    hef = runner.compile()
  File "/home/hailo/my_env/lib/python3.10/site-packages/hailo_sdk_client/runner/client_runner.py", line 896, in compile
    return self._compile()
  File "/home/hailo/my_env/lib/python3.10/site-packages/hailo_sdk_common/states/states.py", line 16, in wrapped_func
    return func(self, *args, **kwargs)
  File "/home/hailo/my_env/lib/python3.10/site-packages/hailo_sdk_client/runner/client_runner.py", line 1113, in _compile
    serialized_hef = self._sdk_backend.compile(fps, self.model_script, mapping_timeout)
  File "/home/hailo/my_env/lib/python3.10/site-packages/hailo_sdk_client/sdk_backend/sdk_backend.py", line 1763, in compile
    hef, mapped_graph_file = self._compile(fps, allocator_script, mapping_timeout)
  File "/home/hailo/my_env/lib/python3.10/site-packages/hailo_sdk_client/sdk_backend/sdk_backend.py", line 1752, in _compile
    raise BackendRuntimeException(
hailo_sdk_client.sdk_backend.sdk_backend_exceptions.BackendRuntimeException: Model requires quantized weights in order to run on HW, but none were given. Did you forget to quantize?
```
