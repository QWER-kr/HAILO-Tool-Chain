
from hailo_sdk_client import ClientRunner

# Load the HAR file
har_path = "/content/yolov8n_renamed_hailo_model.har"

runner = ClientRunner(har=har_path)

from pprint import pprint

try:
    # Access the HailoNet as an OrderedDict
    hn_dict = runner.get_hn()  # Or use runner._hn if get_hn() is unavailable
    print("Inspecting layers from HailoNet (OrderedDict):")

    # Pretty-print each layer
    for key, value in hn_dict.items():
        print(f"Key: {key}")
        pprint(value)
        print("\n" + "="*80 + "\n")  # Add a separator between layers for clarity

except Exception as e:
    print(f"Error while inspecting hn_dict: {e}")
