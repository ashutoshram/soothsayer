import torch
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')  # Replace with your model file

# Export to ONNX
model.export(format="onnx")