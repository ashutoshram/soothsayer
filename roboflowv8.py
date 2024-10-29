import cv2
import numpy as np
import requests
import supervision as sv
from ultralytics import YOLO
from roboflow import Roboflow

# Initialize Roboflow and load the YOLO model
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("your_project_name")
model = project.version("your_version_number").model

# Set up video input/output
input_video_path = "highlights.mp4"
output_video_path = "output_yolo_tracking.mp4"
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Supervision box annotator
box_annotator = sv.BoxAnnotator(thickness=2, color=(0, 255, 0), text_color=(255, 255, 255))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # YOLO Detection using Roboflow model
    pil_image = sv.PILImage.fromarray(frame_rgb)  # Convert to PIL format for YOLO
    results = model.predict(pil_image)
    detections = results.json()["predictions"]

    # Process detections and draw bounding boxes
    boxes, labels = [], []
    for detection in detections:
        x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
        class_name = detection["class"]
        
        # YOLO bounding box format: [x_center, y_center, width, height]
        x_min, y_min = int(x - w / 2), int(y - h / 2)
        x_max, y_max = int(x + w / 2), int(y + h / 2)

        # Add bounding box and label
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(class_name)

    # Annotate frame with bounding boxes
    frame_with_annotations = box_annotator.annotate(frame_rgb, boxes, labels)
    
    # Convert back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame_with_annotations, cv2.COLOR_RGB2BGR)
    
    # Write frame to output video
    out.write(frame_bgr)
    cv2.imshow("YOLO Tracking", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()