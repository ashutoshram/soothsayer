import cv2
import numpy as np
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
from ultralytics import YOLO
import supervision as sv
from supervision import BoxAnnotator, TraceAnnotator
from collections import defaultdict

# Initialize OWL-ViT and YOLOv8 models
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
yolov8_model = YOLO("yolov8s.pt")  # YOLOv8 small model

# Set up video input and output
input_video_path = "highlights.mp4"
output_video_path = "output_trace_tracking.mp4"
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Tracking and annotator setup with default settings
tracker = sv.ByteTrack()
box_annotator = BoxAnnotator()
trace_annotator = TraceAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

# Detection parameters
detection_threshold = 0.3
frame_skip = 50  # Process every 5th frame for efficiency
confirmation_threshold = 0.5  # IoU threshold for confirmation
text_queries = ["NBA Player"]

# Dictionary to keep track of frame counts for each player
track_frame_counts = defaultdict(int)  # Maps tracker_id -> frame count
min_frames_to_confirm = 30  # Minimum frames a player needs to be tracked to be considered confirmed
max_players = 10  # Maximum number of players to track

# Process each frame of the video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Detection
    yolov8_results = yolov8_model(frame)
    
    # Convert YOLOv8 results to Supervision's Detections format with class_id set to 0 by default
    detections = sv.Detections.from_ultralytics(yolov8_results[0])
    detections.class_id = np.zeros(len(detections.xyxy), dtype=int)  # Assign a default class_id of 0

    # Filter YOLO detections based on confidence threshold
    confidence_mask = detections.confidence > detection_threshold
    filtered_boxes = detections.xyxy[confidence_mask]
    filtered_confidences = detections.confidence[confidence_mask]
    filtered_class_ids = detections.class_id[confidence_mask]

    # Re-create `detections` with filtered data to ensure consistent array lengths
    detections = sv.Detections(xyxy=filtered_boxes, confidence=filtered_confidences, class_id=filtered_class_ids)

    # OWL-ViT Detection for confirmation on every nth frame
    if frame_idx % frame_skip == 0:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=text_queries, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = owlvit_model(**inputs)

        # Process OWL-ViT results
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold)
        owlvit_boxes = [list(map(int, box.tolist())) for box in results[0]["boxes"]]

        # Confirm YOLO detections based on OWL-ViT results
        confirmed_boxes = []
        confirmed_scores = []
        for i, (box, score) in enumerate(zip(detections.xyxy, detections.confidence)):
            for owl_box in owlvit_boxes:
                # Calculate Intersection over Union (IoU)
                x1, y1, x2, y2 = box
                ox1, oy1, ox2, oy2 = owl_box
                xi1, yi1 = max(x1, ox1), max(y1, oy1)
                xi2, yi2 = min(x2, ox2), min(y2, oy2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                yolo_area = (x2 - x1) * (y2 - y1)
                owl_area = (ox2 - ox1) * (oy2 - oy1)
                union_area = yolo_area + owl_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # Confirm the detection if IoU is above the threshold
                if iou >= confirmation_threshold:
                    confirmed_boxes.append(box)
                    confirmed_scores.append(score)
                    break  # Only need one OWL-ViT match to confirm

        # Ensure consistent array sizes for detections with confirmed data
        if confirmed_boxes:
            detections = sv.Detections(xyxy=np.array(confirmed_boxes), confidence=np.array(confirmed_scores))
            detections.class_id = np.zeros(len(detections.xyxy), dtype=int)  # Set class_id for confirmed detections

    # Use ByteTrack to track confirmed detections
    if len(detections.xyxy) > 0:  # Only track if there are confirmed detections
        tracked_detections = tracker.update_with_detections(detections)

        # Update frame count for each tracked player
        for tracker_id in tracked_detections.tracker_id:
            track_frame_counts[tracker_id] += 1

        # Filter out players who have not been tracked for enough frames and limit to 10 players
        confirmed_tracked_detections = sv.Detections(
            xyxy=tracked_detections.xyxy[
                [track_frame_counts[tracker_id] >= min_frames_to_confirm for tracker_id in tracked_detections.tracker_id]
            ][:max_players],
            confidence=tracked_detections.confidence[
                [track_frame_counts[tracker_id] >= min_frames_to_confirm for tracker_id in tracked_detections.tracker_id]
            ][:max_players],
            class_id=tracked_detections.class_id[
                [track_frame_counts[tracker_id] >= min_frames_to_confirm for tracker_id in tracked_detections.tracker_id]
            ][:max_players],
            tracker_id=tracked_detections.tracker_id[
                [track_frame_counts[tracker_id] >= min_frames_to_confirm for tracker_id in tracked_detections.tracker_id]
            ][:max_players]
        )

        # Annotate frame with confirmed tracked detections
        annotated_frame = frame.copy()  # Start with a clean frame
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=confirmed_tracked_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=confirmed_tracked_detections,
            labels=[f"Player {tracker_id}" for tracker_id in confirmed_tracked_detections.tracker_id]
        )

        # Write annotated frame to output video
        out.write(annotated_frame)
        cv2.imshow("Trace Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
