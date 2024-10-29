import cv2
import torch
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor

# Load OWL-ViT model and processor
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Load SAM model
sam_checkpoint_path = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
predictor = SamPredictor(sam)

# Set up video input and output
input_video_path = "highlights.mp4"
output_video_path = "output.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the text prompt for object detection
text_queries = ["NBA Player", "Referee"]

# Process each frame of the video
# Parameters for tracking
max_players = 10
detection_threshold = 0.3
confirmation_threshold = 2  # Number of times a player needs to be detected before tracking
trackers = []
confirmed_players = {}
player_appearances = {}

# Define the text prompt for player detection
text_queries = ["NBA Player", "Referee", "Person", "Stephen Curry"]

# Process each frame of the video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image for compatibility with OWL-ViT
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # If we have fewer than 10 players being tracked, continue detecting
    frame_skip = 5
    if len(trackers) < max_players:
        # Prepare inputs for OWL-ViT
        inputs = processor(text=text_queries, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the detections
        boxes = []
        scores = []
        labels = []
        if frame_idx % frame_skip == 0:
            target_sizes = torch.Tensor([pil_image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold)

        # Extract boxes, scores, and labels
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        # Check each detection
        for box, score, label in zip(boxes, scores, labels):
            if text_queries[label] == "NBA Player" and score > detection_threshold:
                # Convert box to absolute coordinates
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                
                # Check if the detected box is close to any tracked box
                found_nearby = False
                for key in list(player_appearances.keys()):
                    x_min_tracked, y_min_tracked, x_max_tracked, y_max_tracked = player_appearances[key]['box']
                    if (abs(x_min - x_min_tracked) < 20 and
                        abs(y_min - y_min_tracked) < 20 and
                        abs(x_max - x_max_tracked) < 20 and
                        abs(y_max - y_max_tracked) < 20):
                        
                        # Increment appearance count if close to an existing player
                        player_appearances[key]['count'] += 1
                        found_nearby = True
                        break

                # If no nearby match found, create a new player entry
                if not found_nearby:
                    player_appearances[(x_min, y_min, x_max, y_max)] = {'box': (x_min, y_min, x_max, y_max), 'count': 1}

                # Start tracking once a player has been confirmed by multiple appearances
                for key, value in list(player_appearances.items()):
                    if value['count'] >= confirmation_threshold and key not in confirmed_players:
                        # Initialize a tracker for this confirmed player
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (value['box'][0], value['box'][1], value['box'][2] - value['box'][0], value['box'][3] - value['box'][1]))
                        trackers.append(tracker)
                        confirmed_players[key] = value['box']

                        # Draw confirmation box
                        draw.rectangle([value['box'][0], value['box'][1], value['box'][2], value['box'][3]], outline="yellow", width=2)
                        draw.text((value['box'][0], value['box'][1] - 10), "Confirmed Player", fill="yellow")

                        # Stop adding if we have reached the max number of players
                        if len(trackers) >= max_players:
                            break

    # Update and draw each tracker for confirmed players
    active_trackers = []
    for tracker, (x_min, y_min, x_max, y_max) in zip(trackers, confirmed_players.values()):
        success, new_box = tracker.update(frame)
        if success:
            active_trackers.append(tracker)
            x, y, w, h = map(int, new_box)
            draw.rectangle((x, y, x + w, y + h), outline="green", width=2)
            draw.text((x, y - 10), "Tracked Player", fill="green")

    # Replace old tracker list with active trackers
    trackers = active_trackers

    # Convert PIL image back to OpenCV format for video output
    frame_with_tracking = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    out.write(frame_with_tracking)
    
    # Display the frame (optional)
    cv2.imshow("Court Tracking", frame_with_tracking)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()