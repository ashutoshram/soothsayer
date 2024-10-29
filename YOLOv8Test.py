from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your model file

# Load the video file
cap = cv2.VideoCapture("highlights.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output video configuration
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Draw boxes on the frame
    annotated_frame = results[0].plot()  # Plot detected boxes on the frame

    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow("Detections", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()