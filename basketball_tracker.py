import torch
from PIL import Image
import clip
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collections import defaultdict
import sys

class BasketballPlayerTracker:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
       
        # Initialize player database with recognizable features
        self.players = {
            "lebron_james": {
                "name": "LeBron James",
                "descriptions": [
                    "LeBron James playing basketball",
                    "LeBron James number 23",
                    "LeBron James number 6",
                    "LeBron James with Lakers jersey",
                    "LeBron James dunking"
                ],
                "team": "Lakers",
                "number": "23"
            },
            "stephen_curry": {
                "name": "Stephen Curry",
                "descriptions": [
                    "Stephen Curry playing basketball",
                    "Stephen Curry number 30",
                    "Stephen Curry with Warriors jersey",
                    "Stephen Curry shooting three pointer"
                ],
                "team": "Warriors",
                "number": "30"
            },
            # Add more players as needed
        }
       
        # Initialize tracking history
        self.tracking_history = defaultdict(list)
       
        # Initialize YOLO model for person detection
        self.person_detector = cv2.dnn.readNetFromONNX("yolov8n.onnx")
       
        # Encode player descriptions
        self.player_features = self._encode_player_descriptions()
       
    def _encode_player_descriptions(self):
        """Encode all player descriptions using CLIP"""
        all_descriptions = []
        for player_data in self.players.values():
            all_descriptions.extend(player_data["descriptions"])
           
        text_tokens = clip.tokenize(all_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
   
    def detect_players(self, frame):
        """
        Detect people in the frame using YOLO
        Returns list of bounding boxes
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
        self.person_detector.setInput(blob)
        detections = self.person_detector.forward()
       
        boxes = []
        confidences = []
        print(detections.shape)
        for detection in detections[0]:  # Loop over detections
            x_center, y_center, width, height, confidence, class_id = detection[:6]
            if confidence > 0.5:  # Filter by confidence threshold
                # Convert center x,y to top-left x,y for bounding box display
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)
                w = int(width)
                h = int(height)
                boxes.append((x, y, w, h))
        return boxes
   
    def identify_player(self, frame, box):
        """
        Identify specific player from detected person using CLIP
        """
        x, y, w, h = box
        person_img = frame[y:y+h, x:x+w]
        if person_img.size == 0:
            return None, 0
           
        # Convert to PIL and preprocess
        person_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(person_img).unsqueeze(0).to(self.device)
       
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
           
        # Calculate similarities with all player descriptions
        similarities = (100.0 * image_features @ self.player_features.T).softmax(dim=-1)
        similarities = similarities[0].cpu().numpy()
       
        # Map similarities back to players
        player_scores = defaultdict(list)
        idx = 0
        for player_id, player_data in self.players.items():
            num_descriptions = len(player_data["descriptions"])
            player_scores[player_id] = similarities[idx:idx + num_descriptions].mean()
            idx += num_descriptions
           
        # Get best match
        best_player = max(player_scores.items(), key=lambda x: x[1])
        return best_player if best_player[1] > 0.3 else (None, 0)
   
    def track_players(self, frame):
        """
        Track and identify players in a single frame
        """
        # Detect people
        boxes = self.detect_players(frame)
       
        # Identify each detected person
        frame_results = []
        for box in boxes:
            player_id, confidence = self.identify_player(frame, box)
            if player_id:
                frame_results.append({
                    'player_id': player_id,
                    'player_name': self.players[player_id]['name'],
                    'box': box,
                    'confidence': confidence
                })
                self.tracking_history[player_id].append(box)
       
        return frame_results
   
    def analyze_player_movement(self, player_id, frame_width, frame_height):
        """
        Analyze movement patterns for a specific player
        """
        if player_id not in self.tracking_history or not self.tracking_history[player_id]:
            return None
           
        movements = []
        positions = self.tracking_history[player_id]
       
        for i in range(1, len(positions)):
            prev_box = positions[i-1]
            curr_box = positions[i]
           
            # Calculate center points
            prev_center = (prev_box[0] + prev_box[2]//2, prev_box[1] + prev_box[3]//2)
            curr_center = (curr_box[0] + curr_box[2]//2, curr_box[1] + curr_box[3]//2)
           
            # Analyze court position
            court_position = self._get_court_position(curr_center, frame_width, frame_height)
           
            # Calculate movement vector
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
           
            movements.append({
                'court_position': court_position,
                'movement_vector': (dx, dy),
                'frame_position': curr_center
            })
           
        return movements
   
    def _get_court_position(self, center_point, frame_width, frame_height):
        """
        Determine player's position on court based on frame coordinates
        """
        x, y = center_point
        x_rel = x / frame_width
        y_rel = y / frame_height
       
        # Basic court position mapping
        if x_rel < 0.33:
            x_pos = "left"
        elif x_rel < 0.66:
            x_pos = "middle"
        else:
            x_pos = "right"
           
        if y_rel < 0.5:
            y_pos = "frontcourt"
        else:
            y_pos = "backcourt"
           
        return f"{y_pos}_{x_pos}"
    
   
    def visualize_tracking(self, frame, results):
        """
        Visualize tracking results on frame
        """
        viz_frame = frame.copy()
       
        for result in results:
            print(result)
            box = result['box']
            player_name = result['player_name']
            confidence = result['confidence']
           
            # Draw bounding box
            x, y, w, h = box
            cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
           
            # Add player name and confidence
            label = f"{player_name}: {confidence:.2%}"
            cv2.putText(viz_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
            # Draw movement trail
            history = self.tracking_history[result['player_id']][-10:]  # Last 10 positions
            for i in range(1, len(history)):
                prev_box = history[i-1]
                curr_box = history[i]
                prev_center = (prev_box[0] + prev_box[2]//2, prev_box[1] + prev_box[3]//2)
                curr_center = (curr_box[0] + curr_box[2]//2, curr_box[1] + curr_box[3]//2)
                cv2.line(viz_frame, prev_center, curr_center, (255, 0, 0), 2)
               
        return viz_frame

def main():
    # Initialize tracker
    tracker = BasketballPlayerTracker()
   
    # Example: Process video
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
   
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    # Initialize video writer
    out = cv2.VideoWriter('output.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30,
                         (frame_width, frame_height))
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
           
        # Track players in frame
        results = tracker.track_players(frame)
       
        # Visualize results
        viz_frame = tracker.visualize_tracking(frame, results)
       
        # Write frame
        cv2.imshow( "Visualization", viz_frame)
        cv2.waitKey(0)

        out.write(viz_frame)
       
        # Analysis for specific player
        for player_id in tracker.players.keys():
            movements = tracker.analyze_player_movement(player_id, frame_width, frame_height)
            if movements:
                print(f"\nMovement analysis for {tracker.players[player_id]['name']}:")
                print(f"Court positions: {[m['court_position'] for m in movements[-5:]]}")
   
    cap.release()
    out.release()

if __name__ == "__main__":
    main()