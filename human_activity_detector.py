import torch
from PIL import Image
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import sys

class CLIPActivityDetector:
    def __init__(self):
        # Load the CLIP model and preprocessing pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
       
        # Define common human activities
        self.activities = [
            "person walking",
            "person running",
            "person sitting",
            "person standing",
            "person jumping",
            "person dancing",
            "person eating",
            "person reading",
            "person typing",
            "person playing sports",
            "person cooking",
            "person sleeping"
        ]
       
        # Encode activity text descriptions
        self.text_features = self._encode_activities()
       
    def _encode_activities(self):
        """Encode all activity descriptions using CLIP's text encoder"""
        text_tokens = clip.tokenize(self.activities).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
   
    def detect_activity(self, image_path, threshold=0.25):
        """
        Detect human activities in an image
       
        Args:
            image_path (str): Path to the image file
            threshold (float): Confidence threshold for detection
           
        Returns:
            list: Detected activities and their confidence scores
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
       
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
           
        # Calculate similarities with all activities
        similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        similarities = similarities[0].cpu().numpy()
       
        # Get detected activities above threshold
        detected = []
        for activity, score in zip(self.activities, similarities):
            if score > threshold:
                detected.append({
                    'activity': activity,
                    'confidence': float(score)
                })
       
        return sorted(detected, key=lambda x: x['confidence'], reverse=True)
   
    def add_custom_activities(self, new_activities):
        """
        Add custom activities to the detector
       
        Args:
            new_activities (list): List of new activity descriptions
        """
        self.activities.extend(new_activities)
        self.text_features = self._encode_activities()

# Example usage
def main():
    # Initialize detector
    detector = CLIPActivityDetector()
   
    # Example of adding custom activities
    custom_activities = [
        "person exercising",
        "person studying",
        "person playing guitar"
    ]
    detector.add_custom_activities(custom_activities)
   
    # Detect activities in an image
    image_path = sys.argv[1]
    detections = detector.detect_activity(image_path, threshold=0.25)
   
    # Print results
    print("\nDetected Activities:")
    for detection in detections:
        print(f"{detection['activity']}: {detection['confidence']:.2%}")

if __name__ == "__main__":
    main()