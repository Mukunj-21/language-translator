import cv2
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os
from safetensors.torch import load_file

class SignLanguageTranslator:
    def __init__(self, model_path="models/sign_language_model"):
        """
        Initialize the sign language translator.
        
        Args:
            model_path: Path to the local model. If None, will download from HuggingFace.
        """
        if model_path and os.path.exists(model_path):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            # self.model = AutoModelForImageClassification.from_pretrained(model_path)
            model_name_or_config = AutoModelForImageClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            # safetensors=True
            )
            self.model = model_name_or_config
        else:
            # Use the HuggingFace model
            model_name = "Heem2/sign-language-classification"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            # self.model = AutoModelForImageClassification.from_pretrained(model_name)
            model_name_or_config = AutoModelForImageClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            safetensors=True
        )
        self.model = model_name_or_config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Map of indices to letters/signs
        self.idx_to_label = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
            6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
            12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
            24: 'Y', 25: 'Z'
        }
        
        # Initialize hand detection
        self.hand_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        if self.hand_detector.empty():
            # Fallback to MediaPipe hands if available
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                self.using_mediapipe = True
            except ImportError:
                print("Warning: Neither OpenCV hand cascade nor MediaPipe is available.")
                self.using_mediapipe = False
        else:
            self.using_mediapipe = False
        
        # Frame skip rate for video processing
        self.frame_skip = 5
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for the model."""
        # Convert to RGB (the model expects RGB)
        if len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Resize to expected input size
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # Use the feature extractor to prepare the input
        inputs = self.feature_extractor(images=frame_resized, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def detect_hand(self, frame):
        """Detect hand in the frame and return the hand region."""
        if self.using_mediapipe:
            # Use MediaPipe for hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Get bounding box of the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                return frame[y_min:y_max, x_min:x_max]
        else:
            # Try basic color-based segmentation for hand detection
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (presumably the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                return frame[y:y+h, x:x+w]
        
        # If no hand detected, return the original frame
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame and return the detected sign.
        
        Args:
            frame: The input frame containing a sign gesture
            
        Returns:
            Detected sign character or None if no confident detection
        """
        # Detect hand in the frame
        hand_region = self.detect_hand(frame)
        
        if hand_region is None or hand_region.size == 0:
            return None
        
        # Preprocess the frame
        inputs = self.preprocess_frame(hand_region)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
        
        # Only return prediction if confidence is high enough
        if confidence > 0.7:  # Adjust threshold as needed
            return self.idx_to_label.get(predicted_class_idx)
        
        return None
    
    def process_video(self, video_path):
        """
        Process a video file and return the detected signs.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detected text and confidence
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        detected_signs = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to speed up processing
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
            
            # Process the frame
            sign = self.process_frame(frame)
            if sign:
                detected_signs.append(sign)
        
        cap.release()
        
        # Simple post-processing to combine consecutive same letters
        processed_signs = []
        for i, sign in enumerate(detected_signs):
            if i == 0 or sign != detected_signs[i-1]:
                processed_signs.append(sign)
        
        detected_text = ''.join(processed_signs)
        
        return {
            "detected_text": detected_text,
            "confidence": 0.8  # Placeholder, could calculate actual confidence
        }
    
    def is_thumb_sign(self, frame):
        """
        Detect if the frame contains a thumb sign (space).
        This is a simplified implementation and might need refinement.
        """
        # Detect hand in the frame
        hand_region = self.detect_hand(frame)
        
        if hand_region is None or hand_region.size == 0:
            return False
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get the largest contour (presumably the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Check the aspect ratio and size of the contour to identify thumb gesture
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h
        
        # A thumb up gesture typically has a low aspect ratio (taller than wide)
        # This is a simple heuristic and might need adjustment
        return aspect_ratio < 0.5 and h > w * 2
    
    def is_sentence_complete_sign(self, frame):
        """
        Detect if the frame contains a sign for sentence completion.
        For simplicity, we'll use a different gesture (e.g., palm facing camera).
        """
        # Similar implementation to thumb sign but with different criteria
        hand_region = self.detect_hand(frame)
        
        if hand_region is None or hand_region.size == 0:
            return False
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get the largest contour (presumably the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calculate convex hull and defects to detect open palm
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
        
        # Count number of defects (spaces between fingers)
        defect_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 1000:  # distance threshold
                    defect_count += 1
        
        # An open palm typically has 4 significant defects (between 5 fingers)
        return defect_count >= 4
