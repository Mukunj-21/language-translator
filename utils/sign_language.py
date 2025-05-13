import cv2
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os
import mediapipe as mp

class SignLanguageTranslator:
    def __init__(self, model_path="models/sign_language_model"):
        """
        Initialize the sign language translator.
        
        Args:
            model_path: Path to the local model. If None, will download from HuggingFace.
        """
        try:
            # Try to load local model first
            if model_path and os.path.exists(model_path):
                print(f"Loading model from local path: {model_path}")
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.float32
                )
            else:
                # Fall back to HuggingFace model
                print("Local model not found, downloading from HuggingFace")
                model_name = "Heem2/sign-language-classification"
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Map of indices to letters/signs
        self.idx_to_label = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
            6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
            12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
            24: 'Y', 25: 'Z'
        }
        
        # Initialize MediaPipe hands for reliable hand detection
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("MediaPipe hands initialized successfully")
        except ImportError:
            print("Warning: MediaPipe not available. Hand detection may be less accurate.")
            self.mp_hands = None
        
        # Frame skip rate for video processing
        self.frame_skip = 5
        
        # Detection confidence threshold
        self.confidence_threshold = 0.6
        
        # For gesture recognition
        self.thumb_up_frames = 0
        self.open_palm_frames = 0
        self.required_frames = 3  # Number of consecutive frames to confirm a gesture
    
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
        if self.mp_hands is not None:
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
                
                # Draw landmarks on a copy of the frame for debugging
                debug_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.mp_drawing.draw_landmarks(
                    debug_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Save or display the debug frame if needed
                # cv2.imwrite("debug_hand.jpg", debug_frame)
                
                # Extract the hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                return hand_region, debug_frame
            
            return None, frame
        else:
            # Fallback to skin color segmentation if MediaPipe not available
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (presumably the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Only process if contour is large enough
                if cv2.contourArea(max_contour) > 1000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(max_contour)
                    
                    # Add padding
                    padding = 30
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    # Extract the hand region
                    hand_region = frame[y:y+h, x:x+w]
                    
                    # Draw the rectangle on a copy of the frame for debugging
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    return hand_region, debug_frame
        
        # If no hand detected, return None
        return None, frame
    
    def process_frame(self, frame):
        """
        Process a single frame and return the detected sign.
        
        Args:
            frame: The input frame containing a sign gesture
            
        Returns:
            Detected sign character or None if no confident detection
        """
        # Detect hand in the frame
        hand_region, debug_frame = self.detect_hand(frame)
        
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
        if confidence > self.confidence_threshold:
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
        confidence_sum = 0
        confidence_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to speed up processing
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
            
            # Process the frame
            hand_region, _ = self.detect_hand(frame)
            if hand_region is None or hand_region.size == 0:
                continue
                
            # Preprocess the frame
            inputs = self.preprocess_frame(hand_region)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get prediction
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
            
            if confidence > self.confidence_threshold:
                sign = self.idx_to_label.get(predicted_class_idx)
                detected_signs.append(sign)
                confidence_sum += confidence
                confidence_count += 1
        
        cap.release()
        
        # Simple post-processing to combine consecutive same letters
        processed_signs = []
        for i, sign in enumerate(detected_signs):
            if i == 0 or sign != detected_signs[i-1]:
                processed_signs.append(sign)
        
        detected_text = ''.join(processed_signs)
        
        # Calculate average confidence
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        return {
            "detected_text": detected_text,
            "confidence": avg_confidence
        }
    
    def is_thumb_sign(self, frame):
        """
        Detect if the frame contains a thumb sign (space).
        Using MediaPipe for more accurate detection.
        """
        if self.mp_hands is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                self.thumb_up_frames = 0
                return False
            
            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            
            # Check thumb position relative to other fingers
            thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            # Check if thumb is pointing up
            if thumb_tip.y < thumb_mcp.y and abs(thumb_tip.x - index_mcp.x) < 0.1:
                self.thumb_up_frames += 1
                if self.thumb_up_frames >= self.required_frames:
                    self.thumb_up_frames = 0
                    return True
            else:
                self.thumb_up_frames = 0
        else:
            # Fallback method using contour analysis
            hand_region, _ = self.detect_hand(frame)
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
            if aspect_ratio < 0.5 and h > w * 2:
                self.thumb_up_frames += 1
                if self.thumb_up_frames >= self.required_frames:
                    self.thumb_up_frames = 0
                    return True
            else:
                self.thumb_up_frames = 0
                
        return False
    
    def is_sentence_complete_sign(self, frame):
        """
        Detect if the frame contains a sign for sentence completion (open palm).
        Using MediaPipe for more accurate detection.
        """
        if self.mp_hands is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                self.open_palm_frames = 0
                return False
            
            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            
            # Calculate distances between fingertips and palm
            palm = hand_landmarks[self.mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            
            # Check if all fingers are extended (open palm)
            fingers_extended = (
                thumb_tip.y < palm.y and
                index_tip.y < palm.y and
                middle_tip.y < palm.y and
                ring_tip.y < palm.y and
                pinky_tip.y < palm.y
            )
            
            if fingers_extended:
                self.open_palm_frames += 1
                if self.open_palm_frames >= self.required_frames:
                    self.open_palm_frames = 0
                    return True
            else:
                self.open_palm_frames = 0
        else:
            # Fallback to contour analysis
            hand_region, _ = self.detect_hand(frame)
            if hand_region is None or hand_region.size == 0:
                return False
            
            # Convert to HSV
            hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # Get the largest contour (presumably the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            try:
                # Calculate convex hull and defects to detect open palm
                hull = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexityDefects(max_contour, hull)
                
                # Count number of defects (spaces between fingers)
                defect_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 10000:  # distance threshold
                            defect_count += 1
                
                # An open palm typically has 4 significant defects (between 5 fingers)
                if defect_count >= 4:
                    self.open_palm_frames += 1
                    if self.open_palm_frames >= self.required_frames:
                        self.open_palm_frames = 0
                        return True
                else:
                    self.open_palm_frames = 0
            except:
                self.open_palm_frames = 0
                
        return False