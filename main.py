# import cv2
# import numpy as np
# from collections import defaultdict, deque
# import torch
# import torchvision.transforms as transforms
# from sklearn.metrics.pairwise import cosine_similarity
# from ultralytics import YOLO
# import os
# from typing import Dict, List, Tuple, Optional
# import pickle

# class PlayerReIdentificationSystem:
#     """
#     A comprehensive player re-identification system that combines object detection
#     with feature extraction and tracking to maintain player identities across frames.
#     """
    
#     def __init__(self, model_path: str, confidence_threshold: float = 0.5):
#         """
#         Initialize the re-identification system.
        
#         Args:
#             model_path: Path to the YOLOv11 model
#             confidence_threshold: Minimum confidence for detections
#         """
#         # Load YOLO model
#         self.model = YOLO(model_path)
#         self.confidence_threshold = confidence_threshold
        
#         # Player tracking data
#         self.player_features = {}  # player_id -> list of feature vectors
#         self.player_positions = {}  # player_id -> list of (x, y) positions
#         self.player_last_seen = {}  # player_id -> frame number
#         self.next_player_id = 1
        
#         # Tracking parameters
#         self.max_distance_threshold = 100  # Max pixel distance for tracking
#         self.feature_similarity_threshold = 0.7  # Cosine similarity threshold
#         self.max_frames_missing = 30  # Max frames before considering player lost
#         self.feature_history_size = 10  # Number of feature vectors to keep per player
        
#         # Color extraction for simple appearance features
#         self.color_extractor = ColorFeatureExtractor()
        
#     def extract_appearance_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
#         """
#         Extract appearance features from a player bounding box.
        
#         Args:
#             image: Full frame image
#             bbox: Bounding box (x1, y1, x2, y2)
            
#         Returns:
#             Feature vector representing player appearance
#         """
#         x1, y1, x2, y2 = bbox
        
#         # Extract player region
#         player_region = image[y1:y2, x1:x2]
        
#         if player_region.size == 0:
#             return np.zeros(64)  # Return zero vector if empty region
        
#         # Resize to standard size
#         player_region = cv2.resize(player_region, (64, 128))
        
#         # Extract color histogram features
#         color_features = self.color_extractor.extract_features(player_region)
        
#         # Extract HOG features (Histogram of Oriented Gradients)
#         hog_features = self.extract_hog_features(player_region)
        
#         # Combine features
#         combined_features = np.concatenate([color_features, hog_features])
        
#         # Normalize features
#         combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-8)
        
#         return combined_features
    
#     def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
#         """Extract HOG features from an image region."""
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Simple gradient-based features
#         grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
#         # Calculate magnitude and direction
#         magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
#         # Create simple histogram of gradients
#         hist, _ = np.histogram(magnitude.flatten(), bins=16, range=(0, 255))
        
#         return hist.astype(np.float32)
    
#     def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
#         """Calculate Euclidean distance between two positions."""
#         return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
#     def calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
#         """Calculate cosine similarity between two feature vectors."""
#         features1 = features1.reshape(1, -1)
#         features2 = features2.reshape(1, -1)
#         return cosine_similarity(features1, features2)[0][0]
    
#     def find_best_match(self, detection_features: np.ndarray, detection_center: Tuple[float, float], 
#                        frame_number: int) -> Optional[int]:
#         """
#         Find the best matching existing player for a new detection.
        
#         Args:
#             detection_features: Feature vector of the detection
#             detection_center: Center position of the detection
#             frame_number: Current frame number
            
#         Returns:
#             Player ID of best match, or None if no good match found
#         """
#         best_match_id = None
#         best_score = -1
        
#         for player_id, player_feature_list in self.player_features.items():
#             # Skip if player hasn't been seen recently
#             if frame_number - self.player_last_seen[player_id] > self.max_frames_missing:
#                 continue
            
#             # Get last known position
#             if player_id in self.player_positions and self.player_positions[player_id]:
#                 last_position = self.player_positions[player_id][-1]
#                 distance = self.calculate_distance(detection_center, last_position)
                
#                 # Skip if too far away
#                 if distance > self.max_distance_threshold:
#                     continue
            
#             # Calculate feature similarity with recent features
#             recent_features = player_feature_list[-3:]  # Use last 3 feature vectors
#             similarities = []
            
#             for features in recent_features:
#                 similarity = self.calculate_feature_similarity(detection_features, features)
#                 similarities.append(similarity)
            
#             # Use maximum similarity
#             max_similarity = max(similarities) if similarities else 0
            
#             # Combine distance and appearance similarity
#             if player_id in self.player_positions and self.player_positions[player_id]:
#                 distance_score = 1.0 / (1.0 + distance / 100.0)  # Normalize distance
#                 combined_score = 0.7 * max_similarity + 0.3 * distance_score
#             else:
#                 combined_score = max_similarity
            
#             if combined_score > best_score and max_similarity > self.feature_similarity_threshold:
#                 best_score = combined_score
#                 best_match_id = player_id
        
#         return best_match_id
    
#     def update_player_data(self, player_id: int, features: np.ndarray, 
#                           center: Tuple[float, float], frame_number: int):
#         """Update player tracking data."""
#         # Update features (keep only recent ones)
#         if player_id not in self.player_features:
#             self.player_features[player_id] = []
        
#         self.player_features[player_id].append(features)
#         if len(self.player_features[player_id]) > self.feature_history_size:
#             self.player_features[player_id].pop(0)
        
#         # Update positions
#         if player_id not in self.player_positions:
#             self.player_positions[player_id] = []
        
#         self.player_positions[player_id].append(center)
#         if len(self.player_positions[player_id]) > self.feature_history_size:
#             self.player_positions[player_id].pop(0)
        
#         # Update last seen frame
#         self.player_last_seen[player_id] = frame_number
    
#     def process_frame(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
#         """
#         Process a single frame and return player detections with IDs.
        
#         Args:
#             frame: Input frame
#             frame_number: Frame number
            
#         Returns:
#             List of detection dictionaries with player IDs
#         """
#         # Run YOLO detection
#         results = self.model(frame, conf=self.confidence_threshold)
        
#         frame_detections = []
        
#         for result in results:
#             boxes = result.boxes
#             if boxes is None:
#                 continue
            
#             for box in boxes:
#                 # Get bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 confidence = box.conf[0].cpu().numpy()
#                 class_id = int(box.cls[0].cpu().numpy())
                
#                 # Only process player detections (assuming class 0 is player)
#                 if class_id != 0:  # Skip if not a player
#                     continue
                
#                 # Convert to integers
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 # Calculate center
#                 center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
#                 # Extract features
#                 features = self.extract_appearance_features(frame, (x1, y1, x2, y2))
                
#                 # Find best matching player
#                 player_id = self.find_best_match(features, center, frame_number)
                
#                 if player_id is None:
#                     # Create new player
#                     player_id = self.next_player_id
#                     self.next_player_id += 1
                
#                 # Update player data
#                 self.update_player_data(player_id, features, center, frame_number)
                
#                 # Add to frame detections
#                 detection = {
#                     'player_id': player_id,
#                     'bbox': (x1, y1, x2, y2),
#                     'center': center,
#                     'confidence': confidence,
#                     'frame_number': frame_number
#                 }
#                 frame_detections.append(detection)
        
#         return frame_detections
    
#     def process_video(self, video_path: str, output_path: str = None) -> List[List[Dict]]:
#         """
#         Process entire video and return all detections.
        
#         Args:
#             video_path: Path to input video
#             output_path: Optional path to save output video with annotations
            
#         Returns:
#             List of frame detections
#         """
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video: {video_path}")
        
#         # Get video properties
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Setup video writer if output path provided
#         out = None
#         if output_path:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         all_detections = []
#         frame_number = 0
        
#         # Colors for different players
#         colors = [
#             (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
#             (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
#         ]
        
#         print("Processing video...")
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Process frame
#             detections = self.process_frame(frame, frame_number)
#             all_detections.append(detections)
            
#             # Draw annotations if output video requested
#             if out is not None:
#                 annotated_frame = frame.copy()
                
#                 for detection in detections:
#                     player_id = detection['player_id']
#                     x1, y1, x2, y2 = detection['bbox']
#                     confidence = detection['confidence']
                    
#                     # Choose color based on player ID
#                     color = colors[player_id % len(colors)]
                    
#                     # Draw bounding box
#                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
#                     # Draw player ID and confidence
#                     label = f"Player {player_id}: {confidence:.2f}"
#                     label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                     cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
#                                 (x1 + label_size[0], y1), color, -1)
#                     cv2.putText(annotated_frame, label, (x1, y1 - 5), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
#                 out.write(annotated_frame)
            
#             frame_number += 1
            
#             if frame_number % 30 == 0:
#                 print(f"Processed {frame_number} frames...")
        
#         cap.release()
#         if out:
#             out.release()
        
#         print(f"Video processing complete. Total frames: {frame_number}")
#         print(f"Total unique players detected: {self.next_player_id - 1}")
        
#         return all_detections
    
#     def save_tracking_data(self, filepath: str):
#         """Save tracking data to file."""
#         tracking_data = {
#             'player_features': self.player_features,
#             'player_positions': self.player_positions,
#             'player_last_seen': self.player_last_seen,
#             'next_player_id': self.next_player_id
#         }
        
#         with open(filepath, 'wb') as f:
#             pickle.dump(tracking_data, f)
    
#     def load_tracking_data(self, filepath: str):
#         """Load tracking data from file."""
#         with open(filepath, 'rb') as f:
#             tracking_data = pickle.load(f)
        
#         self.player_features = tracking_data['player_features']
#         self.player_positions = tracking_data['player_positions']
#         self.player_last_seen = tracking_data['player_last_seen'] 
#         self.next_player_id = tracking_data['next_player_id']


# class ColorFeatureExtractor:
#     """Extract color-based features from image regions."""
    
#     def __init__(self):
#         self.hist_bins = 16
    
#     def extract_features(self, image: np.ndarray) -> np.ndarray:
#         """Extract color histogram features."""
#         # Convert to HSV for better color representation
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
#         # Extract histograms for each channel
#         h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
#         s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
#         v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])
        
#         # Normalize histograms
#         h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
#         s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
#         v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
        
#         # Combine histograms
#         color_features = np.concatenate([h_hist, s_hist, v_hist])
        
#         return color_features


# def main():
#     """Example usage of the Player Re-Identification System."""
    
#     # Paths (update these with your actual paths)
#     model_path = "best.pt"  # Your downloaded YOLOv11 model
#     video_path = "15sec_input_720p.mp4"       # Update with your video path
#     output_video_path = "output_with_tracking.mp4"
    
#     # Initialize the system
#     print("Initializing Player Re-Identification System...")
#     reid_system = PlayerReIdentificationSystem(
#         model_path=model_path,
#         confidence_threshold=0.5
#     )
    
#     # Process the video
#     try:
#         all_detections = reid_system.process_video(
#             video_path=video_path,
#             output_path=output_video_path
#         )
        
#         # Save tracking data
#         reid_system.save_tracking_data("tracking_data.pkl")
        
#         # Print summary statistics
#         print("\n=== TRACKING SUMMARY ===")
#         print(f"Total frames processed: {len(all_detections)}")
#         print(f"Total unique players: {reid_system.next_player_id - 1}")
        
#         # Print per-frame detection counts
#         frame_counts = [len(detections) for detections in all_detections]
#         print(f"Average detections per frame: {np.mean(frame_counts):.2f}")
#         print(f"Max detections in a frame: {max(frame_counts)}")
#         print(f"Min detections in a frame: {min(frame_counts)}")
        
#         # Print player appearance statistics
#         print("\n=== PLAYER STATISTICS ===")
#         for player_id in sorted(reid_system.player_features.keys()):
#             appearances = len(reid_system.player_features[player_id])
#             last_seen = reid_system.player_last_seen[player_id]
#             print(f"Player {player_id}: {appearances} appearances, last seen frame {last_seen}")
        
#     except Exception as e:
#         print(f"Error processing video: {e}")


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from collections import defaultdict, deque
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os
from typing import Dict, List, Tuple, Optional
import pickle

class ImprovedPlayerReIdentificationSystem:
    """
    Enhanced player re-identification system with improved detection and tracking.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.1):
        """
        Initialize with much lower confidence threshold to catch all players.
        """
        # Load YOLO model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold  # Much lower threshold
        
        # Player tracking data
        self.player_features = {}
        self.player_positions = {}
        self.player_last_seen = {}
        self.player_bbox_history = {}  # Store bbox history for better tracking
        self.next_player_id = 1
        
        # More aggressive tracking parameters
        self.max_distance_threshold = 200  # Increased from 100
        self.feature_similarity_threshold = 0.5  # Lowered from 0.7
        self.max_frames_missing = 60  # Increased from 30
        self.feature_history_size = 15  # Increased from 10
        
        # Multi-scale detection
        self.detection_scales = [640, 800, 1024]  # Multiple scales for better detection
        
        # Color extraction for simple appearance features
        self.color_extractor = ColorFeatureExtractor()
        
        print(f"Initialized with confidence threshold: {confidence_threshold}")
        print(f"Max distance threshold: {self.max_distance_threshold}")
        print(f"Feature similarity threshold: {self.feature_similarity_threshold}")
    
    def multi_scale_detection(self, frame: np.ndarray) -> List:
        """
        Run detection at multiple scales to catch more players.
        """
        all_detections = []
        
        for scale in self.detection_scales:
            # Resize frame
            h, w = frame.shape[:2]
            scale_factor = scale / max(h, w)
            
            if scale_factor != 1.0:
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                scaled_frame = cv2.resize(frame, (new_w, new_h))
            else:
                scaled_frame = frame
                scale_factor = 1.0
            
            # Run detection on scaled frame
            results = self.model(scaled_frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get bounding box coordinates and scale back
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = x1/scale_factor, y1/scale_factor, x2/scale_factor, y2/scale_factor
                    
                    all_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'scale': scale
                    })
        
        # Remove duplicate detections using NMS
        filtered_detections = self.non_max_suppression(all_detections)
        return filtered_detections
    
    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections using Non-Maximum Suppression.
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU
            remaining = []
            for det in detections:
                iou = self.calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2
        
        # Calculate intersection area
        xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
        xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_appearance_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Enhanced feature extraction."""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(96)  # Return zero vector if invalid region
        
        # Extract player region
        player_region = image[y1:y2, x1:x2]
        
        # Resize to standard size
        try:
            player_region = cv2.resize(player_region, (64, 128))
        except:
            return np.zeros(96)
        
        # Extract multiple types of features
        color_features = self.color_extractor.extract_features(player_region)
        hog_features = self.extract_hog_features(player_region)
        position_features = self.extract_position_features(bbox, image.shape)
        
        # Combine features
        combined_features = np.concatenate([color_features, hog_features, position_features])
        
        # Normalize features
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features
    
    def extract_position_features(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract position-based features."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Normalize positions
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width_ratio = (x2 - x1) / w
        height_ratio = (y2 - y1) / h
        
        return np.array([center_x, center_y, width_ratio, height_ratio])
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Enhanced HOG feature extraction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Create histogram
        hist_magnitude, _ = np.histogram(magnitude.flatten(), bins=16, range=(0, 255))
        hist_angle, _ = np.histogram(angle.flatten(), bins=16, range=(-np.pi, np.pi))
        
        # Normalize
        hist_magnitude = hist_magnitude.astype(np.float32)
        hist_angle = hist_angle.astype(np.float32)
        
        combined_hist = np.concatenate([hist_magnitude, hist_angle])
        return combined_hist / (np.sum(combined_hist) + 1e-8)
    
    def find_best_match(self, detection_features: np.ndarray, detection_center: Tuple[float, float], 
                       detection_bbox: Tuple[int, int, int, int], frame_number: int) -> Optional[int]:
        """Enhanced matching with bbox similarity."""
        best_match_id = None
        best_score = -1
        
        for player_id, player_feature_list in self.player_features.items():
            # Skip if player hasn't been seen recently
            frames_since_seen = frame_number - self.player_last_seen[player_id]
            if frames_since_seen > self.max_frames_missing:
                continue
            
            # Position-based scoring
            position_score = 0
            if player_id in self.player_positions and self.player_positions[player_id]:
                last_position = self.player_positions[player_id][-1]
                distance = self.calculate_distance(detection_center, last_position)
                
                # Adaptive distance threshold based on time gap
                adaptive_threshold = self.max_distance_threshold * (1 + frames_since_seen / 30.0)
                
                if distance > adaptive_threshold:
                    continue
                
                position_score = 1.0 / (1.0 + distance / 100.0)
            
            # Feature similarity scoring
            feature_scores = []
            recent_features = player_feature_list[-5:]  # Use last 5 features
            
            for features in recent_features:
                similarity = self.calculate_feature_similarity(detection_features, features)
                feature_scores.append(similarity)
            
            max_feature_similarity = max(feature_scores) if feature_scores else 0
            avg_feature_similarity = np.mean(feature_scores) if feature_scores else 0
            
            # Bbox similarity scoring
            bbox_score = 0
            if player_id in self.player_bbox_history and self.player_bbox_history[player_id]:
                last_bbox = self.player_bbox_history[player_id][-1]
                bbox_score = self.calculate_bbox_similarity(detection_bbox, last_bbox)
            
            # Combined scoring with weights
            combined_score = (
                0.4 * max_feature_similarity +
                0.3 * position_score +
                0.2 * avg_feature_similarity +
                0.1 * bbox_score
            )
            
            # Lower threshold for matching
            if combined_score > best_score and max_feature_similarity > self.feature_similarity_threshold:
                best_score = combined_score
                best_match_id = player_id
        
        return best_match_id
    
    def calculate_bbox_similarity(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate similarity between two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2
        
        # Size similarity
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Aspect ratio similarity
        aspect1 = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1
        aspect2 = (x2_p - x1_p) / (y2_p - y1_p) if (y2_p - y1_p) > 0 else 1
        aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2) if max(aspect1, aspect2) > 0 else 0
        
        return 0.7 * size_ratio + 0.3 * aspect_ratio
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        return cosine_similarity(features1, features2)[0][0]
    
    def update_player_data(self, player_id: int, features: np.ndarray, 
                          center: Tuple[float, float], bbox: Tuple[int, int, int, int], frame_number: int):
        """Update player tracking data with bbox history."""
        # Update features
        if player_id not in self.player_features:
            self.player_features[player_id] = []
        
        self.player_features[player_id].append(features)
        if len(self.player_features[player_id]) > self.feature_history_size:
            self.player_features[player_id].pop(0)
        
        # Update positions
        if player_id not in self.player_positions:
            self.player_positions[player_id] = []
        
        self.player_positions[player_id].append(center)
        if len(self.player_positions[player_id]) > self.feature_history_size:
            self.player_positions[player_id].pop(0)
        
        # Update bbox history
        if player_id not in self.player_bbox_history:
            self.player_bbox_history[player_id] = []
        
        self.player_bbox_history[player_id].append(bbox)
        if len(self.player_bbox_history[player_id]) > self.feature_history_size:
            self.player_bbox_history[player_id].pop(0)
        
        # Update last seen frame
        self.player_last_seen[player_id] = frame_number
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """Process frame with multi-scale detection."""
        # Use multi-scale detection for better coverage
        detections = self.multi_scale_detection(frame)
        
        frame_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Only process player detections (skip ball)
            if class_id != 0:  # Assuming class 0 is player
                continue
            
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Extract features
            features = self.extract_appearance_features(frame, bbox)
            
            # Find best matching player
            player_id = self.find_best_match(features, center, bbox, frame_number)
            
            if player_id is None:
                # Create new player
                player_id = self.next_player_id
                self.next_player_id += 1
                print(f"New player {player_id} detected at frame {frame_number}")
            
            # Update player data
            self.update_player_data(player_id, features, center, bbox, frame_number)
            
            # Add to frame detections
            detection_info = {
                'player_id': player_id,
                'bbox': bbox,
                'center': center,
                'confidence': confidence,
                'frame_number': frame_number
            }
            frame_detections.append(detection_info)
        
        return frame_detections
    
    def process_video(self, video_path: str, output_path: str = None) -> List[List[Dict]]:
        """Process video with enhanced tracking."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_number = 0
        
        # Enhanced color palette
        colors = [
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta  
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Lime Green
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (255, 20, 147),   # Deep Pink
            (0, 255, 127),    # Spring Green
            (138, 43, 226),   # Blue Violet
            (255, 69, 0),     # Red Orange
            (50, 205, 50),    # Lime Green
            (255, 215, 0),    # Gold
            (30, 144, 255),   # Dodger Blue
            (220, 20, 60),    # Crimson
        ]
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = self.process_frame(frame, frame_number)
            all_detections.append(detections)
            
            # Enhanced visualization
            if out is not None:
                annotated_frame = frame.copy()
                
                for detection in detections:
                    player_id = detection['player_id']
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    
                    color = colors[player_id % len(colors)]
                    
                    # Draw thick bounding box with white border
                    cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 3)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
                    
                    # Large, visible label
                    label = f"P{player_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 3
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Position label
                    label_y = max(y1 - 10, label_h + 10)
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, 
                                (x1, label_y - label_h - 8), 
                                (x1 + label_w + 16, label_y + 8), 
                                color, -1)
                    cv2.rectangle(annotated_frame, 
                                (x1, label_y - label_h - 8), 
                                (x1 + label_w + 16, label_y + 8), 
                                (255, 255, 255), 3)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1 + 8, label_y), 
                              font, font_scale, (255, 255, 255), thickness)
                    
                    # Draw center point
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
                    cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 2)
                
                # Frame info
                info = f"Frame: {frame_number} | Players: {len(detections)} | Conf: {self.confidence_threshold}"
                cv2.rectangle(annotated_frame, (10, 10), (600, 50), (0, 0, 0), -1)
                cv2.putText(annotated_frame, info, (15, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                out.write(annotated_frame)
            
            frame_number += 1
            
            if frame_number % 30 == 0:
                detected_this_batch = sum(len(dets) for dets in all_detections[-30:])
                print(f"Processed {frame_number}/{total_frames} frames... "
                      f"Avg detections last 30 frames: {detected_this_batch/30:.2f}")
        
        cap.release()
        if out:
            out.release()
        
        # Enhanced statistics
        total_detections = sum(len(dets) for dets in all_detections)
        print(f"\n=== ENHANCED TRACKING SUMMARY ===")
        print(f"Total frames processed: {frame_number}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_number:.2f}")
        print(f"Total unique players: {self.next_player_id - 1}")
        
        # Per-player detailed stats
        print(f"\n=== DETAILED PLAYER STATISTICS ===")
        for player_id in sorted(self.player_features.keys()):
            appearances = len(self.player_features[player_id])
            first_seen = min([frame for frame_dets in all_detections 
                            for det in frame_dets 
                            if det['player_id'] == player_id], default=0)
            last_seen = self.player_last_seen[player_id]
            duration = last_seen - first_seen + 1
            
            print(f"Player {player_id}: {appearances} detections, "
                  f"frames {first_seen}-{last_seen} (duration: {duration})")
        
        return all_detections


class ColorFeatureExtractor:
    """Enhanced color feature extraction."""
    
    def __init__(self):
        self.hist_bins = 12  # Reduced bins for robustness
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract robust color features."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract histograms
        h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])
        
        # Normalize
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
        
        return np.concatenate([h_hist, s_hist, v_hist])


def main():
    """Run the improved tracking system."""
    model_path = "best.pt"  # Your YOLOv11 model
    video_path = "15sec_input_720p.mp4"
    output_video_path = "enhanced_tracking_output.mp4"
    
    # Initialize with very low confidence threshold
    reid_system = ImprovedPlayerReIdentificationSystem(
        model_path=model_path,
        confidence_threshold=0.1  # Very low to catch all players
    )
    
    try:
        all_detections = reid_system.process_video(
            video_path=video_path,
            output_path=output_video_path
        )
        
        print(f"\nOutput video saved as: {output_video_path}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()