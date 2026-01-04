import numpy as np
from collections import deque
import cv2
from copy import deepcopy

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Calculates the Homography matrix using RANSAC to ignore outlier keypoints.
        """
        # RANSAC is essential for sports to ignore misidentified court lines.
        self.m, self.mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0: return points
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        # Internal dimensions for the tactical image
        self.width, self.height = 300, 161
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15 

        # Tactical mapping for your 18 detected keypoints
        self.key_points = [
            (0, 0), (0, 10), (0, 56), (0, 108), (0, 152), (0, 161),  # Left edge
            (150, 161), (150, 0),                                   # Middle line
            (62, 56), (62, 108),                                    # Left Free throw
            (300, 161), (300, 152), (300, 108), (300, 56),          # Right edge
            (300, 10), (300, 0),                                    # Right edge cont.
            (238, 56), (238, 108)                                   # Right Free throw
        ]
        
        # Buffer for temporal smoothing to prevent player vibration
        self.position_buffer = {} 
        self.buffer_size = 5 

    def smooth_positions(self, player_id, new_position):
        """
        Applies a moving average to smooth player movement on the tactical map.
        """
        if player_id not in self.position_buffer:
            self.position_buffer[player_id] = deque(maxlen=self.buffer_size)
        
        self.position_buffer[player_id].append(new_position)
        avg_pos = np.mean(self.position_buffer[player_id], axis=0)
        return avg_pos.tolist()

    def validate_keypoints(self, keypoints_list):
        """
        Filters out distorted keypoints using proportional geometric checks.
        """
        keypoints_list = deepcopy(keypoints_list)
        max_idx = len(self.key_points)

        for frame_idx, frame_results in enumerate(keypoints_list):
            frame_keypoints = frame_results.xy.cpu().numpy()[0]
            
            # Index Guard: Only process indices defined in our tactical map
            detected_indices = [
                i for i, kp in enumerate(frame_keypoints) 
                if kp[0] > 0 and kp[1] > 0 and i < max_idx
            ]
            
            if len(detected_indices) < 4:
                continue
            
            invalid_keypoints = []
            for i in detected_indices:
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2: continue

                j, k = other_indices[0], other_indices[1]
                if j >= max_idx or k >= max_idx: continue

                # Euclidean distances for proportion validation
                d_ij = np.linalg.norm(frame_keypoints[i] - frame_keypoints[j])
                d_ik = np.linalg.norm(frame_keypoints[i] - frame_keypoints[k])
                
                t_ij = np.linalg.norm(np.array(self.key_points[i]) - np.array(self.key_points[j]))
                t_ik = np.linalg.norm(np.array(self.key_points[i]) - np.array(self.key_points[k]))

                if t_ij > 0 and t_ik > 0:
                    error = abs(((d_ij / d_ik) - (t_ij / t_ik)) / (t_ij / t_ik))
                    if error > 0.8: # 80% error threshold                       
                        keypoints_list[frame_idx].xy[0][i] *= 0 
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Projects players to the 2D court and applies temporal smoothing.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}
            raw_kp = frame_keypoints.xy.cpu().numpy()[0]
            max_idx = len(self.key_points)
            
            valid_indices = [i for i, kp in enumerate(raw_kp) if kp[0] > 0 and kp[1] > 0 and i < max_idx]
            
            if len(valid_indices) < 6: # Standard for stable sports projection
                tactical_player_positions.append({})
                continue
            
            source_points = np.array([raw_kp[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                homography = Homography(source_points, target_points)
                
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Map from feet position (bottom-center)
                    foot_pos = np.array([[ (bbox[0]+bbox[2])/2, bbox[3] ]], dtype=np.float32)
                    
                    transformed = homography.transform_points(foot_pos)[0]

                    # Boundary check with 15-pixel clip margin
                    if (-15 <= transformed[0] <= self.width + 15 and 
                        -15 <= transformed[1] <= self.height + 15):
                        
                        raw_x = np.clip(transformed[0], 0, self.width)
                        raw_y = np.clip(transformed[1], 0, self.height)
                        
                        smoothed_pos = self.smooth_positions(player_id, [raw_x, raw_y])
                        tactical_positions[player_id] = smoothed_pos
                    
            except: pass
            
            tactical_player_positions.append(tactical_positions)
            
            # Maintenance: Remove IDs no longer active
            active_ids = set(frame_tracks.keys())
            for cached_id in list(self.position_buffer.keys()):
                if cached_id not in active_ids:
                    del self.position_buffer[cached_id]
        
        return tactical_player_positions