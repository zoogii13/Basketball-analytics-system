import os
import sys
import pathlib
import numpy as np

# Assuming utils is in the parent directory
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance

class SpeedAndDistanceCalculator():
    def __init__(self, 
                 width_in_pixels,
                 height_in_pixels,
                 width_in_meters,
                 height_in_meters):
        
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters
        
        # Human speed limit (km/h)
        self.MAX_HUMAN_SPEED = 40.0 
        # Minimum distance in meters to consider it "movement" (ignores jitter)
        self.MIN_MOVEMENT_THRESHOLD = 0.05 

    def calculate_distance(self, tactical_player_positions):
        previous_players_position = {}
        output_distances = []

        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            output_distances.append({})

            for player_id, current_player_position in tactical_player_position_frame.items():
                if player_id in previous_players_position:
                    previous_position = previous_players_position[player_id]
                    meter_distance = self.calculate_meter_distance(previous_position, current_player_position)
                    
                    # Apply noise threshold
                    if meter_distance < self.MIN_MOVEMENT_THRESHOLD:
                        meter_distance = 0
                        
                    output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id] = current_player_position
        
        return output_distances

    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
         prev_x, prev_y = previous_pixel_position
         curr_x, curr_y = current_pixel_position

         # Convert pixels to meters
         prev_m_x = prev_x * self.width_in_meters / self.width_in_pixels
         prev_m_y = prev_y * self.height_in_meters / self.height_in_pixels
         curr_m_x = curr_x * self.width_in_meters / self.width_in_pixels
         curr_m_y = curr_y * self.height_in_meters / self.height_in_pixels

         return measure_distance((curr_m_x, curr_m_y), (prev_m_x, prev_m_y))

    def calculate_speed(self, distances, fps=30):
        speeds = []
        window_size = 15  # Use 0.5 seconds (at 30fps) for better smoothing
        
        for frame_idx in range(len(distances)):
            speeds.append({})
            
            for player_id in distances[frame_idx].keys():
                # Look back 'window_size' frames
                start_frame = max(0, frame_idx - window_size)
                
                total_distance_m = 0
                frames_counted = 0
                
                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        total_distance_m += distances[i][player_id]
                        frames_counted += 1
                
                # We need a minimum amount of data to calculate an accurate speed
                if frames_counted > 1:
                    time_seconds = frames_counted / fps
                    # Speed = (meters / 1000) / (seconds / 3600) -> km/h
                    speed_kmh = (total_distance_m / 1000) / (time_seconds / 3600)
                    
                    # 1. Apply Logic Cap: Humans don't run 90kmh
                    if speed_kmh > self.MAX_HUMAN_SPEED:
                        speed_kmh = self.MAX_HUMAN_SPEED
                        
                    speeds[frame_idx][player_id] = speed_kmh
                else:
                    speeds[frame_idx][player_id] = 0
                    
        # Optional: Second pass for Exponential Moving Average (EMA) smoothing
        return self._smooth_speeds(speeds)

    def _smooth_speeds(self, speeds):
        """Applies a simple smoothing filter to the calculated speed values."""
        if not speeds: return []
        
        smoothed_speeds = [{} for _ in range(len(speeds))]
        alpha = 0.3  # Smoothing factor (0 to 1). Lower = smoother but laggier.
        
        last_speed = {} # Store last speed per player

        for frame_idx, frame_data in enumerate(speeds):
            for player_id, current_speed in frame_data.items():
                if player_id in last_speed:
                    # Exponential Moving Average formula
                    smoothed = (alpha * current_speed) + ((1 - alpha) * last_speed[player_id])
                    smoothed_speeds[frame_idx][player_id] = smoothed
                    last_speed[player_id] = smoothed
                else:
                    smoothed_speeds[frame_idx][player_id] = current_speed
                    last_speed[player_id] = current_speed
                    
        return smoothed_speeds