import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    def __init__(self):
        self.player_team_dict = {}
        
        # Hard-coded Team 1 baseline (White)
        # Using HSV: High Value (brightness) and Low Saturation (whiteness)
        self.team_1_hsv_threshold = {
            'low': np.array([0, 0, 160]),
            'high': np.array([180, 60, 255])
        }

    def get_player_color(self, frame, bbox):
        # 1. Surgical Crop: Focus only on the middle of the torso
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        height, width, _ = image.shape
        jersey_region = image[int(height*0.2):int(height*0.5), int(width*0.2):int(width*0.8)]

        if jersey_region.size == 0:
            return None

        # 2. Convert to HSV for better color separation
        hsv_jersey = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)

        # 3. MASKING: Filter out the wood court/skin tones (approx. Orange/Yellow/Brown)
        # This prevents the court floor from being picked as a "dominant color"
        lower_skin_court = np.array([0, 40, 40])
        upper_skin_court = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_jersey, lower_skin_court, upper_skin_court)
        
        # Invert mask to keep everything EXCEPT skin and court
        mask = cv2.bitwise_not(mask)
        filtered_pixels = hsv_jersey[mask > 0]

        if filtered_pixels.size == 0:
            # Fallback to the original region if masking was too aggressive
            filtered_pixels = hsv_jersey.reshape(-1, 3)

        # 4. K-Means: Find the single most dominant color in the filtered set
        # Using n_clusters=3 helps separate shadows and highlights into different clusters
        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=1)
        kmeans.fit(filtered_pixels)
        
        # Pick the cluster with the most pixels
        counts = np.bincount(kmeans.labels_)
        dominant_hsv = kmeans.cluster_centers_[np.argmax(counts)]
        
        return dominant_hsv

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_hsv = self.get_player_color(frame, player_bbox)
        
        if player_hsv is None:
            return 1

        # Check if the dominant color is "White"
        # We check: Is Saturation low (< 60) AND is Brightness high (> 160)?
        is_white = (player_hsv[1] < self.team_1_hsv_threshold['high'][1]) and \
                   (player_hsv[2] > self.team_1_hsv_threshold['low'][2])

        if is_white:
            team_id = 1 # Team 1 (White)
        else:
            team_id = 2 # Team 2 (Dark Blue)

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            return player_assignment

        self.player_team_dict = {}

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):        
            player_assignment.append({})
            
            # Reset cache every 200 frames to handle camera angle changes
            if frame_num % 200 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                player_assignment[frame_num][player_id] = team
        
        save_stub(stub_path, player_assignment)
        return player_assignment