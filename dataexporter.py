import json
import numpy as np

class LLMDataFeeder:
    def __init__(self, fps=30):
        self.fps = fps

    def format_for_llm(self, player_tracks, team_assignments, possession, speeds, passes, interceptions, tactical_player_positions):
        game_log = []
        num_frames = len(player_tracks)
        
        for frame_num in range(num_frames):
            # Capture periodically or on key events (pass/interception)
            is_sample_frame = (frame_num % 10 == 0)
            is_event_frame = (passes[frame_num] != -1 or interceptions[frame_num] != -1)

            if not (is_sample_frame or is_event_frame):
                continue

            timestamp = round(frame_num / self.fps, 2)
            
            event_type = "none"
            if passes[frame_num] != -1:
                event_type = f"pass_by_team_{int(passes[frame_num])}"
            elif interceptions[frame_num] != -1:
                event_type = f"interception_by_team_{int(interceptions[frame_num])}"

            # Ensure ball possession ID is a standard Python int
            pos_id = possession[frame_num]
            if isinstance(pos_id, (np.integer, np.int64)):
                pos_id = pos_id.item()

            frame_data = {
                "time_sec": float(round(frame_num / self.fps, 2)),
                "event": event_type,
                "ball_possession_id": int(pos_id),
                "players": []
            }

            for player_id, track in player_tracks[frame_num].items():
                # Cast types for JSON serialization
                p_id = int(player_id) if isinstance(player_id, (np.integer, np.int64)) else player_id
                team = team_assignments[frame_num].get(player_id, "Unknown")
                speed = speeds[frame_num].get(player_id, 0)
                
                # Fetch smoothed coordinates from the tactical converter
                player_coord = tactical_player_positions[frame_num].get(player_id, [0, 0])
                
                frame_data["players"].append({
                    "id": p_id,
                    "team": team,
                    "speed_kmh": float(round(speed, 2)),
                    "coords": [float(player_coord[0]), float(player_coord[1])] # Essential for mplbasketball
                })
            
            game_log.append(frame_data)
            
        return game_log

    def save_to_json(self, data, file_path="basketball_analysis.json"):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)