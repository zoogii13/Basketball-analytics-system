import json
import os

def debug_basketball_data(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total frames in log: {len(data)}")
    
    # Check the first frame that has players
    sample_frame = next((f for f in data if f.get('players')), None)
    
    if not sample_frame:
        print("❌ Error: No frames contain player data.")
        return

    print(f"--- Checking Frame at {sample_frame['time_sec']}s ---")
    for player in sample_frame['players']:
        p_id = player.get('id')
        if 'coords' not in player:
            print(f"❌ Player {p_id}: Missing 'coords' key!")
            continue
            
        x_raw, y_raw = player['coords']
        # Rescale as per your app.py logic
        x_ft = x_raw * (94 / 300)
        y_ft = y_raw * (50 / 161)
        
        print(f"✅ Player {p_id}: Raw({x_raw:.1f}, {y_raw:.1f}) -> Court({x_ft:.1f}ft, {y_ft:.1f}ft)")
        
        if not (0 <= x_ft <= 94) or not (0 <= y_ft <= 50):
            print(f"   ⚠️ Warning: Coordinates are OUT OF BOUNDS for a standard court!")

if __name__ == "__main__":
    debug_basketball_data("basketball_analysis_llm.json")