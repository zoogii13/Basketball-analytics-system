import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import tempfile
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq


st.set_page_config(
    page_title="AI Basketball Analytics Dashboard",
    layout="wide",
    page_icon="🏀"
)


# SIDEBAR 
st.sidebar.title("Upload Match Video")

# 1. Upload Raw Video
raw_video = st.sidebar.file_uploader("Upload Raw Footage", type=["mp4", "mov", "avi"])

# Define paths for generated files
GEN_JSON = "basketball_analysis.json"
GEN_VIDEO = "output_videos/final_output.mp4"

if raw_video:
    # Save raw video to a temporary location for main.py to read
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(raw_video.read())
    raw_input_path = tfile.name

    if st.sidebar.button("Run Analysis"):
        with st.sidebar.status("Processing Video...", expanded=True) as status:
            st.write("Detecting players and ball...")
            # Run your main.py via subprocess
            try:
                subprocess.run([
                    "python", "main.py", 
                    raw_input_path, 
                    "--output_video", GEN_VIDEO
                ], check=True)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.rerun() # Refresh to load the newly created files
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# Check if analysis exists
if not os.path.exists(GEN_JSON):
    st.info("👋 Welcome! Please upload a video in the sidebar and click 'Run Analysis' to generate data.")
    st.stop()

# Load the generated data
with open(GEN_JSON, 'r') as f:
    data = json.load(f)


# TOP NAVIGATION BAR
tabs = st.tabs([
    "Overview",
    "Performance Analysis",
    "Spatial & Network Analysis",
    "CoachBot"
])


# HELPER FUNCTIONS

def summarize_game_for_llm(data, fps=25):
    # Using your original variable names
    team_stats = defaultdict(lambda: {
        "passes": 0,
        "interceptions": 0,
        "avg_speed": [],
        "sprint_frames": 0
    })

    player_stats = defaultdict(lambda: {
        "team": None,
        "avg_speed": [],
        "sprint_frames": 0
    })

    for frame in data:
        event = frame.get("event", "")

        # Event logic
        if event.startswith("pass_by_team"):
            team = int(event.split("_")[-1])
            team_stats[team]["passes"] += 1

        if event.startswith("interception_by_team"):
            team = int(event.split("_")[-1])
            team_stats[team]["interceptions"] += 1

        # Kinematics logic
        for p in frame.get("players", []):
            p_id = p["id"]
            team_id = p["team"]
            speed = p.get("speed_kmh", 0)

            # Player stats
            player_stats[p_id]["team"] = team_id
            player_stats[p_id]["avg_speed"].append(speed)
            
            # Team stats
            team_stats[team_id]["avg_speed"].append(speed)

            # Logic for "High Intensity" (Sprints > 20km/h)
            if speed > 20:
                player_stats[p_id]["sprint_frames"] += 1
                team_stats[team_id]["sprint_frames"] += 1

    return {
        "teams": {
            t: {
                "passes": v["passes"],
                "interceptions": v["interceptions"],
                "avg_speed": round(np.mean(v["avg_speed"]), 2) if v["avg_speed"] else 0,
                # Total distance in km: (Sum of speeds / 3600) / fps
                "distance_km": round((sum(v["avg_speed"]) / 3600) / fps, 2) if v["avg_speed"] else 0
            } for t, v in team_stats.items()
        },
        "players": {
            p: {
                "team": v["team"],
                "avg_speed": round(np.mean(v["avg_speed"]), 2) if v["avg_speed"] else 0,
                "top_speed": round(max(v["avg_speed"]), 2) if v["avg_speed"] else 0,
                # How many seconds they spent sprinting
                "sprint_time_sec": round(v["sprint_frames"] / fps, 1)
            } for p, v in player_stats.items()
        }
    }

with tabs[0]:
    st.title("Match Overview")
    
    # top row metrics for immediate impact
    m_col3, m_col4 = st.columns(2)
    
    # Pre-calculate data for metrics
    total_frames = len(data)
    duration_sec = total_frames / 30  # Assuming 30fps
    
    # Extract team IDs and ball control
    team_ids = sorted(list(set(p["team"] for f in data for p in f["players"])))
    possession_counts = defaultdict(int)
    for f in data:
        if f.get("ball_possession_id") is not None:
            # Finding which team the possessor belongs to
            for p in f["players"]:
                if p["id"] == f["ball_possession_id"]:
                    possession_counts[p["team"]] += 1

    
    with m_col3:
        if len(team_ids) >= 2:
            t1_pos = (possession_counts[team_ids[0]] / total_frames) * 100
            st.metric(f"Team {team_ids[0]} Possession", f"{t1_pos:.1f}%")
    with m_col4:
        # Simple intensity score based on avg speed across all players
        all_speeds = [p["speed_kmh"] for f in data for p in f["players"]]
        avg_int = np.mean(all_speeds) if all_speeds else 0
        st.metric("Game Intensity", f"{avg_int:.1f} km/h")

    st.divider()

    # Main Layout: Video and Event Timeline
    col_vid, col_events = st.columns([2, 1])

    with col_vid:
        st.subheader("Final Tactical Analysis")
        if os.path.exists(GEN_VIDEO):
            st.video(GEN_VIDEO)
        else:
            st.info("Processed video will appear here after analysis.")


# TAB 2: PERFORMANCE ANALYSIS
with tabs[1]:
    st.title("Performance Analysis")

    # ---- 1. TEAM COMPARISON DATA ----
    st.subheader("Team-Wide Comparison")
    
    # Aggregate data efficiently
    team_data = defaultdict(lambda: {"passes": 0, "interceptions": 0, "distances": []})
    player_data = defaultdict(lambda: {"team": None, "speeds": [], "distances": []})
    
    for f in data:
        # Event Logic
        if f["event"].startswith("pass_by_team"):
            team_data[int(f["event"].split("_")[-1])]["passes"] += 1
        if f["event"].startswith("interception_by_team"):
            team_data[int(f["event"].split("_")[-1])]["interceptions"] += 1
        
        # Player Kinematics
        for p in f["players"]:
            p_id, t_id, spd = p["id"], p["team"], p["speed_kmh"]
            player_data[p_id]["team"] = t_id
            player_data[p_id]["speeds"].append(spd)
            # Distance per frame = speed / (3600 * fps)
            dist = spd / (3600 * 30) 
            player_data[p_id]["distances"].append(dist)
            team_data[t_id]["distances"].append(dist)

    # Display Team Metrics in Columns
    cols = st.columns(len(team_data))
    for i, (t_id, stats) in enumerate(team_data.items()):
        with cols[i]:
            st.markdown(f"### Team {t_id}")
            st.write(f"**Total Distance:** {sum(stats['distances']):.2f} km")
            st.write(f"**Pass Efficiency:** {stats['passes']} successful")
            st.write(f"**Interceptions:** {stats['interceptions']}")

    st.divider()

    # ---- 2. INDIVIDUAL PLAYER DEEP DIVE ----
    col_sel, col_stats = st.columns([1, 2])
    
    with col_sel:
        st.subheader("Player Profile")
        selected_player = st.selectbox("Select Player to Analyze", sorted(player_data.keys()))
        
        p_stats = player_data[selected_player]
        avg_spd = np.mean(p_stats["speeds"])
        max_spd = np.max(p_stats["speeds"])
        total_dist = sum(p_stats["distances"]) * 1000 # Convert to meters for readability
        
        st.metric("Total Workload", f"{total_dist:.0f} meters")
        st.metric("Avg Speed", f"{avg_spd:.2f} km/h")
        st.metric("Top Speed", f"{max_spd:.2f} km/h", 
                  delta=f"{max_spd - avg_spd:.1f} km/h burst")

    with col_stats:
        st.subheader("Speed & Intensity Profile")
        
        # Create a DataFrame for a multi-line chart
        speed_df = pd.DataFrame({
            "Player Speed": p_stats["speeds"],
            "Team Average": [np.mean(team_data[p_stats['team']]['distances']) * 3600 * 30] * len(p_stats["speeds"])
        })
        
        st.line_chart(speed_df)
        
        intensity_frames = len([s for s in p_stats["speeds"] if s > 15])
        intensity_pct = (intensity_frames / len(p_stats["speeds"])) * 100
        st.write(f"**High Intensity Zone (>15km/h):** {intensity_pct:.1f}% of match time")

    # ---- 3. RANKINGS ----
    st.divider()
    st.subheader("Squad Rankings")
    
    # Create a ranking table
    rank_list = []
    for p_id, v in player_data.items():
        rank_list.append({
            "Player": p_id,
            "Team": v["team"],
            "Avg Speed": np.mean(v["speeds"]),
            "Distance (m)": sum(v["distances"]) * 1000
        })
    
    rank_df = pd.DataFrame(rank_list).sort_values(by="Distance (m)", ascending=False)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)


# TAB 3: SPATIAL & NETWORK ANALYSIS
with tabs[2]:
    st.title("Spatial & Network Analysis")

    # ---- 1. HEATMAP SECTION (PLAYER & TEAM) ----
    st.subheader("Field Coverage & Occupancy")
    
    h_col1, h_col2 = st.columns([1, 3])
    with h_col1:
        mode = st.radio("Heatmap Mode", ["Individual Player", "Full Team"])
        
        # Aggregate all positions
        player_positions = defaultdict(list)
        team_positions = defaultdict(list)
        for f in data:
            for p in f["players"]:
                if len(p["coords"]) == 2:
                    player_positions[p["id"]].append(p["coords"])
                    team_positions[p["team"]].append(p["coords"])

        if mode == "Individual Player":
            target = st.selectbox("Select Player", sorted(player_positions.keys()))
            coords = np.array(player_positions[target])
            title = f"Movement Heatmap: Player {target}"
        else:
            target = st.selectbox("Select Team", sorted(team_positions.keys()))
            coords = np.array(team_positions[target])
            title = f"Occupancy Heatmap: Team {target}"

    with h_col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(coords[:, 0], coords[:, 1], gridsize=20, cmap='YlOrRd', mincnt=1)

        ax.set_xlim(0, 28); ax.set_ylim(0, 15)
        ax.axvline(14, color='white', linestyle='--', alpha=0.5) # Half court
        plt.colorbar(hb, label='Time Spent in Zone')
        ax.set_title(title)
        st.pyplot(fig)

    st.divider()

    # ---- 2. ADVANCED PASSING NETWORK ----
    st.subheader("Passing Connectivity")
    
    n_col1, n_col2 = st.columns([1, 2])
    
    with n_col1:
        st.info("""
        **Network Key:**
        - **Nodes:** Players
        - **Edges:** Passing connection
        - **Thickness:** Volume of passes
        """)
        
        # Build weighted graph
        G = nx.DiGraph()
        prev_possessor = None
        for f in data:
            curr = f.get("ball_possession_id")
            if f["event"].startswith("pass_by_team") and prev_possessor is not None:
                if curr != prev_possessor and curr is not None:
                    if G.has_edge(prev_possessor, curr):
                        G[prev_possessor][curr]['weight'] += 1
                    else:
                        G.add_edge(prev_possessor, curr, weight=1)
            prev_possessor = curr

    with n_col2:
        if G.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            pos = nx.circular_layout(G) # Circular shows team chemistry better
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#FF4B4B', ax=ax)
            nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
            
            # Draw edges with varying thickness
            weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                                   arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)
            
            plt.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No pass sequences detected to build network.")


# TAB 4: COACHBOT
with tabs[3]:
    st.title("AI CoachBot")

    summary = summarize_game_for_llm(data)
    question = st.text_area("Ask the coach:")

    if st.button("Ask Coach") and question:
        client = Groq(api_key=os.getenv("APIKEY"))

        prompt = f"""
        You are 'CoachAI', an elite Tactical Analyst for an NBA team. 
        You are analyzing match tracking data and event logs.
        MATCH SUMMARY:
        {json.dumps(summary, indent=2)}

        QUESTION:
        {question}
        RESPONSE GUIDELINES:
- Start with a 1-sentence "Game Pulse" (overall impression).
- Use professional terminology: 'Transition Defense', 'Pick and Roll spacing', 'Secondary Break', 'Ball Reversals'.
- Identify ONE 'Tactical Red Flag' (e.g., fatigue or lack of ball movement).
- Provide 2 actionable 'Drill Recommendations' to fix the issues found.

Be authoritative, concise, and data-driven.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        st.success(response.choices[0].message.content)
