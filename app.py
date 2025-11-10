import sys
# sys.stdout.reconfigure(encoding="utf-8") # Removed for Streamlit compatibility

import json, re
from time import sleep
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from playwright.sync_api import sync_playwright
import streamlit as st
import requests  # For calling Gemini API
import random
import time

# ---------- Gemini API Call Function ----------
def call_gemini_api(prompt, max_retries=5):
    """
    Calls the Gemini API to get the AI analysis.
    Includes exponential backoff for retries.
    """
    # Load the API key from Streamlit's secrets.
    # This will read from .streamlit/secrets.toml
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{
                "text": "You are a professional football analyst. Your tone is insightful, confident, and clear. Provide a concise, multi-paragraph analysis of the match."
            }]
        }
    }
    headers = {'Content-Type': 'application/json'}
    
    base_delay = 1  # 1 second
    for i in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return text
                else:
                    return f"Error: Received an unexpected response structure from API: {result}"
            else:
                # Handle non-200 errors, but still retry
                st.error(f"API Error: Status Code {response.status_code}. Retrying...")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: {e}. Retrying...")
            
        # Exponential backoff
        delay = base_delay * (2 ** i) + random.uniform(0, 1)
        time.sleep(delay)
        
    return "Error: Failed to get analysis from AI after several retries."


# ---------- AI Analysis Formatting ----------
def get_ai_analysis(event_data, avg_data, graph_data, stats_data):
    """
    Formats the data and generates a prompt for the AI analysis.
    """
    try:
        # --- Event Info ---
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        home_score = event_data["event"]["homeScore"].get("display", "N/A")
        away_score = event_data["event"]["awayScore"].get("display", "N/A")
        
        # --- Stats Summary ---
        overview_group = next((g for g in stats_data["statistics"][0]["groups"] if g["groupName"] == "Match overview"), None)
        stats_list = []
        if overview_group:
            for item in overview_group["statisticsItems"]:
                stats_list.append(f"- {item['name']}: {item['homeValue']} (Home) vs {item['awayValue']} (Away)")
        stats_summary = "\n".join(stats_list)

        # --- Momentum Summary ---
        graph_points = graph_data.get("graphPoints", [])
        home_pos = sum(p['value'] for p in graph_points if p['value'] > 0)
        away_neg = sum(p['value'] for p in graph_points if p['value'] < 0)
        away_pos = abs(away_neg)
        total_momentum = home_pos + away_pos
        home_perc = (home_pos / total_momentum * 100) if total_momentum > 0 else 50
        away_perc = (away_pos / total_momentum * 100) if total_momentum > 0 else 50
        momentum_summary = f"Home team controlled {home_perc:.0f}% of the attack momentum vs. Away team's {away_perc:.0f}%."

        # --- Player Positions Summary ---
        home_players = get_starters(avg_data, "home")
        away_players = get_starters(avg_data, "away")
        home_pos_list = [f"- {p['player']['shortName']} ({p['player']['position']})" for p in home_players]
        away_pos_list = [f"- {p['player']['shortName']} ({p['player']['position']})" for p in away_players]
        home_pos_summary = "\n".join(home_pos_list)
        away_pos_summary = "\n".join(away_pos_list)

        # --- Construct the Prompt ---
        prompt = f"""
        Analyze the following football match based on the data provided.

        MATCH: {home_team} vs. {away_team}
        FINAL SCORE: {home_score} - {away_score}

        KEY STATISTICS (Home vs. Away):
        {stats_summary}

        ATTACK MOMENTUM:
        {momentum_summary}

        STARTING LINEUPS (Average Position):
        Home Team ({home_team}):
        {home_pos_summary}

        Away Team ({away_team}):
        {away_pos_summary}

        YOUR ANALYSIS:
        Based on this data, provide a tactical analysis of the match. Discuss:
        1.  How the match statistics (like possession, shots, big chances) reflect the final score.
        2.  How the attack momentum flowed and which team capitalized on their periods of pressure.
        3.  What the average player positions suggest about each team's formation and strategy (e.g., defensive, high-press, wide-play).
        """
        
        return call_gemini_api(prompt)

    except Exception as e:
        st.error(f"Error preparing data for AI analysis: {e}")
        return "Error: Could not generate AI analysis due to missing data."


# ---------- Fetch ----------
@st.cache_data(show_spinner=False)  # Cache the data fetch
def fetch_json(url):
    """Fetches JSON from SofaScore API using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        sleep(2)  # Give it a moment to settle
        html = page.content()
        browser.close()
    m = re.search(r"<pre.*?>(.*?)</pre>", html, re.DOTALL)
    raw = m.group(1) if m else html
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error("Failed to decode JSON from page. The API structure might have changed or the page didn't load correctly.")
        return None

def get_starters(data, team):
    subs = data.get("substitutions", [])
    sub_ids = {s["playerIn"]["id"] for s in subs if s["isHome"] == (team == "home")}
    return [p for p in data.get(team, []) if p["player"]["id"] not in sub_ids]

def get_id_from_url(url):
    """Extracts the event ID from a SofaScore URL."""
    match = re.search(r'#id:(\d+)', url)
    if match:
        return match.group(1)
    
    # Fallback if #id: is not present (less common)
    match = re.search(r'/(\d+)$', url.split(',')[0])
    if match:
        return match.group(1)
        
    return None

# ---------- Draw ----------
def plot_match(event, avg, graph, stats):
    """Generates the Matplotlib figure. Returns fig."""
    home_team = event["event"]["homeTeam"]["name"]
    away_team = event["event"]["awayTeam"]["name"]
    home = get_starters(avg, "home")
    away = get_starters(avg, "away")
    
    # Use a dark background to match original style
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(11, 14)) # Increased height for better spacing
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2], hspace=0.4) # Increased hspace
    ax_pitch = fig.add_subplot(gs[0])
    ax_graph = fig.add_subplot(gs[1])
    ax_stats = fig.add_subplot(gs[2])
    
    fig.patch.set_facecolor('#0f0f0f') # Set figure background

    # ---- Opta Pitch ----
    pitch = Pitch(pitch_type="opta", axis=False, label=False, pitch_color="#15531a", line_color="white")
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_title(
        f"{home_team} (Blue) vs {away_team} (Red)\nAverage Player Positions",
        fontsize=14, color="white", weight="bold", pad=12,
    )

    def draw_team(players, mirror=False, color_main="blue"):
        for p in players:
            x, y = (100 - p["averageX"], p["averageY"]) if mirror else (p["averageX"], p["averageY"])
            num = p["player"].get("jerseyNumber", "")
            name = p["player"]["shortName"]
            pos = p["player"]["position"]
            color = "yellow" if pos == "G" else color_main
            pitch.scatter(x, y, ax=ax_pitch, c=color, s=300, edgecolors="black", zorder=3)
            ax_pitch.text(x, y, str(num), color="white", fontsize=9, ha="center", va="center", weight="bold", zorder=4)
            ax_pitch.text(x, y - 2.5, name, color="white", fontsize=7, ha="center", va="top", zorder=4) # Adjusted position

    draw_team(home, mirror=False, color_main="#3182bd") # Brighter Blue
    draw_team(away, mirror=True, color_main="#e53e3e")  # Brighter Red

    # ---------- Attack Momentum ----------
    minutes, values = [], []
    if "graphPoints" in graph:
        minutes = [p["minute"] for p in graph["graphPoints"]]
        values = [p["value"] for p in graph["graphPoints"]]
    
    ax_graph.plot(minutes, values, color="white", lw=1.2)
    ax_graph.axhline(0, color="gray", lw=0.8, ls="--")
    ax_graph.fill_between(minutes, 0, values, where=[v > 0 for v in values], color="#3182bd", alpha=0.45)
    ax_graph.fill_between(minutes, 0, values, where=[v < 0 for v in values], color="#e53e3e", alpha=0.45)
    ax_graph.set_facecolor("#1a1a1a") # Darker face
    
    if minutes:
        max_min = max(minutes) + 1
        ax_graph.set_xlim(0, max_min)
        ticks = list(range(0, int(max_min) + 5, 10)) # 10 min ticks
        ax_graph.set_xticks(ticks)
        ax_graph.set_xticklabels([str(t) for t in ticks], color="white", fontsize=8)
    
    ax_graph.set_xlabel("Minute", color="white", fontsize=10)
    ax_graph.set_ylabel("Momentum", color="white", fontsize=10)
    ax_graph.tick_params(axis="y", colors="white", labelsize=8)
    ax_graph.set_title(
        f"Attack Momentum: {home_team} (Blue) vs {away_team} (Red)", color="white", fontsize=11, pad=8
    )

    # ---------- Stats: compact SofaScore style ----------
    ax_stats.set_facecolor("#1a1a1a")
    ax_stats.axis("off")

    overview = []
    if "statistics" in stats and stats["statistics"]:
        overview_group = next((g for g in stats["statistics"][0]["groups"] if g["groupName"] == "Match overview"), None)
        if overview_group:
            overview = overview_group["statisticsItems"]

    if overview:
        y_positions = list(range(len(overview)))
        ax_stats.set_ylim(-1, len(overview))

        for i, s in enumerate(overview):
            name = s["name"]
            home_val, away_val = s.get("homeValue", 0), s.get("awayValue", 0)
            
            # Handle potential string values (like possession '55%')
            try:
                home_val = float(str(home_val).replace('%', ''))
                away_val = float(str(away_val).replace('%', ''))
            except ValueError:
                home_val, away_val = 0, 0

            max_val = max(home_val, away_val) or 1
            h_ratio, a_ratio = home_val / max_val, away_val / max_val

            y = len(overview) - i - 1
            ax_stats.barh(y, h_ratio, color="#3182bd", height=0.5, align="center", alpha=0.7)
            ax_stats.barh(y, -a_ratio, color="#e53e3e", height=0.5, align="center", alpha=0.7)

            # Display values
            home_str = s.get("home", str(home_val))
            away_str = s.get("away", str(away_val))

            ax_stats.text(0.05, y, home_str, color="white", fontsize=9, ha="left", va="center", weight="bold")
            ax_stats.text(-0.05, y, away_str, color="white", fontsize=9, ha="right", va="center", weight="bold")
            ax_stats.text(0, y, name, color="white", fontsize=9, ha="center", va="center")

        ax_stats.set_xlim(-1.2, 1.2)
        ax_stats.set_title("Match Overview Statistics", color="white", fontsize=12, pad=10)

    plt.tight_layout()
    return fig


# ---------- Main Streamlit App ----------
def run_app():
    st.set_page_config(layout="wide", page_title="SofaScore AI Analyst")
    st.title("⚽ SofaScore AI Match Analyst")
    
    st.info("Paste a full SofaScore match URL (e.g., `https://www.sofascore.com/...#id:123456`)")
    url = st.text_input(
        "SofaScore Match URL:", 
        "https://www.sofascore.com/football/match/fk-spartak-subotica-fk-crvena-zvezda/ZccsrOo#id:14015522"
    )

    if st.button("Analyze Match"):
        if not url:
            st.warning("Please paste a URL first.")
            st.stop()
            
        event_id = get_id_from_url(url)
        if not event_id:
            st.error("Could not find an Event ID in the URL. Please use a valid SofaScore match URL containing '#id:...'")
            st.stop()

        base_api_url = f"https://www.sofascore.com/api/v1/event/{event_id}"
        
        with st.spinner("Fetching data and analyzing match... This may take a moment."):
            try:
                # Fetch all data
                event_data = fetch_json(base_api_url)
                avg_data = fetch_json(f"{base_api_url}/average-positions")
                graph_data = fetch_json(f"{base_api_url}/graph")
                stats_data = fetch_json(f"{base_api_url}/statistics")

                if not all([event_data, avg_data, graph_data, stats_data]):
                    st.error("Failed to fetch all required data. Please check the URL and try again.")
                    st.stop()

                # Get AI Analysis
                st.subheader("🤖 AI Match Analysis")
                ai_analysis = get_ai_analysis(event_data, avg_data, graph_data, stats_data)
                st.markdown(ai_analysis)

                # Plot Match Visuals
                st.subheader("📊 Match Visualizations")
                fig = plot_match(event_data, avg_data, graph_data, stats_data)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)


if __name__ == "__main__":
    run_app()
