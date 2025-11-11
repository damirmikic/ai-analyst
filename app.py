import sys
import streamlit as st
import json
import re
import requests
import subprocess
from time import sleep
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from playwright.sync_api import sync_playwright
import io

# -----------------------------------------------------------------------------
# App Configuration & Initial Setup
# -----------------------------------------------------------------------------

# Set the page configuration for a modern, wide layout
st.set_page_config(
    layout="wide",
    page_title="SofaScore AI Analyst",
    page_icon="⚽"
)

# Initialize session state for the chatbot and to hold match data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "match_data" not in st.session_state:
    st.session_state.match_data = None

# -----------------------------------------------------------------------------
# Playwright Installation (Cached)
# -----------------------------------------------------------------------------

@st.cache_resource
def install_playwright():
    """
    Installs the Playwright Chromium browser executable in the Streamlit environment.
    This is cached to run only once per app startup.
    """
    with st.spinner("Setting up the analysis engine (installing browser)..."):
        try:
            # We specify 'chromium' to avoid downloading all browsers
            subprocess.run(["playwright", "install", "chromium"], check=True, timeout=300)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            st.error(f"Failed to install Playwright browser. This app cannot continue. Error: {e}")
            st.stop()

# Run the installation
install_playwright()

# -----------------------------------------------------------------------------
# Data Fetching Functions
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def fetch_json(url):
    """
    Fetches JSON data from a URL using Playwright to render the page first.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            sleep(2)
            
            content = page.content()
            browser.close()

        m = re.search(r"<pre.*?>(.*?)</pre>", content, re.DOTALL)
        if m:
            raw = m.group(1)
            return json.loads(raw)
        else:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                st.error(f"Failed to find JSON data on page: {url}", icon="🚨")
                return None
    except Exception as e:
        st.error(f"Error fetching data with Playwright: {e}", icon="🚨")
        return None

def get_starters(data, team):
    """
    Filters out substituted players to get the starting lineup's average positions.
    """
    if data is None:
        return []
    subs = data.get("substitutions", [])
    sub_ids = {s["playerIn"]["id"] for s in subs if s["isHome"] == (team == "home")}
    return [p for p in data.get(team, []) if p["player"]["id"] not in sub_ids]

# -----------------------------------------------------------------------------
# AI Analysis & Chatbot Functions
# -----------------------------------------------------------------------------

def get_gemini_api_key():
    """Fetches the Gemini API key from Streamlit secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        st.error("Please add your `GEMINI_API_KEY` to the Streamlit secrets to enable AI features.", icon="🔐")
        return None
    return api_key

def format_player_data_for_ai(avg_data):
    """Formats average position data into a text string for the AI prompt."""
    if not avg_data:
        return "No average position data available.\n"
    
    player_strings = []
    
    home_players = get_starters(avg_data, "home")
    if home_players:
        player_strings.append("Home Team Average Positions (X, Y):")
        for p in home_players:
            player_strings.append(
                f"  - {p['player']['shortName']} (#{p['player'].get('jerseyNumber', 'N/A')}, {p['player']['position']}): X={p['averageX']:.1f}, Y={p['averageY']:.1f}"
            )
    
    away_players = get_starters(avg_data, "away")
    if away_players:
        player_strings.append("\nAway Team Average Positions (X, Y):")
        for p in away_players:
            player_strings.append(
                f"  - {p['player']['shortName']} (#{p['player'].get('jerseyNumber', 'N/A')}, {p['player']['position']}): X={p['averageX']:.1f}, Y={p['averageY']:.1f}"
            )
            
    return "\n".join(player_strings)

def format_stats_for_ai(stats_data, home_team, away_team):
    """Formats key stats into a text string for the AI prompt."""
    if not stats_data or "statistics" not in stats_data:
        return "No statistics available.\n"
        
    try:
        overview_group = next(
            g for g in stats_data["statistics"][0]["groups"] if g["groupName"] == "Match overview"
        )
        stats_list = ["Key Match Statistics:"]
        for item in overview_group["statisticsItems"]:
            stats_list.append(
                f"  - {item['name']}: {home_team} {item['homeValue']} - {away_team} {item['awayValue']}"
            )
        return "\n".join(stats_list)
    except StopIteration:
        return "Could not find 'Match overview' stats.\n"
    except Exception:
        return "Error parsing stats.\n"

@st.cache_data(ttl=600)
def get_ai_analysis_summary(api_key, event_data, avg_data, graph_data, stats_data):
    """
    Generates the main (cached) AI match summary.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    try:
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        
        stats_summary = format_stats_for_ai(stats_data, home_team, away_team)
        player_summary = format_player_data_for_ai(avg_data)

        # Format attack momentum
        momentum_summary = (
            f"Attack momentum data (positive for {home_team}, negative for {away_team}): "
            + ", ".join([f"{p['minute']}'_({p['value']})" for p in graph_data["graphPoints"][::5]]) # Sample every 5th point
        )

        prompt = f"""
        Act as a professional football analyst. Your task is to provide a concise, insightful, and well-written match report for {home_team} vs {away_team}.
        Do not just list the stats; interpret them to tell the story of the match.
        
        Here is the data:
        {stats_summary}

        {player_summary}

        {momentum_summary}

        Based on all this data, please provide:
        1.  **Match Summary (Headline):** A short, punchy paragraph describing the overall narrative and result of the match.
        2.  **Tactical Analysis:** An analysis of the teams' tactics. Who was more dominant and why? How did the average positions, momentum, and stats support this?
        3.  **Key Talking Point:** Based on the data, identify the single most important factor or dynamic that decided this match (e.g., "Home team's midfield control," "Away team's clinical finishing," "A tale of two halves").
        
        Be professional, insightful, and use engaging language.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [{"text": f"You are a world-class football analyst summarizing a match: {home_team} vs {away_team}."}]
            },
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
        }
        
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        st.error(f"Error during AI analysis: {e}", icon="🤖")
        return "AI analysis failed. Could not generate the report."

def get_chatbot_response(api_key, chat_history, match_context):
    """
    Gets a response from the AI chatbot based on the conversation history and match data.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    # Prepare the data context for the system prompt
    home_team = match_context["event_data"]["event"]["homeTeam"]["name"]
    away_team = match_context["event_data"]["event"]["awayTeam"]["name"]
    
    stats_summary = format_stats_for_ai(match_context["stats_data"], home_team, away_team)
    player_summary = format_player_data_for_ai(match_context["avg_data"])
    
    system_prompt = f"""
    You are a specialist Football Tactical Analyst Chatbot.
    You are analyzing one specific match: {home_team} vs. {away_team}.
    Your entire analysis MUST be based *only* on the data provided below.
    Do NOT invent any data (like scores, goals, or events) not present.
    
    Here is the complete data for this match:

    --- DATA START ---
    {stats_summary}
    
    {player_summary}
    --- DATA END ---

    The user will now ask you questions about this specific match.
    Answer their questions by interpreting the provided data. Focus on player positions, formations, and how they relate to the statistics.
    Be concise, insightful, and directly answer the user's question.
    """

    # Convert Streamlit history to Gemini format
    gemini_history = []
    for msg in chat_history:
        gemini_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "contents": gemini_history,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
    }

    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        st.error(f"Chatbot API call failed: {e}", icon="🤖")
        return "Sorry, I'm having trouble connecting to my tactical brain right now."
    except (KeyError, IndexError):
        st.error("Received an unexpected response from the AI. Please try again.", icon="🤖")
        return "Sorry, I got a confusing message from the AI. Could you rephrase that?"

# -----------------------------------------------------------------------------
# Plotting Function
# -----------------------------------------------------------------------------

def plot_match(event_data, avg_data, graph_data, stats_data):
    """
    Generates and displays the Matplotlib plot with all match data.
    """
    try:
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        home_players = get_starters(avg_data, "home")
        away_players = get_starters(avg_data, "away")

        # --- UPDATED: Wider figure for legends ---
        fig = plt.figure(figsize=(13, 12), facecolor="#0E1117") # Match Streamlit dark theme
        
        # --- UPDATED: Gridspec for pitch + legends ---
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2], hspace=0.35)
        
        # Create a sub-gridspec in the first row (gs[0]) for legends and pitch
        gs_top = gs[0].subgridspec(1, 3, width_ratios=[1.2, 4, 1.2], wspace=0.05)
        
        ax_home_legend = fig.add_subplot(gs_top[0])
        ax_pitch = fig.add_subplot(gs_top[1])
        ax_away_legend = fig.add_subplot(gs_top[2])
        
        ax_graph = fig.add_subplot(gs[1])
        ax_stats = fig.add_subplot(gs[2])
        
        fig.patch.set_facecolor("#0E1117")
        
        # --- NEW: Setup legend axes ---
        ax_home_legend.set_facecolor("#0E1117")
        ax_away_legend.set_facecolor("#0E1117")
        ax_home_legend.axis('off')
        ax_away_legend.axis('off')
        
        ax_home_legend.set_title(f"{home_team} (Blue)", color="white", fontsize=11, weight="bold", pad=10)
        ax_away_legend.set_title(f"{away_team} (Red)", color="white", fontsize=11, weight="bold", pad=10)

        # ---- Opta Pitch ----
        pitch = Pitch(
            pitch_type="opta", axis=False, label=False, pitch_color="#067032", line_color="white"
        )
        pitch.draw(ax=ax_pitch) # Draw on the middle axes
        ax_pitch.set_title(
            f"Average Player Positions (Starters)", # Simplified title
            fontsize=14, color="white", weight="bold", pad=12,
        )

        def draw_team(players, mirror=False, color_main="blue", ax_legend=None):
            legend_items = []
            
            for p in players:
                x, y = (
                    (100 - p["averageX"], p["averageY"])
                    if mirror
                    else (p["averageX"], p["averageY"])
                )
                num = p["player"].get("jerseyNumber", "")
                name = p["player"]["shortName"]
                pos = p["player"]["position"]
                color = "yellow" if pos == "G" else color_main
                
                # --- UPDATED: Only draw number on pitch ---
                pitch.scatter(
                    x, y, ax=ax_pitch, c=color, s=300, edgecolors="black", zorder=3
                )
                ax_pitch.text(
                    x, y, str(num), color="white", fontsize=9, ha="center", va="center", weight="bold", zorder=4,
                )
                # Player name text removed from pitch
                
                legend_items.append((num, name, color))

            # --- NEW: Draw legend on the provided axes ---
            if ax_legend:
                # Sort by number
                legend_items.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999)
                
                y_step = 0.045
                y_pos = 0.95
                for num, name, color in legend_items:
                    text_color = "yellow" if color == "yellow" else "white"
                    # Align numbers to the right, names to the left
                    ax_legend.text(0.35, y_pos, f"{num}", color=text_color, fontsize=9, weight="bold", ha="right", va="top")
                    ax_legend.text(0.4, y_pos, f"- {name}", color="white", fontsize=9, ha="left", va="top")
                    y_pos -= y_step

        # --- UPDATED: Call draw_team with legend axes ---
        draw_team(home_players, mirror=False, color_main="#3B82F6", ax_legend=ax_home_legend)
        draw_team(away_players, mirror=True, color_main="#EF4444", ax_legend=ax_away_legend)

        # ---------- Attack Momentum ----------
        minutes = [p["minute"] for p in graph_data["graphPoints"]]
        values = [p["value"] for p in graph_data["graphPoints"]]
        ax_graph.plot(minutes, values, color="white", lw=1.2)
        ax_graph.axhline(0, color="gray", lw=0.8, ls="--")
        ax_graph.fill_between(
            minutes, 0, values, where=[v > 0 for v in values], color="#3B82F6", alpha=0.45
        )
        ax_graph.fill_between(
            minutes, 0, values, where=[v < 0 for v in values], color="#EF4444", alpha=0.45
        )
        ax_graph.set_facecolor("#262730")
        ax_graph.set_xlim(0, max(minutes) + 1)
        ticks = list(range(0, int(max(minutes)) + 5, 5))
        ax_graph.set_xticks(ticks)
        ax_graph.set_xticklabels([str(t) for t in ticks], color="white", fontsize=8)
        ax_graph.set_xlabel("Minute", color="white", fontsize=10)
        ax_graph.set_ylabel("Momentum", color="white", fontsize=10)
        ax_graph.tick_params(axis="y", colors="white", labelsize=8)
        ax_graph.spines["top"].set_color("white")
        ax_graph.spines["bottom"].set_color("white")
        ax_graph.spines["left"].set_color("white")
        ax_graph.spines["right"].set_color("white")
        ax_graph.set_title(
            f"Attack Momentum: {home_team} (Blue) vs {away_team} (Red)",
            color="white", fontsize=11, pad=8,
        )

        # ---------- Stats: compact SofaScore style ----------
        ax_stats.set_facecolor("#0E1117")
        ax_stats.axis("off")

        overview = next(
            g for g in stats_data["statistics"][0]["groups"] if g["groupName"] == "Match overview"
        )["statisticsItems"]
        
        possession = next((s for s in overview if s["name"] == "Ball possession"), None)
        overview_stats = [s for s in overview if s["name"] != "Ball possession"]

        y_positions = list(range(len(overview_stats)))
        ax_stats.set_ylim(-1, len(overview_stats) + 1) # Make room for possession

        # Handle possession bar at the top
        if possession:
            h_val, a_val = possession["homeValue"], possession["awayValue"]
            total = h_val + a_val or 1
            h_ratio, a_ratio = h_val / total, a_val / total
            y_pos = len(overview_stats)
            
            # --- UPDATED: Reversed possession bar ---
            # Home (blue) now negative/left
            ax_stats.barh(y_pos, -h_ratio, color="#3B82F6", height=0.6, align="center", edgecolor="white")
            ax_stats.text(-h_ratio/2, y_pos, f"{h_val}%", color="white", fontsize=10, ha="center", va="center", weight="bold")
            # Away (red) now positive/right
            ax_stats.barh(y_pos, a_ratio, color="#EF4444", height=0.6, align="center", edgecolor="white")
            ax_stats.text(a_ratio/2, y_pos, f"{a_val}%", color="white", fontsize=10, ha="center", va="center", weight="bold")
            
            ax_stats.text(0, y_pos + 0.6, "Ball Possession", color="white", fontsize=11, ha="center", va="center", weight="bold")

        # Handle other stats
        for i, s in enumerate(overview_stats):
            name = s["name"]
            home_val, away_val = s["homeValue"], s["awayValue"]
            max_val = max(home_val, away_val) or 1
            h_ratio, a_ratio = home_val / max_val, away_val / max_val

            y = len(overview_stats) - i - 1
            
            # --- UPDATED: Reversed stats bars ---
            # Home (blue) now negative/left
            ax_stats.barh(y, -h_ratio, color="#3B82F6", height=0.4, align="center", alpha=0.7)
            # Away (red) now positive/right
            ax_stats.barh(y, a_ratio, color="#EF4444", height=0.4, align="center", alpha=0.7)

            # --- UPDATED: Reversed text labels ---
            # Home val (blue) on the left
            ax_stats.text(-1.1, y, str(home_val), color="white", fontsize=9, ha="right", va="center")
            # Away val (red) on the right
            ax_stats.text(1.1, y, str(away_val), color="white", fontsize=9, ha="left", va="center")
            
            ax_stats.text(0, y, name, color="white", fontsize=9, ha="center", va="center", weight="bold")

        ax_stats.set_xlim(-1.2, 1.2)
        ax_stats.set_title("Match Overview Statistics", color="white", fontsize=12, pad=20)

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Failed to plot match data: {e}", icon="📈")
        return None

# -----------------------------------------------------------------------------
# Main Streamlit App UI
# -----------------------------------------------------------------------------

def main():
    st.title("⚽ SofaScore AI Match Analyst")
    st.markdown("Paste a SofaScore match URL to get a deep-dive tactical analysis, visual charts, and an interactive AI chatbot.")

    url = st.text_input(
        "Paste a SofaScore match URL",
        "https.www.sofascore.com/football/match/manchester-city-real-madrid/KcsU",
        key="url_input"
    )

    if st.button("Analyze Match", type="primary"):
        if not url:
            st.warning("Please paste a URL to analyze.", icon="👇")
            return

        # Extract event ID from URL
        match = re.search(r"#id:(\d+)", url)
        if not match:
            match = re.search(r"/(\d+)$", url) # Fallback for simple URLs
        if not match:
             # Fallback for URLs like /manchester-city-real-madrid/KcsU
            match = re.search(r'/([^/]+)/(\d+)$', url.split('#')[0])
            if not match:
                match = re.search(r'/([^/]+)$', url.split('#')[0])
                if match and not match.group(1).isdigit(): # Check if it's not just the ID
                     st.error("Could not find a valid Event ID in the URL. Please use a valid SofaScore match URL.", icon="🔗")
                     return
                elif not match:
                     st.error("Could not find a valid Event ID in the URL. Please use a valid SofaScore match URL.", icon="🔗")
                     return

        event_id = match.group(1) if match.group(1).isdigit() else match.group(2)
        
        # --- FIX: Corrected https:// ---
        base_api_url = f"https://www.sofascore.com/api/v1/event/{event_id}"

        # Clear previous chat history and data
        st.session_state.messages = []
        st.session_state.match_data = None
        
        # All data fetching and processing happens here
        try:
            with st.spinner("Brewing the tactical insights... Fetching match data..."):
                event_data = fetch_json(base_api_url)
            with st.spinner("Analyzing player movements... Fetching average positions..."):
                avg_data = fetch_json(f"{base_api_url}/average-positions")
            with st.spinner("Reading the game's flow... Fetching attack momentum..."):
                graph_data = fetch_json(f"{base_api_url}/graph")
            with st.spinner("Counting the stats... Fetching statistics..."):
                stats_data = fetch_json(f"{base_api_url}/statistics")

            if not all([event_data, avg_data, graph_data, stats_data]):
                st.error("Failed to fetch all required match data. The match may be too old, not yet played, or not supported.", icon="🚨")
                return

            # Store all fetched data in session_state for the chatbot
            st.session_state.match_data = {
                "event_data": event_data,
                "avg_data": avg_data,
                "graph_data": graph_data,
                "stats_data": stats_data
            }
            
            # Generate the main AI summary
            api_key = get_gemini_api_key()
            if api_key:
                with st.spinner("Summoning the AI analyst for the match report..."):
                    summary = get_ai_analysis_summary(api_key, event_data, avg_data, graph_data, stats_data)
                    st.session_state.match_data["ai_summary"] = summary
            else:
                 st.session_state.match_data["ai_summary"] = "AI analysis is disabled. Please add your Gemini API key."
            
            # Generate the plot
            with st.spinner("Plotting the pitch and stats..."):
                 fig = plot_match(event_data, avg_data, graph_data, stats_data)
                 st.session_state.match_data["plot_fig"] = fig


        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}", icon="🔥")
            st.exception(e)
            return

    # -------------------------------------------------------------------------
    # Display Results using TABS (only if data is loaded)
    # -------------------------------------------------------------------------
    if st.session_state.match_data:
        
        match_data = st.session_state.match_data
        home_team = match_data["event_data"]["event"]["homeTeam"]["name"]
        away_team = match_data["event_data"]["event"]["awayTeam"]["name"]
        
        # --- UPDATED: Add match score to header ---
        home_score = match_data["event_data"]["event"]["homeScore"]["current"]
        away_score = match_data["event_data"]["event"]["awayScore"]["current"]
        st.header(f"Analysis: {home_team} {home_score} - {away_score} {away_team}")
        
        tab1, tab2, tab3 = st.tabs(["🤖 AI Analyst Report", "📊 Visual Insights", "💬 Tactical Chatbot"])

        with tab1:
            st.subheader("AI Match Report")
            summary_text = match_data.get("ai_summary", "No summary available.")
            st.markdown(summary_text)
            
            # --- Download Button ---
            st.download_button(
                label="📥 Download Report",
                data=summary_text,
                file_name=f"{home_team}_vs_{away_team}_analysis.txt",
                mime="text/plain",
            )

        with tab2:
            st.subheader("Visual Data")
            fig = match_data.get("plot_fig")
            if fig:
                st.pyplot(fig)
            else:
                st.warning("The plot for this match could not be generated.")

        with tab3:
            st.subheader("Tactical Chatbot")
            st.markdown("Ask the AI about player positions, team formations, and stats from *this specific match*.")
            
            api_key = get_gemini_api_key()
            
            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("Ask about a player's position or the team's formation..."):
                if not api_key:
                    st.error("Chatbot is disabled. Please add your Gemini API key to Streamlit secrets.", icon="🔐")
                    return
                    
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_chatbot_response(api_key, st.session_state.messages, st.session_state.match_data)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
