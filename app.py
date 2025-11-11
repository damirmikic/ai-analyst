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

# ---------- Playwright Installation Fix ----------
# This runs once when the app starts, to make sure Playwright's browser is installed.
@st.cache_resource
def install_playwright():
    """
    Installs the Playwright Chromium browser executable in the Streamlit environment.
    This is cached to run only once per app startup.
    """
    st.write("Installing browser... This may take a moment.")
    try:
        # We specify 'chromium' to avoid downloading all browsers
        subprocess.run(["playwright", "install", "chromium"], check=True, timeout=300)
        st.write("Browser installation complete.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        st.error(f"Failed to install Playwright browser: {e}")
        st.error("Please ensure your packages.txt file is correctly set up if deploying.")
    except subprocess.TimeoutExpired:
        st.error("Browser installation timed out.")

# Run the installation
install_playwright()

# ---------- Fetch Data ----------
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def fetch_json(url):
    """
    Fetches JSON data from a URL using Playwright to render the page first.
    This is necessary for SofaScore as it loads data dynamically.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            sleep(2)  # Give it a moment just in case
            
            # Find the JSON data embedded in a <pre> tag (SofaScore API)
            content = page.content()
            browser.close()

        m = re.search(r"<pre.*?>(.*?)</pre>", content, re.DOTALL)
        if m:
            raw = m.group(1)
            return json.loads(raw)
        else:
            # Fallback for pages that might not have the <pre> tag but are JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                st.error(f"Failed to find JSON data on page: {url}")
                return None
    except Exception as e:
        st.error(f"Error fetching data with Playwright: {e}")
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


# ---------- AI Analysis ----------
@st.cache_data(ttl=600)
def get_ai_analysis(event_data, avg_data, graph_data, stats_data):
    """
    Generates a professional match analysis using the Gemini API.
    """
    # Retrieve the API key from Streamlit's secrets
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return "Please add your `GEMINI_API_KEY` to the Streamlit secrets to enable AI analysis."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    # --- Simplify data for the prompt ---
    try:
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        
        # Format stats
        overview_group = next(
            g for g in stats_data["statistics"][0]["groups"] if g["groupName"] == "Match overview"
        )
        stats_list = []
        for item in overview_group["statisticsItems"]:
            stats_list.append(
                f"{item['name']}: {home_team} {item['homeValue']} - {away_team} {item['awayValue']}"
            )
        stats_summary = "\n".join(stats_list)

        # Format attack momentum
        momentum_summary = (
            f"Attack momentum data points (positive for {home_team}, negative for {away_team}): "
            + ", ".join([str(p["value"]) for p in graph_data["graphPoints"][:20]]) # Limit points
        )

        # Format average positions (just player names and positions)
        home_players = [
            f"{p['player']['shortName']} ({p['player']['position']})"
            for p in get_starters(avg_data, "home")
        ]
        away_players = [
            f"{p['player']['shortName']} ({p['player']['position']})"
            for p in get_starters(avg_data, "away")
        ]
        
        prompt = f"""
        Act as a professional football analyst. Your task is to provide a concise, insightful match report.
        Do not just list the stats; interpret them.
        
        Here is the data:
        Home Team: {home_team}
        Away Team: {away_team}

        Key Match Statistics:
        {stats_summary}

        Attack Momentum (Positive values favor {home_team}, Negative values favor {away_team}):
        {momentum_summary}

        Starting Formations (Player Name (Position)):
        {home_team}: {', '.join(home_players)}
        {away_team}: {', '.join(away_players)}

        Based on all this data, please provide:
        1.  **Match Summary:** A short paragraph describing the overall narrative of the match.
        2.  **Tactical Analysis:** An analysis of the teams' tactics. Who was more dominant? How did the average positions and momentum reflect the stats?
        3.  **Key Performer:** Based on the data, suggest a key performing area or dynamic (e.g., "Home team's midfield control" or "Away team's counter-attack").
        
        Be professional, insightful, and concise.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [{"text": "You are a world-class football analyst."}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 8192,
            },
        }

        headers = {"Content-Type": "application/json"}
        
        # Use exponential backoff for retries
        for i in range(3):  # Retry up to 3 times
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                
                result = response.json()
                if "candidates" in result:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    st.warning(f"AI response format unexpected: {result}")
                    return "AI analysis failed (unexpected response format)."
            
            except requests.exceptions.RequestException as e:
                st.warning(f"AI analysis request failed (Attempt {i+1}): {e}")
                sleep(2**i) # Exponential backoff: 1s, 2s, 4s
                
        return "AI analysis failed after multiple attempts. Please check API key and network."

    except Exception as e:
        st.error(f"Error during AI analysis data preparation: {e}")
        return f"AI analysis failed. Could not prepare data. Error: {e}"


# ---------- Draw ----------
def plot_match(event_data, avg_data, graph_data, stats_data):
    """
    Generates and displays the Matplotlib plot with all match data.
    """
    try:
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        home_players = get_starters(avg_data, "home")
        away_players = get_starters(avg_data, "away")

        fig = plt.figure(figsize=(11, 12), facecolor="#0E1117") # Match Streamlit dark theme
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2], hspace=0.35)
        ax_pitch = fig.add_subplot(gs[0])
        ax_graph = fig.add_subplot(gs[1])
        ax_stats = fig.add_subplot(gs[2])
        
        fig.patch.set_facecolor("#0E1117")

        # ---- Opta Pitch ----
        pitch = Pitch(
            pitch_type="opta", axis=False, label=False, pitch_color="#067032", line_color="white"
        )
        pitch.draw(ax=ax_pitch)
        ax_pitch.set_title(
            f"{home_team} (Blue) vs {away_team} (Red)\nAverage Player Positions (Starters)",
            fontsize=14,
            color="white",
            weight="bold",
            pad=12,
        )

        def draw_team(players, mirror=False, color_main="blue"):
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
                pitch.scatter(
                    x, y, ax=ax_pitch, c=color, s=300, edgecolors="black", zorder=3
                )
                ax_pitch.text(
                    x,
                    y,
                    str(num),
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=4,
                )
                ax_pitch.text(
                    x, y - 2, name, color="white", fontsize=7, ha="center", va="center", zorder=4
                )

        draw_team(home_players, mirror=False, color_main="#3B82F6") # Blue
        draw_team(away_players, mirror=True, color_main="#EF4444") # Red

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
            color="white",
            fontsize=11,
            pad=8,
        )

        # ---------- Stats: compact SofaScore style ----------
        ax_stats.set_facecolor("#0E1117")
        ax_stats.axis("off")

        overview = next(
            g for g in stats_data["statistics"][0]["groups"] if g["groupName"] == "Match overview"
        )["statisticsItems"]
        
        # Filter out "Possession" to handle it separately
        possession = next((s for s in overview if s["name"] == "Ball possession"), None)
        overview_stats = [s for s in overview if s["name"] != "Ball possession"]

        y_positions = list(range(len(overview_stats)))
        ax_stats.set_ylim(-1, len(overview_stats))

        # Handle possession bar at the top
        if possession:
            h_val, a_val = possession["homeValue"], possession["awayValue"]
            total = h_val + a_val or 1
            h_ratio, a_ratio = h_val / total, a_val / total
            y_pos = len(overview_stats)
            
            # Home possession bar
            ax_stats.barh(y_pos, h_ratio, color="#3B82F6", height=0.6, align="center", edgecolor="white")
            ax_stats.text(h_ratio/2, y_pos, f"{h_val}%", color="white", fontsize=10, ha="center", va="center", weight="bold")
            
            # Away possession bar
            ax_stats.barh(y_pos, -a_ratio, color="#EF4444", height=0.6, align="center", edgecolor="white")
            ax_stats.text(-a_ratio/2, y_pos, f"{a_val}%", color="white", fontsize=10, ha="center", va="center", weight="bold")
            
            ax_stats.text(0, y_pos + 0.5, "Ball Possession", color="white", fontsize=11, ha="center", va="center", weight="bold")

        # Handle other stats
        for i, s in enumerate(overview_stats):
            name = s["name"]
            home_val, away_val = s["homeValue"], s["awayValue"]
            max_val = max(home_val, away_val) or 1
            h_ratio, a_ratio = home_val / max_val, away_val / max_val

            y = len(overview_stats) - i - 1
            ax_stats.barh(y, h_ratio, color="#3B82F6", height=0.4, align="center", alpha=0.7)
            ax_stats.barh(y, -a_ratio, color="#EF4444", height=0.4, align="center", alpha=0.7)

            ax_stats.text(-1.1, y, str(away_val), color="white", fontsize=9, ha="right", va="center")
            ax_stats.text(1.1, y, str(home_val), color="white", fontsize=9, ha="left", va="center")
            ax_stats.text(0, y, name, color="white", fontsize=9, ha="center", va="center", weight="bold")

        ax_stats.set_xlim(-1.2, 1.2)
        ax_stats.set_title("Match Overview Statistics", color="white", fontsize=12, pad=10)

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Failed to plot match data: {e}")
        st.error(f"Avg Data: {avg_data}")
        st.error(f"Stats Data: {stats_data}")
        return None

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(layout="wide", page_title="SofaScore AI Analyst")
    st.title("⚽ SofaScore AI Match Analyst")

    url = st.text_input(
        "Paste a SofaScore match URL",
        "https.www.sofascore.com/football/match/fk-spartak-subotica-fk-crvena-zvezda/ZccsrOo#id:14015522,tab:statistics",
    )

    if st.button("Analyze Match"):
        if not url:
            st.warning("Please paste a URL")
            return

        # Extract event ID from URL
        match = re.search(r"#id:(\d+)", url)
        if not match:
            match = re.search(r"/(\d+)$", url) # Fallback for simple URLs
            
        if not match:
            st.error("Could not find a valid Event ID in the URL.")
            return

        event_id = match.group(1)
        base_api_url = f"https://www.sofascore.com/api/v1/event/{event_id}"

        with st.spinner("Fetching and analyzing match data... This can take up to a minute."):
            try:
                # Fetch all data points
                event_data = fetch_json(base_api_url)
                avg_data = fetch_json(f"{base_api_url}/average-positions")
                graph_data = fetch_json(f"{base_api_url}/graph")
                stats_data = fetch_json(f"{base_api_url}/statistics")

                if not all([event_data, avg_data, graph_data, stats_data]):
                    st.error("Failed to fetch all required match data. The match may be too old or not supported.")
                    return

                # --- AI Analysis Section ---
                st.subheader("🤖 AI Analyst Report")
                with st.spinner("Asking the AI analyst for insights..."):
                    analysis = get_ai_analysis(event_data, avg_data, graph_data, stats_data)
                    st.markdown(analysis)

                # --- Visual Plot Section ---
                st.subheader("📊 Match Visualizations")
                fig = plot_match(event_data, avg_data, graph_data, stats_data)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Could not generate match plots.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
