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
from collections import Counter

# -----------------------------------------------------------------------------
# App Configuration & Initial Setup
# -----------------------------------------------------------------------------

# Set the page configuration for a modern, wide layout
st.set_page_config(
    layout="wide",
    page_title="SofaScore AI Analyst",
    page_icon="⚽"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "match_data" not in st.session_state:
    st.session_state.match_data = None
if "player_analysis_cache" not in st.session_state:
    st.session_state.player_analysis_cache = {} # For the new player analysis tab

# -----------------------------------------------------------------------------
# Playwright Installation (Cached)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def install_playwright():
    """
    Installs the Playwright Chromium browser executable in the Streamlit environment.
    This is cached to run only once per app startup.
    """
    with st.spinner("🚀 Deploying scouting drones to prep the match browser..."):
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

@st.cache_data(ttl=600, show_spinner=False)  # Cache data for 10 minutes
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

        # Try to find JSON inside <pre> tag first (common on sofascore)
        m = re.search(r"<pre.*?>(.*?)</pre>", content, re.DOTALL)
        if m:
            raw = m.group(1)
            return json.loads(raw)
        else:
            # If no <pre> tag, try to parse the whole content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                st.error(f"Failed to find JSON data on page: {url}", icon="🚨")
                return None
    except Exception as e:
        st.error(f"Error fetching data with Playwright: {e}", icon="🚨")
        return None

def camel_to_title(text):
    """Convert camelCase or snake_case keys to Title Case labels."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("_", " ")
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
    return text.strip().title()


def get_starters(data, team):
    """
    Filters out substituted players to get the starting lineup's average positions.
    """
    if data is None:
        return []
    subs = data.get("substitutions", [])
    sub_ids = {s["playerIn"]["id"] for s in subs if s["isHome"] == (team == "home")}
    return [p for p in data.get(team, []) if p["player"]["id"] not in sub_ids]

def get_all_players(lineup_data):
    """
    Gets a list of all players (starters and subs) from the lineup data for the dropdown.
    """
    players = []
    if not lineup_data:
        return players

    home_players = lineup_data.get("home", {}).get("players", [])
    away_players = lineup_data.get("away", {}).get("players", [])

    for p in home_players:
        players.append({"name": p["player"]["name"], "id": p["player"]["id"], "team": "home"})
    for p in away_players:
        players.append({"name": p["player"]["name"], "id": p["player"]["id"], "team": "away"})
    
    return players

def get_player_by_id(lineup_data, player_id, team):
    """
    Finds a specific player's full data from the lineup JSON.
    """
    if not lineup_data:
        return None
    
    team_players = lineup_data.get(team, {}).get("players", [])
    for p in team_players:
        if p["player"]["id"] == player_id:
            return p
    return None


@st.cache_data(ttl=600)
def fetch_player_heatmap(event_id, player_id):
    """Fetch heatmap information for a player in a specific match."""
    if not event_id or not player_id:
        return None
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/player/{player_id}/heatmap"
    return fetch_json(url)


@st.cache_data(ttl=600)
def fetch_player_season_statistics(player_id):
    """Fetch aggregated seasonal statistics for a player."""
    if not player_id:
        return None
    url = f"https://www.sofascore.com/api/v1/player/{player_id}/statistics"
    return fetch_json(url)


@st.cache_data(ttl=600)
def fetch_player_attribute_overviews(player_id):
    """Fetch SofaScore attribute overview data for a player."""
    if not player_id:
        return None
    url = f"https://www.sofascore.com/api/v1/player/{player_id}/attribute-overviews"
    return fetch_json(url)


def format_heatmap_summary(heatmap_data):
    """Create a concise textual summary from heatmap JSON data."""
    if not heatmap_data:
        return "No heatmap data available."

    lines = ["Heatmap Insights:"]

    # Gather potential dict containers that may hold metadata
    metric_candidates = []
    if isinstance(heatmap_data, dict):
        metric_candidates.append(heatmap_data)
        inner = heatmap_data.get("heatmap")
        if isinstance(inner, dict):
            metric_candidates.append(inner)

    metrics_added = {"total": False, "max": False, "peak": False, "zones": False}
    for candidate in metric_candidates:
        if not isinstance(candidate, dict):
            continue

        total_actions = candidate.get("total") or candidate.get("eventsCount")
        if not metrics_added["total"] and total_actions:
            lines.append(f"- Total recorded actions: {total_actions}")
            metrics_added["total"] = True

        max_value = candidate.get("max") or candidate.get("maxValue")
        if not metrics_added["max"] and max_value is not None:
            lines.append(f"- Peak intensity value: {max_value}")
            metrics_added["max"] = True

        peak_coordinates = (
            candidate.get("maxTouchesCoordinate")
            or candidate.get("maxCoordinate")
            or candidate.get("peakCoordinate")
        )
        if not metrics_added["peak"] and isinstance(peak_coordinates, dict):
            x = peak_coordinates.get("x")
            y = peak_coordinates.get("y")
            if x is not None and y is not None:
                lines.append(f"- Hottest zone around pitch coordinates (x={x}, y={y})")
                metrics_added["peak"] = True

        zones = candidate.get("zones") or candidate.get("clusters")
        if not metrics_added["zones"] and isinstance(zones, list) and zones:
            sorted_zones = sorted(
                [z for z in zones if isinstance(z, dict) and z.get("value")],
                key=lambda z: z.get("value", 0),
                reverse=True,
            )
            top_zones = sorted_zones[:3]
            if top_zones:
                lines.append("- Top hot zones:")
                for zone in top_zones:
                    label = camel_to_title(zone.get("name") or zone.get("zone") or "Zone")
                    value = zone.get("value")
                    lines.append(f"  • {label}: intensity {value}")
                metrics_added["zones"] = True

    # Normalise coordinate list format like the sample payload `{"heatmap": [{"x": .., "y": ..}, ...]}`
    coord_list = []
    if isinstance(heatmap_data, dict):
        raw_coords = heatmap_data.get("heatmap")
        if isinstance(raw_coords, list):
            coord_list = raw_coords
        elif isinstance(raw_coords, dict):
            maybe_points = raw_coords.get("points") or raw_coords.get("heatmap")
            if isinstance(maybe_points, list):
                coord_list = maybe_points
        if not coord_list:
            fallback = heatmap_data.get("points") or heatmap_data.get("coordinates")
            if isinstance(fallback, list):
                coord_list = fallback
    elif isinstance(heatmap_data, list):
        coord_list = heatmap_data

    valid_points = [
        {"x": float(p["x"]), "y": float(p["y"])}
        for p in coord_list
        if isinstance(p, dict)
        and isinstance(p.get("x"), (int, float))
        and isinstance(p.get("y"), (int, float))
    ]

    if valid_points:
        total_points = len(valid_points)
        avg_x = sum(p["x"] for p in valid_points) / total_points
        avg_y = sum(p["y"] for p in valid_points) / total_points
        min_x = min(p["x"] for p in valid_points)
        max_x = max(p["x"] for p in valid_points)
        min_y = min(p["y"] for p in valid_points)
        max_y = max(p["y"] for p in valid_points)

        if not metrics_added["total"]:
            lines.append(f"- Heatmap points captured: {total_points}")
        else:
            lines.append(f"- Coordinate samples recorded: {total_points}")

        lines.append(
            f"- Average action location at x={avg_x:.1f}, y={avg_y:.1f} on SofaScore's 0-100 pitch scale"
        )
        lines.append(
            f"- Activity spread covers x {min_x:.0f}–{max_x:.0f} and y {min_y:.0f}–{max_y:.0f}"
        )

        def classify_third(x_val):
            if x_val < 33.3:
                return "Defensive third"
            if x_val < 66.6:
                return "Middle third"
            return "Attacking third"

        def classify_lane(y_val):
            if y_val < 33.3:
                return "left wing"
            if y_val < 66.6:
                return "central lane"
            return "right wing"

        third_counts = Counter(classify_third(p["x"]) for p in valid_points)
        lane_counts = Counter(classify_lane(p["y"]) for p in valid_points)
        zone_counts = Counter(
            (classify_third(p["x"]), classify_lane(p["y"]))
            for p in valid_points
        )

        dominant_third, third_count = third_counts.most_common(1)[0]
        dominant_lane, lane_count = lane_counts.most_common(1)[0]
        (zone_third, zone_lane), zone_count = zone_counts.most_common(1)[0]

        lines.append(
            f"- {dominant_third} contained {third_count / total_points * 100:.0f}% of their recorded actions"
        )
        lines.append(
            f"- Preference for the {dominant_lane} ({lane_count / total_points * 100:.0f}% of touches)"
        )
        lines.append(
            f"- Busiest zone: {zone_third} / {zone_lane} ({zone_count / total_points * 100:.0f}% of samples)"
        )

    if len(lines) == 1:
        lines.append("- Heatmap data structure available but no key metrics identified.")

    return "\n".join(lines)


def format_season_stats_for_ai(statistics_data):
    """Convert the seasonal statistics response into a readable summary."""
    if not statistics_data or not isinstance(statistics_data, dict):
        return "No seasonal statistics available."

    def format_number(value):
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    # ------------------------------------------------------------------
    # Legacy aggregate payload: {"statistics": {"total": {...}, ...}}
    # ------------------------------------------------------------------
    base_stats = statistics_data.get("statistics")
    if isinstance(base_stats, dict):
        lines = []

        def add_section(title, payload, limit=None):
            if not isinstance(payload, dict) or not payload:
                return
            lines.append(f"{title}:")
            items = list(payload.items())
            if limit:
                items = items[:limit]
            for key, value in items:
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip()):
                    lines.append(f"- {camel_to_title(key)}: {format_number(value)}")

        add_section("Season Totals", base_stats.get("total"))
        add_section("Per 90 Metrics", base_stats.get("per90"))
        add_section("Season Averages", base_stats.get("average"))

        tournaments = statistics_data.get("tournaments")
        if isinstance(tournaments, list) and tournaments:
            lines.append("Competition Highlights:")
            for tournament in tournaments[:3]:
                name = tournament.get("name") or tournament.get("tournament", {}).get("name")
                appearances = tournament.get("appearances") or tournament.get("statistics", {}).get("appearances")
                goals = tournament.get("goals") or tournament.get("statistics", {}).get("goals")
                assists = tournament.get("assists") or tournament.get("statistics", {}).get("assists")
                parts = []
                if appearances is not None:
                    parts.append(f"Apps: {format_number(appearances)}")
                if goals is not None:
                    parts.append(f"Goals: {format_number(goals)}")
                if assists is not None:
                    parts.append(f"Assists: {format_number(assists)}")
                summary = ", ".join(parts) if parts else "No basic stats available"
                if name:
                    lines.append(f"- {name}: {summary}")

        if lines:
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Newer payload: {"seasons": [{"statistics": {...}} ...]}
    # ------------------------------------------------------------------
    seasons = statistics_data.get("seasons")
    if isinstance(seasons, list) and seasons:
        lines = ["Season-by-season performance:"]

        def format_accuracy_line(label, made_key, total_key=None, pct_key=None, stats=None):
            stats = stats or {}
            made = stats.get(made_key) if made_key else None
            total = stats.get(total_key) if total_key else None
            pct = stats.get(pct_key) if pct_key else None
            if made is None and total is None and pct is None:
                return None

            parts = []
            if made is not None:
                ratio = format_number(made)
                if total is not None:
                    ratio += f"/{format_number(total)}"
                parts.append(ratio)
            elif total is not None:
                parts.append(f"?/{format_number(total)}")

            if pct is not None:
                parts.append(f"{format_number(pct)}%")

            return f"{label}: {' '.join(parts)}" if parts else None

        for idx, season in enumerate(seasons, start=1):
            stats = season.get("statistics") if isinstance(season, dict) else None
            if not isinstance(stats, dict) or not stats:
                continue

            season_info = season.get("season") if isinstance(season, dict) else None
            season_label = None
            if isinstance(season_info, dict):
                for key in ("name", "displayName", "year", "slug"):
                    value = season_info.get(key)
                    if value:
                        season_label = str(value)
                        break
            if not season_label:
                season_label = str(season.get("name")) if season.get("name") else None
            if not season_label:
                season_label = f"Season {idx}"

            type_label = stats.get("type")
            header = season_label
            if type_label:
                header += f" ({camel_to_title(type_label)})"

            lines.append(f"- {header}:")

            summary_metrics = [
                ("appearances", "Apps"),
                ("minutesPlayed", "Minutes"),
                ("goals", "Goals"),
                ("assists", "Assists"),
                ("goalsAssistsSum", "G+A"),
                ("expectedGoals", "xG"),
                ("expectedAssists", "xA"),
                ("rating", "Rating"),
            ]

            summary_parts = [
                f"{label}: {format_number(stats[key])}"
                for key, label in summary_metrics
                if stats.get(key) is not None
            ]
            if summary_parts:
                lines.append("  • Summary: " + ", ".join(summary_parts))

            passing_parts = []
            passes_line = format_accuracy_line(
                "Passes",
                "accuratePasses",
                total_key="totalPasses",
                pct_key="accuratePassesPercentage",
                stats=stats,
            )
            if passes_line:
                passing_parts.append(passes_line)

            key_passes = stats.get("keyPasses")
            if key_passes is not None:
                passing_parts.append(f"Key passes: {format_number(key_passes)}")

            long_balls_line = format_accuracy_line(
                "Long balls",
                "accurateLongBalls",
                total_key="totalLongBalls",
                pct_key="accurateLongBallsPercentage",
                stats=stats,
            )
            if long_balls_line:
                passing_parts.append(long_balls_line)

            crosses_line = format_accuracy_line(
                "Crosses",
                "accurateCrosses",
                total_key="totalCross",
                pct_key="accurateCrossesPercentage",
                stats=stats,
            )
            if crosses_line:
                passing_parts.append(crosses_line)

            if passing_parts:
                lines.append("  • Passing: " + "; ".join(passing_parts))

            attacking_metrics = [
                ("totalShots", "Shots"),
                ("shotsOnTarget", "On target"),
                ("successfulDribbles", "Successful dribbles"),
                ("bigChancesCreated", "Big chances created"),
                ("bigChancesMissed", "Big chances missed"),
            ]
            attacking_parts = [
                f"{label}: {format_number(stats[key])}"
                for key, label in attacking_metrics
                if stats.get(key) is not None
            ]
            if attacking_parts:
                lines.append("  • Attacking: " + ", ".join(attacking_parts))

            defensive_metrics = [
                ("tackles", "Tackles"),
                ("interceptions", "Interceptions"),
                ("aerialDuelsWon", "Aerial duels won"),
                ("dribbledPast", "Dribbled past"),
                ("goalsConceded", "Goals conceded"),
                ("cleanSheet", "Clean sheets"),
                ("errorLeadToGoal", "Errors leading to goal"),
            ]
            defensive_parts = [
                f"{label}: {format_number(stats[key])}"
                for key, label in defensive_metrics
                if stats.get(key) is not None
            ]
            if defensive_parts:
                lines.append("  • Defensive: " + ", ".join(defensive_parts))

            discipline_metrics = [
                ("yellowCards", "Yellow cards"),
                ("redCards", "Red cards"),
            ]
            discipline_parts = [
                f"{label}: {format_number(stats[key])}"
                for key, label in discipline_metrics
                if stats.get(key) is not None
            ]
            if discipline_parts:
                lines.append("  • Discipline: " + ", ".join(discipline_parts))

        if len(lines) > 1:
            return "\n".join(lines)

    return "Season statistics fetched but no readable metrics were identified."


def format_attribute_overviews_for_ai(attribute_data):
    """Format the attribute overview response into bullet points."""
    if not attribute_data:
        return "No attribute overview data available."

    if not isinstance(attribute_data, dict):
        return "No attribute overview data available."

    # ------------------------------------------------------------------
    # Legacy format handling: {"attributeOverviews": [{...}]}
    # ------------------------------------------------------------------
    overviews = attribute_data.get("attributeOverviews")
    if isinstance(overviews, list) and overviews:
        lines = ["Attribute Overview:"]
        for overview in overviews:
            group_name = overview.get("groupName") or overview.get("name")
            if group_name:
                lines.append(f"- {group_name}:")
            attributes = overview.get("attributes")
            if isinstance(attributes, list):
                for attribute in attributes[:6]:  # limit to keep prompt concise
                    attr_name = camel_to_title(attribute.get("name") or attribute.get("attribute"))
                    value = attribute.get("value")
                    if attr_name and value is not None:
                        lines.append(f"  • {attr_name}: {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Newer format handling: separate average & per-season player metrics
    # ------------------------------------------------------------------
    player_overviews = attribute_data.get("playerAttributeOverviews")
    average_overviews = attribute_data.get("averageAttributeOverviews")

    if not isinstance(player_overviews, list) or not player_overviews:
        return "No attribute overview data available."

    def year_shift_label(year_shift):
        if year_shift == 0:
            return "Current season"
        if year_shift == 1:
            return "Last season"
        if year_shift is None:
            return "Season overview"
        return f"{year_shift} seasons ago"

    avg_lookup = {}
    if isinstance(average_overviews, list):
        for avg in average_overviews:
            if isinstance(avg, dict):
                avg_lookup[avg.get("yearShift")] = avg

    metric_order = ["attacking", "technical", "tactical", "defending", "creativity"]

    def format_value(val):
        if isinstance(val, float):
            return f"{val:.1f}" if not val.is_integer() else f"{int(val)}"
        return str(val)

    lines = ["Attribute Overview:"]
    for overview in sorted(
        [o for o in player_overviews if isinstance(o, dict)],
        key=lambda o: o.get("yearShift", 0),
    ):
        year_shift = overview.get("yearShift")
        label = year_shift_label(year_shift)
        position = overview.get("position")
        header = f"- {label}"
        if position:
            header += f" ({position})"
        lines.append(header + ":")

        avg_for_year = avg_lookup.get(year_shift)

        for metric in metric_order:
            value = overview.get(metric)
            if value is None:
                continue

            line = f"  • {camel_to_title(metric)}: {format_value(value)}"

            if isinstance(avg_for_year, dict) and isinstance(value, (int, float)):
                avg_value = avg_for_year.get(metric)
                if isinstance(avg_value, (int, float)):
                    diff = value - avg_value
                    if isinstance(diff, float) and not diff.is_integer():
                        diff_display = f"{diff:+.1f}"
                    elif isinstance(diff, float):
                        diff_display = f"{int(diff):+d}"
                    else:
                        diff_display = f"{diff:+d}"
                    line += f" ({diff_display} vs avg)"

            lines.append(line)

    if len(lines) == 1:
        return "No attribute overview data available."

    return "\n".join(lines)

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

def format_lineup_stats_for_ai(lineup_data, home_team, away_team):
    """
    Formats the detailed player stats from the /lineups endpoint for the AI.
    """
    if not lineup_data:
        return "No detailed player stats available.\n"

    def get_player_stats(players):
        player_lines = []
        for p in players:
            # We include subs this time for a full report
            stats = p.get("statistics", {})
            name = p["player"]["shortName"]
            pos = p["position"]
            rating = stats.get("rating", "N/A")
            minutes = stats.get("minutesPlayed", "N/A")
            passes = f"{stats.get('accuratePass', 'N/A')}/{stats.get('totalPass', 'N/A')}"
            duels_won = stats.get("duelWon", 0) or 0
            duels_lost = stats.get("duelLost", 0) or 0
            total_duels = duels_won + duels_lost
            kilometers = stats.get('kilometersCovered', 'N/A')
            
            player_lines.append(
                f"  - {name} ({pos}, Rating: {rating}, Mins: {minutes}): "
                f"Passes: {passes}, "
                f"Duels Won: {duels_won}/{total_duels}, "
                f"KM Covered: {kilometers}"
            )
        return "\n".join(player_lines)

    home_lines = get_player_stats(lineup_data.get("home", {}).get("players", []))
    away_lines = get_player_stats(lineup_data.get("away", {}).get("players", []))
    
    return (
        f"Detailed Player Statistics ({home_team}):\n{home_lines}\n\n"
        f"Detailed Player Statistics ({away_team}):\n{away_lines}\n"
    )

def format_single_player_stats_for_ai(player_data):
    """
    Formats one player's stats into a clean string for display and AI analysis.
    """
    if not player_data:
        return "Player data not found."
    
    stats = player_data.get("statistics", {})
    player = player_data.get("player", {})
    
    # Helper to safely get stats
    def get_stat(key, default="0"):
        return stats.get(key, default) or default
    
    # Build the stat block
    stat_lines = [
        f"**Player:** {player.get('name', 'N/A')} ({player.get('position', 'N/A')})",
        f"**Rating:** {get_stat('rating', 'N/A')}",
        f"**Minutes Played:** {get_stat('minutesPlayed', 'N/A')}",
        "---",
        "**Attacking:**",
        f"- Goals: {get_stat('goals')}",
        f"- Expected Goals (xG): {get_stat('expectedGoals', '0.00')}",
        f"- Assists: {get_stat('goalAssist')}",
        f"- Expected Assists (xA): {get_stat('expectedAssists', '0.00')}",
        f"- Total Shots: {get_stat('totalShots')}",
        f"- Shots on Target: {get_stat('onTargetScoringAttempt')}",
        f"- Dribbles (Succ.): {get_stat('successfulDribble', '0')}/{get_stat('totalDribble', '0')}",
        "---",
        "**Passing:**",
        f"- Accurate Passes: {get_stat('accuratePass', '0')}/{get_stat('totalPass', '0')} ({get_stat('passAccuracy', '0')}%)",
        f"- Key Passes: {get_stat('keyPass')}",
        f"- Accurate Long Balls: {get_stat('accurateLongBalls', '0')}/{get_stat('totalLongBalls', '0')}",
        "---",
        "**Defending:**",
        f"- Tackles Won: {get_stat('wonTackle', '0')}/{get_stat('totalTackle', '0')}",
        f"- Interceptions: {get_stat('interceptionWon')}",
        f"- Clearances: {get_stat('totalClearance')}",
        f"- Blocks: {get_stat('challengeLost')}", # Note: This mapping might be off, Sofascore JSON is tricky
        "---",
        "**Duels:**",
        f"- Ground Duels Won: {get_stat('duelWon', '0')}/{get_stat('totalGroundDuel', '0')}",
        f"- Aerial Duels Won: {get_stat('aerialWon', '0')}/{get_stat('totalAerialDuel', '0')}",
        f"- Possession Lost: {get_stat('possessionLostCtrl')}",
        "---",
        "**Goalkeeping (if applicable):**",
        f"- Saves: {get_stat('saves')}",
        f"- Goals Prevented (xGOT): {get_stat('goalsPrevented', '0.00')}"
    ]
    
    return "\n".join(stat_lines)

def call_gemini_api(api_key, system_prompt, user_prompt, chat_history=None):
    """
    A generic function to call the Gemini API.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    # Prepare history if provided
    gemini_history = []
    if chat_history:
        for msg in chat_history:
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
    
    # Add the final user prompt to the history
    gemini_history.append({"role": "user", "parts": [{"text": user_prompt}]})

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
    except Exception as e:
        st.error(f"An unexpected error occurred during AI call: {e}", icon="🤖")
        return f"An error occurred: {e}"


@st.cache_data(ttl=600, show_spinner=False)
def get_ai_analysis_summary(api_key, event_data, avg_data, graph_data, stats_data, lineup_data, home_score, away_score):
    """
    Generates the main (cached) AI match summary using all available data.
    """
    try:
        home_team = event_data["event"]["homeTeam"]["name"]
        away_team = event_data["event"]["awayTeam"]["name"]
        
        # Format all our data for the AI
        stats_summary = format_stats_for_ai(stats_data, home_team, away_team)
        player_summary = format_player_data_for_ai(avg_data)
        lineup_summary = format_lineup_stats_for_ai(lineup_data, home_team, away_team)

        # Format attack momentum
        momentum_summary = (
            f"Attack momentum data (positive for {home_team}, negative for {away_team}): "
            + ", ".join([f"{p['minute']}'_({p['value']})" for p in graph_data["graphPoints"][::5]]) # Sample every 5th point
        )

        system_prompt = f"You are a world-class football analyst summarizing a match: {home_team} vs {away_team}."
        user_prompt = f"""
        Act as a professional football analyst. Your task is to provide a concise, insightful, and well-written match report for {home_team} vs {away_team}.
        Do not just list the stats; interpret them to tell the story of the match.
        Use all the data provided, including the detailed player stats, to make specific observations.
        
        Here is the data:
        
        --- FINAL SCORE ---
        {home_team}: {home_score}
        {away_team}: {away_score}
        --- (This is the most important fact, all analysis must reflect this result) ---
        
        --- Match Overview Statistics ---
        {stats_summary}

        --- Detailed Player Statistics ---
        {lineup_summary}
        
        --- Average Player Positions ---
        {player_summary}

        --- Attack Momentum ---
        {momentum_summary}

        Based on all this data, please provide:
        1.  **Match Summary (Headline):** A short, punchy paragraph describing the overall narrative and result of the match, grounded in the final score.
        2.  **Tactical Analysis:** An analysis of the teams' tactics based on the data. Who won and why? How did the average positions, momentum, and detailed player stats support this outcome?
        3.  **Key Players:** Based on the detailed stats (ratings, duels, passes, etc.), name one standout player for the winning team and one key player (good or bad) for the losing team, and explain why.
        4.  **Key Talking Point:** Based on all the data, identify the single most important factor that decided this match (e.g., "Home team's midfield control," "Away team's clinical finishing").
        
        Be professional, insightful, and use engaging language.
        """
        
        # We pass an empty chat history for the main summary
        return call_gemini_api(api_key, system_prompt, user_prompt, chat_history=[])

    except Exception as e:
        st.error(f"Error during AI analysis: {e}", icon="🤖")
        return "AI analysis failed. Could not generate the report."

def get_chatbot_response(api_key, chat_history, match_context):
    """
    Gets a response from the AI chatbot based on the conversation history and all match data.
    """
    # Prepare the data context for the system prompt
    home_team = match_context["event_data"]["event"]["homeTeam"]["name"]
    away_team = match_context["event_data"]["event"]["awayTeam"]["name"]
    home_score = match_context["event_data"]["event"]["homeScore"]["current"]
    away_score = match_context["event_data"]["event"]["awayScore"]["current"]
    
    stats_summary = format_stats_for_ai(match_context["stats_data"], home_team, away_team)
    player_summary = format_player_data_for_ai(match_context["avg_data"])
    lineup_summary = format_lineup_stats_for_ai(match_context["lineup_data"], home_team, away_team)
    
    system_prompt = f"""
    You are a specialist Football Tactical Analyst Chatbot.
    You are analyzing one specific match: {home_team} vs. {away_team}.
    Your entire analysis MUST be based *only* on the data provided below.
    Do NOT invent any data (like scores, goals, or events) not present.
    
    --- FINAL SCORE ---
    {home_team}: {home_score}
    {away_team}: {away_score}
    --- (This is the most important fact) ---

    Here is the complete data for this match:

    --- DATA START ---
    
    --- Match Overview Statistics ---
    {stats_summary}

    --- Detailed Player Statistics ---
    {lineup_summary}
    
    --- Average Player Positions ---
    {player_summary}

    --- DATA END ---

    The user will now ask you questions about this specific match.
    Answer their questions by interpreting the provided data. Focus on player positions, formations, and how they relate to the statistics.
    Be concise, insightful, and directly answer the user's question.
    """

    # call_gemini_api handles the history formatting, so we pass the last message as the "user_prompt"
    # and the rest as "chat_history"
    
    user_prompt = chat_history[-1]["content"]
    history_to_pass = chat_history[:-1] # All *except* the latest user prompt
    
    return call_gemini_api(api_key, system_prompt, user_prompt, chat_history=history_to_pass)


@st.cache_data(ttl=600)
def get_player_match_analysis(
    api_key,
    player_stats_str,
    player_name,
    home_team,
    away_team,
    home_score,
    away_score,
    stats_summary,
    heatmap_summary,
):
    """Generates an AI analysis for a single player's match performance."""

    system_prompt = (
        f"You are a world-class football scout analyzing {player_name}'s performance in "
        f"{home_team} vs. {away_team} (Final Score: {home_score}-{away_score})."
    )

    user_prompt = f"""
    Please provide a concise analysis of {player_name}'s match using the following data.

    --- {player_name}'s Match Stats ---
    {player_stats_str}

    --- Heatmap Context ---
    {heatmap_summary}

    --- Overall Match Context ---
    Final Score: {home_team} {home_score} - {away_score} {away_team}
    {stats_summary}

    Based on this information:
    1. **Overall Performance:** Summarize their influence in 1-2 sentences.
    2. **Strengths:** Highlight the biggest positives in their display.
    3. **Areas to Improve:** Identify weaknesses or tactical limitations seen in this match.

    Stay factual and rely solely on the data above.
    """

    return call_gemini_api(api_key, system_prompt, user_prompt, chat_history=[])


@st.cache_data(ttl=600)
def get_player_season_analysis(
    api_key,
    player_name,
    team_name,
    season_stats_summary,
    attribute_summary,
):
    """Generate an AI analysis summarizing a player's season-long output."""

    system_prompt = (
        f"You are a seasoned technical analyst preparing a seasonal dossier on {player_name} for {team_name}."
    )

    user_prompt = f"""
    Produce a sharp seasonal evaluation of {player_name} using only the data supplied.

    --- Aggregated Season Statistics ---
    {season_stats_summary}

    --- Attribute Overview ---
    {attribute_summary}

    Please provide:
    1. **Season Snapshot:** 2-3 sentences summarizing their consistency and role.
    2. **Key Strengths:** Bullet the standout qualities or metrics that define their season.
    3. **Development Areas:** Suggest one or two areas that require improvement based on the numbers.

    Keep the tone professional and data-grounded.
    """

    return call_gemini_api(api_key, system_prompt, user_prompt, chat_history=[])


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
        st.exception(e) # Print full traceback for debugging
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
        
        base_api_url = f"https://www.sofascore.com/api/v1/event/{event_id}"

        # Clear previous chat history and data
        st.session_state.messages = []
        st.session_state.match_data = None
        st.session_state.player_analysis_cache = {} # Clear player cache
        
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
            with st.spinner("Getting the tea... Fetching detailed lineups..."):
                lineup_data = fetch_json(f"{base_api_url}/lineups")

            if not all([event_data, avg_data, graph_data, stats_data, lineup_data]):
                st.error("Failed to fetch all required match data. The match may be too old, not yet played, or not supported.", icon="🚨")
                return

            # Get score here to pass to summary
            home_score = event_data["event"]["homeScore"]["current"]
            away_score = event_data["event"]["awayScore"]["current"]

            # Store all fetched data in session_state for the chatbot
            st.session_state.match_data = {
                "event_data": event_data,
                "avg_data": avg_data,
                "graph_data": graph_data,
                "stats_data": stats_data,
                "lineup_data": lineup_data,
                "event_id": event_id,
            }
            
            # Generate the main AI summary
            api_key = get_gemini_api_key()
            if api_key:
                with st.spinner("Summoning the AI analyst for the match report..."):
                    summary = get_ai_analysis_summary(
                        api_key, event_data, avg_data, graph_data, stats_data, lineup_data,
                        home_score, away_score # Pass the score
                    )
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
        
        # Add match score to header
        home_score = match_data["event_data"]["event"]["homeScore"]["current"]
        away_score = match_data["event_data"]["event"]["awayScore"]["current"]
        st.header(f"Analysis: {home_team} {home_score} - {away_score} {away_team}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Analyst Report", "📊 Visual Insights", "💬 Tactical Chatbot", "🧑‍🔬 Player Analysis"])

        with tab1:
            st.subheader("AI Match Report")
            summary_text = match_data.get("ai_summary", "No summary available.")
            st.markdown(summary_text)
            
            # Download Button
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
            if prompt := st.chat_input("Ask about a player's rating or the team's formation..."):
                if not api_key:
                    st.error("Chatbot is disabled. Please add your Gemini API key to Streamlit secrets.", icon="🔐")
                else:
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
        
        with tab4:
            st.subheader("Detailed Player Analysis")
            all_players = get_all_players(match_data.get("lineup_data"))
            
            if not all_players:
                st.warning("No player lineup data available for analysis.")
                return

            player_options = [(p["name"], p["id"], p["team"]) for p in all_players]
            
            # Format options for the selectbox: "Player Name (Home/Away)"
            def format_player_option(option):
                name, _id, team = option
                team_label = "Home" if team == "home" else "Away"
                return f"{name} ({team_label})"

            selected_option = st.selectbox(
                "Select a player to analyze:",
                options=player_options,
                format_func=format_player_option,
                index=None,
                placeholder="Choose a player..."
            )

            if selected_option:
                player_name, player_id, player_team = selected_option
                
                # Get the full data for the selected player
                player_data = get_player_by_id(match_data["lineup_data"], player_id, player_team)
                
                if player_data:
                    # Format their stats for display
                    player_stats_str = format_single_player_stats_for_ai(player_data)
                    
                    col1, col2 = st.columns([1, 1.2])
                    
                    with col1:
                        st.markdown(player_stats_str)
                    
                    with col2:
                        mode_options = {
                            "Match Analysis": "match",
                            "Season Analysis": "season",
                        }
                        selected_mode_label = st.radio(
                            "Choose analysis focus",
                            options=list(mode_options.keys()),
                            horizontal=True,
                            key=f"analysis_mode_{player_id}",
                        )
                        selected_mode = mode_options[selected_mode_label]

                        analyze_key = f"analyze_{player_id}_{selected_mode}"
                        analysis_cache_key = f"analysis_{player_id}_{selected_mode}"

                        button_label = (
                            f"Analyze {player_name}'s Match" if selected_mode == "match"
                            else f"Analyze {player_name}'s Season"
                        )

                        if st.button(button_label, key=analyze_key):
                            api_key = get_gemini_api_key()
                            if api_key:
                                with st.spinner(f"Analyzing {player_name} ({selected_mode_label.lower()})..."):
                                    match_ctx = st.session_state.match_data
                                    home_team = match_ctx["event_data"]["event"]["homeTeam"]["name"]
                                    away_team = match_ctx["event_data"]["event"]["awayTeam"]["name"]

                                    cache_payload = {"mode": selected_mode, "analysis": "", "context": {}}

                                    if selected_mode == "match":
                                        home_score = match_ctx["event_data"]["event"]["homeScore"]["current"]
                                        away_score = match_ctx["event_data"]["event"]["awayScore"]["current"]
                                        stats_summary = format_stats_for_ai(
                                            match_ctx["stats_data"], home_team, away_team
                                        )
                                        event_id = match_ctx.get("event_id") or match_ctx["event_data"]["event"].get("id")
                                        heatmap_data = fetch_player_heatmap(event_id, player_id)
                                        heatmap_summary = format_heatmap_summary(heatmap_data)

                                        analysis = get_player_match_analysis(
                                            api_key,
                                            player_stats_str,
                                            player_name,
                                            home_team,
                                            away_team,
                                            home_score,
                                            away_score,
                                            stats_summary,
                                            heatmap_summary,
                                        )

                                        cache_payload["analysis"] = analysis
                                        cache_payload["context"] = {
                                            "heatmap_summary": heatmap_summary,
                                            "stats_summary": stats_summary,
                                        }
                                    else:
                                        team_name = home_team if player_team == "home" else away_team
                                        season_stats = fetch_player_season_statistics(player_id)
                                        season_summary = format_season_stats_for_ai(season_stats)
                                        attribute_data = fetch_player_attribute_overviews(player_id)
                                        attribute_summary = format_attribute_overviews_for_ai(attribute_data)

                                        analysis = get_player_season_analysis(
                                            api_key,
                                            player_name,
                                            team_name,
                                            season_summary,
                                            attribute_summary,
                                        )

                                        cache_payload["analysis"] = analysis
                                        cache_payload["context"] = {
                                            "season_summary": season_summary,
                                            "attribute_summary": attribute_summary,
                                        }

                                    st.session_state.player_analysis_cache[analysis_cache_key] = cache_payload
                            else:
                                st.error("Cannot analyze player. Please add your Gemini API key.")

                        cache_entry = st.session_state.player_analysis_cache.get(analysis_cache_key)
                        if cache_entry:
                            if isinstance(cache_entry, dict):
                                analysis_text = cache_entry.get("analysis", "")
                                context = cache_entry.get("context", {})
                            else:
                                analysis_text = str(cache_entry)
                                context = {}

                            with st.container(border=True):
                                if analysis_text:
                                    st.markdown(analysis_text)
                                else:
                                    st.info("Analysis generated but no textual output was returned.")

                                if selected_mode == "match" and context.get("heatmap_summary"):
                                    st.markdown("**Heatmap summary used in analysis:**")
                                    st.markdown(context["heatmap_summary"])
                                elif selected_mode == "season":
                                    if context.get("season_summary"):
                                        st.markdown("**Season statistics summary used in analysis:**")
                                        st.markdown(context["season_summary"])
                                    if context.get("attribute_summary"):
                                        st.markdown("**Attribute overview included in analysis:**")
                                        st.markdown(context["attribute_summary"])

                                if analysis_text:
                                    st.download_button(
                                        label="📥 Download Player Report",
                                        data=analysis_text,
                                        file_name=f"{player_name.replace(' ', '_').lower()}_{selected_mode}_analysis.txt",
                                        mime="text/plain",
                                        key=f"download_{player_id}_{selected_mode}",
                                    )

                else:
                    st.error("Could not find data for the selected player.")

if __name__ == "__main__":
    main()
