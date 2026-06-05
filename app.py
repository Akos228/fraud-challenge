"""
Interface Streamlit — Détection de fraude INTELO.

Lancement : streamlit run app.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_detection import detect_fraud, load_transactions

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"
SUSPICION_THRESHOLD = 0.50
PREVIEW_ROWS = 10

# Palette fintech professionnelle (noir / bleu)
_COLOR_BG = "#000000"
_COLOR_SURFACE = "#0A1628"
_COLOR_SURFACE_ALT = "#0D1B2E"
_COLOR_BLUE = "#0066FF"
_COLOR_ACCENT = "#1E90FF"
_COLOR_TEXT = "#E8EDF5"
_COLOR_TEXT_MUTED = "#8BA3C7"
_COLOR_ALERT = "#FF4D6A"
_COLOR_OK = "#00C896"
_COLOR_OK_SOFT = "#1E90FF"
_COLOR_HIGHLIGHT = "#FFD166"

MAP_HEIGHT = 650
MAP_HEIGHT_PRESENTATION = 720
MAP_HEIGHT_FULLSCREEN = 680

_ANALYSIS_STEPS = [
    (0.15, "Profilage utilisateurs..."),
    (0.32, "Analyse géographique..."),
    (0.50, "Détection de vélocité..."),
    (0.68, "Scoring des risques..."),
    (0.86, "Finalisation du rapport..."),
]

_COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "FR": (46.23, 2.21),
    "DE": (51.17, 10.45),
    "ES": (40.46, -3.75),
    "IT": (41.87, 12.57),
    "GB": (55.38, -3.44),
    "BE": (50.50, 4.47),
    "NL": (52.13, 5.29),
    "CH": (46.82, 8.23),
    "PT": (39.40, -8.22),
    "AT": (47.52, 14.55),
    "PL": (51.92, 19.15),
    "SE": (60.13, 18.64),
    "US": (37.09, -95.71),
    "CA": (56.13, -106.35),
    "MX": (23.63, -102.55),
    "BR": (-14.24, -51.93),
    "AR": (-38.42, -63.62),
    "CL": (-35.68, -71.54),
    "CO": (4.57, -74.30),
    "JP": (36.20, 138.25),
    "CN": (35.86, 104.20),
    "KR": (35.91, 127.77),
    "IN": (20.59, 78.96),
    "SG": (1.35, 103.82),
    "TH": (15.87, 100.99),
    "AE": (23.42, 53.85),
    "SA": (23.89, 45.08),
    "HK": (22.40, 114.11),
    "TW": (23.70, 120.96),
    "VN": (14.06, 108.28),
    "MY": (4.21, 101.98),
    "AU": (-25.27, 133.78),
    "NZ": (-40.90, 174.89),
    "ZA": (-30.56, 22.94),
    "NG": (9.08, 8.68),
    "EG": (26.82, 30.80),
    "MA": (31.79, -7.09),
    "KE": (-0.02, 37.91),
    "SN": (14.50, -14.45),
    "RU": (61.52, 105.32),
    "TR": (38.96, 35.24),
    "IL": (31.05, 34.85),
}

_ISO2_TO_ISO3: dict[str, str] = {
    "FR": "FRA", "DE": "DEU", "ES": "ESP", "IT": "ITA", "GB": "GBR",
    "BE": "BEL", "NL": "NLD", "CH": "CHE", "PT": "PRT", "AT": "AUT",
    "PL": "POL", "SE": "SWE", "US": "USA", "CA": "CAN", "MX": "MEX",
    "BR": "BRA", "AR": "ARG", "CL": "CHL", "CO": "COL", "JP": "JPN",
    "CN": "CHN", "KR": "KOR", "IN": "IND", "SG": "SGP", "TH": "THA",
    "AE": "ARE", "SA": "SAU", "HK": "HKG", "TW": "TWN", "VN": "VNM",
    "MY": "MYS", "AU": "AUS", "NZ": "NZL", "ZA": "ZAF", "NG": "NGA",
    "EG": "EGY", "MA": "MAR", "KE": "KEN", "SN": "SEN", "RU": "RUS",
    "TR": "TUR", "IL": "ISR",
}

_COUNTRY_CONTINENT: dict[str, str] = {
    "FR": "EU", "DE": "EU", "ES": "EU", "IT": "EU", "GB": "EU", "BE": "EU",
    "NL": "EU", "CH": "EU", "PT": "EU", "AT": "EU", "PL": "EU", "SE": "EU",
    "US": "NA", "CA": "NA", "MX": "NA", "BR": "SA", "AR": "SA", "CL": "SA",
    "CO": "SA", "JP": "AS", "CN": "AS", "KR": "AS", "IN": "AS", "SG": "AS",
    "TH": "AS", "AE": "AS", "SA": "AS", "HK": "AS", "TW": "AS", "VN": "AS",
    "MY": "AS", "AU": "OC", "NZ": "OC", "ZA": "AF", "NG": "AF", "EG": "AF",
    "MA": "AF", "KE": "AF", "SN": "AF", "RU": "EU", "TR": "AS", "IL": "AS",
}

_MIN_TRAVEL_HOURS: dict[tuple[str, str], float] = {
    ("same", "same"): 0.0,
    ("same", "neighbor"): 1.0,
    ("same", "far"): 3.0,
    ("diff", "neighbor"): 6.0,
    ("diff", "far"): 10.0,
}

_COUNTRY_GEO_SCOPE: dict[str, str] = {
    "FR": "europe", "DE": "europe", "ES": "europe", "IT": "europe",
    "GB": "europe", "BE": "europe", "NL": "europe", "CH": "europe",
    "PT": "europe", "AT": "europe", "PL": "europe", "SE": "europe",
    "RU": "europe", "TR": "europe", "IL": "asia",
    "US": "north america", "CA": "north america", "MX": "north america",
    "BR": "south america", "AR": "south america", "CL": "south america",
    "CO": "south america",
    "JP": "asia", "CN": "asia", "KR": "asia", "IN": "asia", "SG": "asia",
    "TH": "asia", "AE": "asia", "SA": "asia", "HK": "asia", "TW": "asia",
    "VN": "asia", "MY": "asia",
    "AU": "australia", "NZ": "australia",
    "ZA": "africa", "NG": "africa", "EG": "africa", "MA": "africa",
    "KE": "africa", "SN": "africa",
}

_COUNTRY_ZOOM_SCALE: dict[str, float] = {
    "US": 2.8, "CA": 2.5, "BR": 2.2, "AU": 2.8, "RU": 1.8,
}

_ALL_COUNTRIES_LABEL = "Tous"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_COLOR_SURFACE,
    font=dict(family="Inter, sans-serif", color=_COLOR_TEXT, size=12),
    margin=dict(l=24, r=24, t=48, b=24),
    legend=dict(
        bgcolor="rgba(11,20,34,0.8)",
        bordercolor="rgba(11,95,255,0.25)",
        borderwidth=1,
    ),
)


def _plotly_layout(**overrides) -> dict:
    """Merge base Plotly theme with per-chart overrides (avoids duplicate kwargs)."""
    return {**_PLOTLY_LAYOUT, **overrides}

_CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {{
        background-color: {_COLOR_BG};
        color: {_COLOR_TEXT};
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 100%;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {_COLOR_SURFACE} 0%, {_COLOR_BG} 100%);
        border-right: 1px solid rgba(11, 95, 255, 0.18);
    }}
    [data-testid="stSidebar"] * {{
        color: {_COLOR_TEXT} !important;
    }}

    h1, h2, h3, h4, p, label, span {{
        color: {_COLOR_TEXT};
    }}

    .prestige-header {{
        padding: 2.5rem 0 2rem 0;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(11, 95, 255, 0.15);
    }}
    .prestige-eyebrow {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: {_COLOR_ACCENT};
        margin-bottom: 0.85rem;
    }}
    .prestige-title {{
        font-size: 2.35rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0 0 0.75rem 0;
        line-height: 1.15;
    }}
    .prestige-subtitle {{
        font-size: 1.05rem;
        color: {_COLOR_TEXT_MUTED};
        margin: 0;
        max-width: 640px;
        line-height: 1.65;
        font-weight: 400;
    }}

    .glass-card {{
        background: rgba(11, 20, 34, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(11, 95, 255, 0.22);
        border-radius: 16px;
        padding: 2rem 2rem 1.75rem 2rem;
        box-shadow: 0 0 48px rgba(11, 95, 255, 0.06), inset 0 1px 0 rgba(255,255,255,0.04);
        height: 100%;
    }}
    .glass-card-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.01em;
    }}
    .glass-card-desc {{
        font-size: 0.9rem;
        color: {_COLOR_TEXT_MUTED};
        margin: 0 0 1.5rem 0;
        line-height: 1.55;
    }}

    .mono {{
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2.5rem;
    }}
    @media (max-width: 900px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-card {{
        background: rgba(11, 20, 34, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(11, 95, 255, 0.18);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 0 32px rgba(11, 95, 255, 0.05);
    }}
    .kpi-label {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {_COLOR_TEXT_MUTED};
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: {_COLOR_TEXT_MUTED};
        margin-top: 0.35rem;
    }}

    .section-label {{
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {_COLOR_ACCENT};
        margin: 2.5rem 0 1rem 0;
    }}
    .section-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }}
    .section-desc {{
        font-size: 0.88rem;
        color: {_COLOR_TEXT_MUTED};
        margin: 0 0 1.25rem 0;
    }}

    .chart-panel {{
        background: rgba(11, 20, 34, 0.5);
        border: 1px solid rgba(11, 95, 255, 0.15);
        border-radius: 14px;
        padding: 0.5rem 0.25rem 0.25rem 0.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 28px rgba(11, 95, 255, 0.04);
    }}

    .premium-table-wrap {{
        overflow-x: auto;
        border: 1px solid rgba(11, 95, 255, 0.18);
        border-radius: 14px;
        background: rgba(11, 20, 34, 0.55);
        box-shadow: 0 0 40px rgba(11, 95, 255, 0.05);
    }}
    .premium-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }}
    .premium-table th {{
        text-align: left;
        padding: 0.85rem 1rem;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {_COLOR_TEXT_MUTED};
        border-bottom: 1px solid rgba(11, 95, 255, 0.2);
        background: rgba(0, 0, 0, 0.25);
    }}
    .premium-table td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(11, 95, 255, 0.08);
        color: {_COLOR_TEXT};
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }}
    .premium-table tr:hover td {{
        background: rgba(11, 95, 255, 0.06);
    }}
    .premium-table tr.alert-row td {{
        background: rgba(255, 77, 106, 0.06);
    }}

    .pill {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .pill-ok {{
        background: rgba(11, 95, 255, 0.15);
        color: {_COLOR_ACCENT};
        border: 1px solid rgba(61, 139, 255, 0.35);
    }}
    .pill-alert {{
        background: rgba(255, 77, 106, 0.12);
        color: #FF8FA3;
        border: 1px solid rgba(255, 77, 106, 0.35);
    }}

    .preview-box {{
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(11, 95, 255, 0.12);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        font-size: 0.75rem;
        color: {_COLOR_TEXT_MUTED};
        overflow-x: auto;
    }}
    .preview-box table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
    }}
    .preview-box th, .preview-box td {{
        padding: 0.3rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid rgba(11, 95, 255, 0.08);
    }}

    .score-meter-wrap {{
        margin: 0.75rem 0 0.5rem 0;
    }}
    .score-meter-track {{
        height: 6px;
        background: rgba(11, 95, 255, 0.12);
        border-radius: 999px;
        overflow: hidden;
    }}
    .score-meter-fill {{
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, {_COLOR_BLUE} 0%, {_COLOR_ALERT} 100%);
    }}
    .score-meter-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: {_COLOR_TEXT_MUTED};
        margin-top: 0.35rem;
    }}

    .alert-reason {{
        font-size: 0.9rem;
        color: {_COLOR_TEXT};
        line-height: 1.55;
        margin-top: 0.5rem;
    }}

    div[data-testid="stFileUploader"] {{
        background: rgba(0, 0, 0, 0.3);
        border: 1px dashed rgba(61, 139, 255, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
    }}
    div[data-testid="stFileUploader"] button {{
        background: {_COLOR_BLUE} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid rgba(11, 95, 255, 0.18) !important;
        border-radius: 12px !important;
        background: rgba(11, 20, 34, 0.5) !important;
        margin-bottom: 0.5rem;
    }}
    div[data-testid="stExpander"] summary {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        color: {_COLOR_TEXT} !important;
    }}

    .stButton > button[kind="primary"] {{
        background: {_COLOR_BLUE} !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(61, 139, 255, 0.4) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 24px rgba(11, 95, 255, 0.25) !important;
        letter-spacing: 0.02em;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {_COLOR_ACCENT} !important;
        box-shadow: 0 0 32px rgba(61, 139, 255, 0.35) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {_COLOR_ACCENT} !important;
        border: 1px solid rgba(61, 139, 255, 0.35) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stRadio"] > div {{
        gap: 0.5rem;
    }}
    div[data-testid="stRadio"] label {{
        background: rgba(11, 20, 34, 0.5);
        border: 1px solid rgba(11, 95, 255, 0.15);
        border-radius: 8px;
        padding: 0.5rem 0.85rem;
        font-size: 0.85rem;
    }}

    hr {{
        border-color: rgba(11, 95, 255, 0.12) !important;
        margin: 2.5rem 0 !important;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    .dashboard-reveal {{
        animation: fadeInDashboard 0.85s ease-out forwards;
    }}
    @keyframes fadeInDashboard {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .kpi-value-animated {{
        animation: kpiReveal 1.1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        opacity: 0;
    }}
    .kpi-card:nth-child(1) .kpi-value-animated {{ animation-delay: 0.05s; }}
    .kpi-card:nth-child(2) .kpi-value-animated {{ animation-delay: 0.15s; }}
    .kpi-card:nth-child(3) .kpi-value-animated {{ animation-delay: 0.25s; }}
    .kpi-card:nth-child(4) .kpi-value-animated {{ animation-delay: 0.35s; }}
    @keyframes kpiReveal {{
        0% {{ opacity: 0; transform: translateY(14px) scale(0.92); }}
        60% {{ opacity: 1; transform: translateY(-2px) scale(1.02); }}
        100% {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    .country-chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 1.75rem 0;
        align-items: center;
    }}
    .country-chips-label {{
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {_COLOR_TEXT_MUTED};
        margin-right: 0.35rem;
    }}

    .alert-pulse-panel {{
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        margin-bottom: 2rem;
    }}
    .alert-pulse-item {{
        background: rgba(11, 20, 34, 0.65);
        border: 1px solid rgba(255, 77, 106, 0.22);
        border-left: 4px solid {_COLOR_ALERT};
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        animation: slideInAlert 0.55s ease-out forwards, pulseBorder 2.2s ease-in-out infinite;
        opacity: 0;
    }}
    .alert-pulse-item:nth-child(1) {{ animation-delay: 0.05s, 0s; }}
    .alert-pulse-item:nth-child(2) {{ animation-delay: 0.12s, 0s; }}
    .alert-pulse-item:nth-child(3) {{ animation-delay: 0.19s, 0s; }}
    .alert-pulse-item:nth-child(4) {{ animation-delay: 0.26s, 0s; }}
    .alert-pulse-item:nth-child(5) {{ animation-delay: 0.33s, 0s; }}
    @keyframes slideInAlert {{
        from {{ opacity: 0; transform: translateX(-22px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes pulseBorder {{
        0%, 100% {{
            border-left-color: {_COLOR_ALERT};
            box-shadow: -3px 0 14px rgba(255, 77, 106, 0.25);
        }}
        50% {{
            border-left-color: #FF8FA3;
            box-shadow: -5px 0 22px rgba(255, 77, 106, 0.55);
        }}
    }}
    .alert-pulse-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #FFFFFF;
        font-weight: 600;
    }}
    .alert-pulse-meta {{
        font-size: 0.78rem;
        color: {_COLOR_TEXT_MUTED};
        margin-top: 0.25rem;
    }}

    .map-legend-bar {{
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
        padding: 0.65rem 1rem;
        margin-bottom: 0.75rem;
        background: rgba(11, 20, 34, 0.55);
        border: 1px solid rgba(11, 95, 255, 0.15);
        border-radius: 10px;
        font-size: 0.78rem;
        color: {_COLOR_TEXT_MUTED};
    }}
    .map-legend-item {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }}
    .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}
    .legend-dot-ok {{ background: {_COLOR_OK_SOFT}; box-shadow: 0 0 8px rgba(30,144,255,0.5); }}
    .legend-dot-alert {{ background: {_COLOR_ALERT}; box-shadow: 0 0 10px rgba(255,77,106,0.6); }}
    .legend-arc {{
        width: 22px;
        height: 0;
        border-top: 2px dashed {_COLOR_ALERT};
        display: inline-block;
    }}

    .map-hero-panel {{
        background: linear-gradient(180deg, rgba(11, 20, 34, 0.9) 0%, rgba(0, 0, 0, 0.5) 100%);
        border: 1px solid rgba(11, 95, 255, 0.28);
        border-radius: 18px;
        padding: 0.75rem 0.5rem 0.35rem 0.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 0 64px rgba(11, 95, 255, 0.14);
    }}

    .detail-panel {{
        background: rgba(11, 20, 34, 0.75);
        border: 1px solid rgba(11, 95, 255, 0.22);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        min-height: 300px;
    }}
    .detail-panel-title {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {_COLOR_ACCENT};
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}
    .detail-tx-id {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        color: #FFFFFF;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .detail-meta {{
        font-size: 0.85rem;
        color: {_COLOR_TEXT_MUTED};
        line-height: 1.7;
        margin-bottom: 0.75rem;
    }}
    .detail-meta strong {{ color: {_COLOR_TEXT}; }}

    @keyframes kpiReveal {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes scoreFill {{
        from {{ width: 0%; }}
    }}
    .kpi-value-animated {{
        animation: kpiReveal 0.6s ease-out;
    }}
    .kpi-card {{
        animation: kpiReveal 0.5s ease-out both;
    }}
    .kpi-card:nth-child(2) {{ animation-delay: 0.07s; }}
    .kpi-card:nth-child(3) {{ animation-delay: 0.14s; }}
    .kpi-card:nth-child(4) {{ animation-delay: 0.21s; }}
    .score-meter-fill {{
        animation: scoreFill 0.85s ease-out;
    }}

    .presentation-mode .prestige-header,
    .presentation-mode .country-chips,
    .presentation-mode hr {{
        display: none !important;
    }}
</style>
"""


def _build_dataframe(transactions: list[dict], results: list[dict]) -> pd.DataFrame:
    """Fusionne transactions et résultats de détection."""
    tx_df = pd.DataFrame(transactions)
    res_df = pd.DataFrame(results)
    if (
        not tx_df.empty
        and not res_df.empty
        and "transaction_id" in tx_df.columns
        and "transaction_id" in res_df.columns
    ):
        df = tx_df.merge(res_df, on="transaction_id", how="left")
    elif not tx_df.empty and not res_df.empty:
        df = pd.concat([tx_df, res_df], axis=1)
    elif not tx_df.empty:
        df = tx_df.copy()
    elif not res_df.empty:
        df = res_df.copy()
    else:
        df = pd.DataFrame()

    if "is_suspicious" not in df.columns:
        df["is_suspicious"] = False
    else:
        df["is_suspicious"] = df["is_suspicious"].fillna(False).astype(bool)

    if "fraud_score" not in df.columns:
        df["fraud_score"] = 0.0
    else:
        df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "timestamp" in df.columns:
        df["parsed_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["parsed_ts"] = pd.NaT

    df["statut"] = df["is_suspicious"].map({True: "ALERTE", False: "CONFORME"})
    if "reason" not in df.columns:
        df["reason"] = ""
    df["reason"] = df["reason"].fillna("").astype(str)
    return df


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "Critique"
    if score >= SUSPICION_THRESHOLD:
        return "Élevé"
    if score >= 0.25:
        return "Modéré"
    return "Faible"


def _format_amount(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f}"


def _continent(country: str | None) -> str | None:
    if not country:
        return None
    return _COUNTRY_CONTINENT.get(str(country).upper())


def _min_travel_hours(country_a: str, country_b: str) -> float:
    if not country_a or not country_b:
        return 0.0
    if country_a.upper() == country_b.upper():
        return _MIN_TRAVEL_HOURS[("same", "same")]
    cont_a = _continent(country_a)
    cont_b = _continent(country_b)
    if cont_a is None or cont_b is None:
        return _MIN_TRAVEL_HOURS[("diff", "far")]
    if cont_a == cont_b:
        return _MIN_TRAVEL_HOURS[("same", "far")]
    return _MIN_TRAVEL_HOURS[("diff", "far")]


def _available_countries(df: pd.DataFrame) -> list[str]:
    if "country" not in df.columns:
        return []
    codes = (
        df["country"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    return sorted(codes)


def _filter_by_country(df: pd.DataFrame, country: str | None) -> pd.DataFrame:
    if not country or country == _ALL_COUNTRIES_LABEL or "country" not in df.columns:
        return df.copy()
    return df[df["country"].astype(str).str.upper().str.strip() == country.upper()].copy()


def _marker_sizes(amounts: pd.Series) -> list[float]:
    valid = amounts.fillna(0).clip(lower=0)
    mx = float(valid.max()) if len(valid) else 0.0
    if mx <= 0:
        return [12.0] * len(amounts)
    return [
        min(max(math.sqrt(float(v) / mx) * 22 + 10, 10), 38) for v in valid
    ]


def _apply_dashboard_filters(
    df: pd.DataFrame,
    countries: list[str],
    users: list[str],
    statut_filter: str,
    amount_range: tuple[float, float],
    timeline_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    view = df.copy()
    if view.empty:
        return view
    if countries and "country" in view.columns:
        view = view[view["country"].astype(str).str.upper().str.strip().isin(countries)]
    if users and "user_id" in view.columns:
        view = view[view["user_id"].astype(str).isin(users)]
    if "is_suspicious" in view.columns:
        if statut_filter == "Alertes uniquement":
            view = view[view["is_suspicious"]]
        elif statut_filter == "Conformes uniquement":
            view = view[~view["is_suspicious"]]
    if "amount" in view.columns:
        lo, hi = amount_range
        amt = view["amount"].fillna(0)
        view = view[(amt >= lo) & (amt <= hi)]
    if timeline_cutoff is not None and "parsed_ts" in view.columns:
        valid_ts = view["parsed_ts"].notna()
        view = view[~valid_ts | (view["parsed_ts"] <= timeline_cutoff)]
    return view


def _prepare_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return pd.DataFrame()
    map_df = df.dropna(subset=["country"]).copy()
    map_df["country"] = map_df["country"].astype(str).str.upper().str.strip()
    map_df = map_df[(map_df["country"] != "") & map_df["country"].isin(_COUNTRY_COORDS)]
    if map_df.empty:
        return map_df
    if "transaction_id" in map_df.columns:
        jitters = map_df["transaction_id"].astype(str).map(
            lambda t: ((hash(t) % 7) - 3) * 0.18
        )
    else:
        jitters = pd.Series(range(len(map_df)), index=map_df.index).map(
            lambda i: ((int(i) % 7) - 3) * 0.18
        )
    map_df["lat"] = map_df["country"].map(lambda c: _COUNTRY_COORDS[c][0]) + jitters
    map_df["lon"] = (
        map_df["country"].map(lambda c: _COUNTRY_COORDS[c][1]) + jitters * 1.4
    )
    map_df["iso3"] = map_df["country"].map(_ISO2_TO_ISO3)
    return map_df


def _country_stats(map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return pd.DataFrame()
    rows = []
    for country, grp in map_df.groupby("country"):
        iso3 = _ISO2_TO_ISO3.get(country)
        if not iso3:
            continue
        alert_count = int(grp["is_suspicious"].sum()) if "is_suspicious" in grp.columns else 0
        tx_count = len(grp)
        total_amount = grp["amount"].sum() if "amount" in grp.columns else 0.0
        merchants = grp["merchant"].dropna().astype(str)
        top_merchant = (
            merchants.value_counts().index[0] if not merchants.empty else "—"
        )
        rows.append(
            {
                "country": country,
                "iso3": iso3,
                "tx_count": tx_count,
                "alert_count": alert_count,
                "alert_density": alert_count / tx_count if tx_count else 0.0,
                "total_amount": total_amount,
                "top_merchant": top_merchant,
            }
        )
    return pd.DataFrame(rows)


def _find_geo_fraud_arcs(
    map_df: pd.DataFrame, user_filter: str | None = None
) -> list[dict]:
    arcs: list[dict] = []
    if map_df.empty or "user_id" not in map_df.columns:
        return arcs
    work = map_df.dropna(subset=["parsed_ts", "country", "user_id"]).sort_values(
        "parsed_ts"
    )
    for user_id, group in work.groupby("user_id"):
        if user_filter and str(user_id) != user_filter:
            continue
        prev_row = None
        for _, row in group.reset_index(drop=True).iterrows():
            if prev_row is not None:
                c1 = str(prev_row["country"]).upper()
                c2 = str(row["country"]).upper()
                if c1 != c2 and c1 in _COUNTRY_COORDS and c2 in _COUNTRY_COORDS:
                    hours = (
                        row["parsed_ts"] - prev_row["parsed_ts"]
                    ).total_seconds() / 3600.0
                    if hours < _min_travel_hours(c1, c2):
                        arcs.append(
                            {
                                "user_id": row["user_id"],
                                "from_country": c1,
                                "to_country": c2,
                                "from_lat": _COUNTRY_COORDS[c1][0],
                                "from_lon": _COUNTRY_COORDS[c1][1],
                                "to_lat": _COUNTRY_COORDS[c2][0],
                                "to_lon": _COUNTRY_COORDS[c2][1],
                                "hours": round(hours, 2),
                                "min_hours": _min_travel_hours(c1, c2),
                            }
                        )
            prev_row = row
    return arcs


def _great_circle_arc(
    lat1: float, lon1: float, lat2: float, lon2: float, n_points: int = 24
) -> tuple[list[float], list[float]]:
    lats, lons = [], []
    for i in range(n_points + 1):
        t = i / n_points
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        arc_lift = math.sin(math.pi * t) * 8.0
        lats.append(lat + arc_lift)
        lons.append(lon)
    return lats, lons


def _build_world_map_figure(
    df: pd.DataFrame,
    selected_country: str | None = None,
    height: int = 650,
    highlight_tx_id: str | None = None,
    journey_user_id: str | None = None,
) -> go.Figure:
    map_df = _prepare_map_df(df)
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(
            **_plotly_layout(
                height=height,
                annotations=[
                    dict(
                        text="Aucune coordonnée disponible",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color=_COLOR_TEXT_MUTED),
                    )
                ],
            )
        )
        return fig

    stats_df = _country_stats(map_df)
    if stats_df.empty:
        fig = go.Figure()
        fig.update_layout(
            **_plotly_layout(
                height=height,
                annotations=[
                    dict(
                        text="Aucune statistique pays disponible",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color=_COLOR_TEXT_MUTED),
                    )
                ],
            )
        )
        return fig

    has_selection = bool(
        selected_country and selected_country != _ALL_COUNTRIES_LABEL
    )

    fig = go.Figure()

    if has_selection:
        dim_df = stats_df.copy()
        dim_df["display_z"] = dim_df["alert_density"] * 0.25
        fig.add_trace(
            go.Choropleth(
                locations=dim_df["iso3"],
                z=dim_df["display_z"],
                locationmode="ISO-3",
                colorscale=[[0, "#1a2332"], [1, "#2a3548"]],
                showscale=False,
                marker_line_color="rgba(11,95,255,0.12)",
                marker_line_width=0.5,
                hoverinfo="skip",
            )
        )
        sel = stats_df[stats_df["country"] == selected_country.upper()]
        if not sel.empty:
            row = sel.iloc[0]
            fig.add_trace(
                go.Choropleth(
                    locations=[row["iso3"]],
                    z=[1.0],
                    locationmode="ISO-3",
                    colorscale=[[0, _COLOR_BLUE], [1, _COLOR_ACCENT]],
                    showscale=False,
                    marker_line_color="#FFFFFF",
                    marker_line_width=1.5,
                    hovertemplate=(
                        f"<b>{row['country']}</b> (sélectionné)<br>"
                        "Transactions: %{customdata[0]}<br>"
                        "Alertes: %{customdata[1]}<br>"
                        "Montant total: %{customdata[2]:,.2f}<br>"
                        "Top marchand: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=[[
                        row["tx_count"],
                        row["alert_count"],
                        row["total_amount"],
                        row["top_merchant"],
                    ]],
                )
            )
    else:
        fig.add_trace(
            go.Choropleth(
                locations=stats_df["iso3"],
                z=stats_df["alert_density"],
                locationmode="ISO-3",
                colorscale=[
                    [0.0, "#0D1B2E"],
                    [0.25, "#1a3a6b"],
                    [0.5, "#0066FF"],
                    [0.75, "#FF6B8A"],
                    [1.0, "#FF4D6A"],
                ],
                colorbar=dict(
                    title=dict(
                        text="Densité alertes",
                        font=dict(color=_COLOR_TEXT_MUTED, size=11),
                    ),
                    tickfont=dict(color=_COLOR_TEXT_MUTED, size=10),
                    bgcolor="rgba(11,20,34,0.8)",
                    bordercolor="rgba(11,95,255,0.2)",
                    len=0.55,
                ),
                marker_line_color="rgba(11,95,255,0.25)",
                marker_line_width=0.6,
                customdata=stats_df[
                    ["country", "tx_count", "alert_count", "total_amount", "top_merchant"]
                ].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Transactions: %{customdata[1]}<br>"
                    "Alertes: %{customdata[2]}<br>"
                    "Montant total: %{customdata[3]:,.2f}<br>"
                    "Top marchand: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    arc_user = journey_user_id if highlight_tx_id else None
    arcs = _find_geo_fraud_arcs(map_df, user_filter=arc_user)
    arc_added = False
    for arc in arcs:
        lats, lons = _great_circle_arc(
            arc["from_lat"], arc["from_lon"], arc["to_lat"], arc["to_lon"]
        )
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(color=_COLOR_ALERT, width=2.5, dash="dash"),
                name="Fraude géographique" if not arc_added else None,
                legendgroup="geo_fraud",
                showlegend=not arc_added,
                hovertemplate=(
                    f"<b>Fraude géographique</b><br>"
                    f"Client: {arc['user_id']}<br>"
                    f"{arc['from_country']} → {arc['to_country']}<br>"
                    f"Délai: {arc['hours']}h (min. {arc['min_hours']}h)"
                    "<extra></extra>"
                ),
            )
        )
        arc_added = True

    if "is_suspicious" in map_df.columns:
        conforme = map_df[~map_df["is_suspicious"]]
        alerte = map_df[map_df["is_suspicious"]]
    else:
        conforme = map_df.copy()
        alerte = map_df.iloc[0:0].copy()

    conforme_sizes = _marker_sizes(conforme["amount"]) if not conforme.empty else []
    if not conforme.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=conforme["lat"],
                lon=conforme["lon"],
                mode="markers",
                name="Conforme",
                legendgroup="conforme",
                marker=dict(
                    size=conforme_sizes,
                    color=_COLOR_OK_SOFT,
                    opacity=0.92,
                    line=dict(width=1, color="rgba(255,255,255,0.4)"),
                ),
                customdata=conforme[
                    ["transaction_id", "user_id", "country", "amount", "merchant", "fraud_score", "statut"]
                ].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Client: %{customdata[1]}<br>"
                    "Pays: %{customdata[2]}<br>"
                    "Montant: %{customdata[3]:,.2f}<br>"
                    "Marchand: %{customdata[4]}<br>"
                    "Score: %{customdata[5]:.2f}<extra></extra>"
                ),
            )
        )

    alert_hover = (
        "<b>%{customdata[0]}</b><br>"
        "Client: %{customdata[1]}<br>"
        "Pays: %{customdata[2]}<br>"
        "Montant: %{customdata[3]:,.2f}<br>"
        "Marchand: %{customdata[4]}<br>"
        "Score: %{customdata[5]:.2f}<br>"
        "%{customdata[6]}<extra></extra>"
    )
    alert_custom = alerte[
        ["transaction_id", "user_id", "country", "amount", "merchant", "fraud_score", "reason", "statut"]
    ].values

    alerte_sizes = _marker_sizes(alerte["amount"]) if not alerte.empty else []
    if not alerte.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=alerte["lat"],
                lon=alerte["lon"],
                mode="markers",
                name="Alerte",
                legendgroup="alerte",
                marker=dict(
                    size=[s * 1.2 for s in alerte_sizes],
                    color=_COLOR_ALERT,
                    opacity=0.95,
                    line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
                ),
                customdata=alert_custom,
                hovertemplate=alert_hover,
            )
        )

    if highlight_tx_id and "transaction_id" in map_df.columns:
        hi = map_df[map_df["transaction_id"].astype(str) == str(highlight_tx_id)]
        if not hi.empty:
            row = hi.iloc[0]
            hi_size = _marker_sizes(hi["amount"])[0] * 1.7
            fig.add_trace(
                go.Scattergeo(
                    lat=[row["lat"]],
                    lon=[row["lon"]],
                    mode="markers+text",
                    text=[str(highlight_tx_id)],
                    textposition="top center",
                    textfont=dict(color=_COLOR_HIGHLIGHT, size=11),
                    name="Sélection",
                    hoverinfo="skip",
                    marker=dict(
                        size=hi_size,
                        color="rgba(255,209,102,0.2)",
                        line=dict(width=3, color=_COLOR_HIGHLIGHT),
                    ),
                )
            )

    geo_kwargs: dict = dict(
        bgcolor="rgba(0,0,0,0)",
        landcolor="#111827",
        oceancolor=_COLOR_SURFACE,
        lakecolor=_COLOR_SURFACE,
        showland=True,
        showcountries=True,
        countrycolor="rgba(11,95,255,0.18)",
        coastlinecolor="rgba(11,95,255,0.25)",
        projection_type="natural earth",
    )

    if highlight_tx_id and "transaction_id" in map_df.columns:
        hi = map_df[map_df["transaction_id"].astype(str) == str(highlight_tx_id)]
        if not hi.empty:
            geo_kwargs["scope"] = "world"
            geo_kwargs["center"] = dict(lat=hi.iloc[0]["lat"], lon=hi.iloc[0]["lon"])
            geo_kwargs["projection_scale"] = 3.2
    elif has_selection and selected_country in _COUNTRY_COORDS:
        scope = _COUNTRY_GEO_SCOPE.get(selected_country, "world")
        lat, lon = _COUNTRY_COORDS[selected_country]
        zoom = _COUNTRY_ZOOM_SCALE.get(selected_country, 3.5)
        geo_kwargs["scope"] = scope
        geo_kwargs["center"] = dict(lat=lat, lon=lon)
        geo_kwargs["projection_scale"] = zoom
    else:
        geo_kwargs["scope"] = "world"

    fig.update_geos(**geo_kwargs)

    layout_updates: dict = _plotly_layout(
        height=height,
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            domain=dict(x=[0, 1], y=[0, 1]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(11,20,34,0.85)",
            bordercolor="rgba(11,95,255,0.25)",
            borderwidth=1,
            font=dict(size=11, color=_COLOR_TEXT),
        ),
    )

    fig.update_layout(**layout_updates)
    return fig


def _init_session_state() -> None:
    """Pré-charge les données d'exemple au premier lancement."""
    defaults = {
        "transactions": load_transactions(str(SAMPLE_CSV)),
        "data_source": "sample",
        "results": None,
        "analyzed": False,
        "dashboard_reveal": False,
        "presentation_mode": False,
        "selected_tx_id": None,
        "active_country": _ALL_COUNTRIES_LABEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_sample_preview() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV).head(PREVIEW_ROWS)


def _render_prestige_header(subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="prestige-header">
            <div class="prestige-eyebrow">Plateforme d'intelligence transactionnelle</div>
            <h1 class="prestige-title">INTELO · Fraud Intelligence</h1>
            <p class="prestige-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_cards(df: pd.DataFrame, animated: bool = False) -> None:
    total = len(df)
    suspicious = int(df["is_suspicious"].sum()) if "is_suspicious" in df.columns else 0
    safe = total - suspicious
    rate = (suspicious / total * 100) if total else 0.0
    avg_score = float(df["fraud_score"].mean()) if total and "fraud_score" in df.columns else 0.0

    value_cls = "kpi-value kpi-value-animated" if animated else "kpi-value"
    cards = [
        ("Transactions analysées", str(total), "Lot complet traité"),
        ("Alertes détectées", str(suspicious), "Priorité investigation"),
        ("Transactions conformes", str(safe), "Profil cohérent"),
        ("Taux d'alerte", f"{rate:.1f} %", f"Score moyen {avg_score:.2f}"),
    ]
    html = '<div class="kpi-grid">'
    for label, value, sub in cards:
        html += f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="{value_cls}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_risk_gauge(df: pd.DataFrame, score_override: float | None = None) -> None:
    if score_override is not None:
        avg_score = score_override
        title = f"Risque transaction · {_risk_level(score_override)}"
    else:
        avg_score = float(df["fraud_score"].mean()) if len(df) else 0.0
        title = "Risque portefeuille"
    pct = round(avg_score * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix=" %", font=dict(size=28, color="#FFFFFF")),
            title=dict(
                text=title,
                font=dict(size=14, color=_COLOR_TEXT_MUTED),
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=_COLOR_TEXT_MUTED, dtick=25),
                bar=dict(color=_COLOR_BLUE),
                bgcolor="rgba(11,20,34,0.6)",
                bordercolor="rgba(11,95,255,0.3)",
                steps=[
                    dict(range=[0, 25], color="rgba(0,200,150,0.15)"),
                    dict(range=[25, 50], color="rgba(30,144,255,0.15)"),
                    dict(range=[50, 75], color="rgba(255,140,100,0.2)"),
                    dict(range=[75, 100], color="rgba(255,77,106,0.25)"),
                ],
                threshold=dict(
                    line=dict(color=_COLOR_ALERT, width=3),
                    thickness=0.8,
                    value=SUSPICION_THRESHOLD * 100,
                ),
            ),
        )
    )
    fig.update_layout(
        **_plotly_layout(height=220, margin=dict(l=30, r=30, t=50, b=20))
    )
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_fraud_heatmap_timeline(df: pd.DataFrame) -> None:
    if df.empty or "parsed_ts" not in df.columns or "country" not in df.columns:
        return
    work = df.dropna(subset=["parsed_ts", "country"]).copy()
    if work.empty:
        return
    work["country"] = work["country"].astype(str).str.upper().str.strip()
    work["hour"] = work["parsed_ts"].dt.hour
    work["is_alert_num"] = (
        work["is_suspicious"].astype(int) if "is_suspicious" in work.columns else 0
    )

    pivot = (
        work.groupby(["hour", "country"])["is_alert_num"]
        .mean()
        .reset_index()
        .pivot_table(index="country", columns="hour", values="is_alert_num", fill_value=0.0)
    )
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0.0
    pivot = pivot[sorted(pivot.columns)]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}h" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#0D1B2E"],
                [0.3, "#1a3a6b"],
                [0.6, "#FF6B8A"],
                [1.0, "#FF4D6A"],
            ],
            colorbar=dict(
                title=dict(
                    text="Densité",
                    font=dict(color=_COLOR_TEXT_MUTED, size=11),
                ),
                tickfont=dict(color=_COLOR_TEXT_MUTED, size=10),
            ),
            hovertemplate=(
                "Pays: %{y}<br>Heure: %{x}<br>Densité alertes: %{z:.0%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_plotly_layout(
            height=280,
            xaxis_title="Heure (UTC)",
            yaxis_title="Pays",
        )
    )
    st.markdown(
        '<p class="section-desc" style="margin-top:0">'
        "Carte de chaleur — heures × pays (densité d'alertes)</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_alert_pulse_panel(df: pd.DataFrame) -> None:
    if df.empty or "is_suspicious" not in df.columns:
        return
    suspicious = df[df["is_suspicious"]].sort_values("fraud_score", ascending=False)
    if suspicious.empty:
        return
    st.markdown('<div class="section-label">Signaux actifs</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Alertes en temps réel</p>'
        '<p class="section-desc">Transactions suspectes détectées — priorité investigation.</p>',
        unsafe_allow_html=True,
    )
    items = []
    for _, row in suspicious.head(5).iterrows():
        tid = row.get("transaction_id", "—")
        score = float(row.get("fraud_score", 0))
        country = row.get("country") or "—"
        amount = _format_amount(row.get("amount"))
        reason = str(row.get("reason", ""))[:80]
        items.append(
            f"""
            <div class="alert-pulse-item">
                <div class="alert-pulse-title">{tid} · {score:.0%} · {country}</div>
                <div class="alert-pulse-meta">{amount} — {reason}</div>
            </div>
            """
        )
    st.markdown(f'<div class="alert-pulse-panel">{"".join(items)}</div>', unsafe_allow_html=True)


def _render_country_chips(countries: list[str]) -> None:
    selected = st.session_state.get("active_country", _ALL_COUNTRIES_LABEL)
    chip_cols = st.columns(min(len(countries) + 1, 8))
    options = [_ALL_COUNTRIES_LABEL] + countries[:7]
    for i, code in enumerate(options):
        with chip_cols[i]:
            label = "Tous" if code == _ALL_COUNTRIES_LABEL else code
            btn_type = "primary" if selected == code else "secondary"
            if st.button(label, key=f"chip_{code}", type=btn_type, use_container_width=True):
                st.session_state["active_country"] = code
                st.rerun()


def _render_map_legend() -> None:
    st.markdown(
        """
        <div class="map-legend-bar">
            <div class="map-legend-item">
                <span class="legend-dot legend-dot-ok"></span> Conforme
            </div>
            <div class="map-legend-item">
                <span class="legend-dot legend-dot-alert"></span> Alerte
            </div>
            <div class="map-legend-item">
                <span class="legend-arc"></span> Arc rouge = fraude géographique
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_transaction_detail_panel(row: pd.Series | None) -> None:
    st.markdown('<div class="detail-panel">', unsafe_allow_html=True)
    st.markdown('<div class="detail-panel-title">Transaction sélectionnée</div>', unsafe_allow_html=True)
    if row is None:
        st.markdown(
            '<p class="detail-meta">Cliquez un point sur la carte ou choisissez une '
            "transaction dans la barre latérale.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return
    tid = row.get("transaction_id", "—")
    score = float(row.get("fraud_score", 0))
    is_alert = bool(row.get("is_suspicious"))
    pill = "pill-alert" if is_alert else "pill-ok"
    pill_text = "ALERTE" if is_alert else "CONFORME"
    st.markdown(f'<div class="detail-tx-id">{tid}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pill {pill}">{pill_text}</span>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail-meta">
            <strong>Client</strong> {row.get("user_id") or "—"}<br>
            <strong>Montant</strong> {_format_amount(row.get("amount"))} {row.get("currency") or ""}<br>
            <strong>Pays</strong> {row.get("country") or "—"}<br>
            <strong>Commerçant</strong> {row.get("merchant") or "—"}<br>
            <strong>Horodatage</strong> {row.get("timestamp") or "—"}<br>
            <strong>Score</strong> {score:.0%} · {_risk_level(score)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="alert-reason">{row.get("reason", "")}</div>', unsafe_allow_html=True)
    st.markdown(_score_meter_html(score), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_world_map(
    df: pd.DataFrame,
    selected_country: str | None = None,
    height: int = 650,
    key: str = "world_map",
    highlight_tx_id: str | None = None,
    journey_user_id: str | None = None,
) -> None:
    fig = _build_world_map_figure(
        df,
        selected_country=selected_country,
        height=height,
        highlight_tx_id=highlight_tx_id,
        journey_user_id=journey_user_id,
    )
    _render_map_legend()
    st.markdown('<div class="map-hero-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.markdown("</div>", unsafe_allow_html=True)


def _apply_plotly_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(**_plotly_layout(height=height))
    fig.update_xaxes(
        gridcolor="rgba(11,95,255,0.1)",
        linecolor="rgba(11,95,255,0.2)",
        zerolinecolor="rgba(11,95,255,0.1)",
    )
    fig.update_yaxes(
        gridcolor="rgba(11,95,255,0.1)",
        linecolor="rgba(11,95,255,0.2)",
        zerolinecolor="rgba(11,95,255,0.1)",
    )
    return fig


def _render_donut_chart(df: pd.DataFrame) -> None:
    if df.empty or "statut" not in df.columns:
        st.caption("Aucune donnée pour la répartition des statuts.")
        return
    counts = df["statut"].value_counts().reindex(["CONFORME", "ALERTE"], fill_value=0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.62,
                marker=dict(colors=[_COLOR_OK, _COLOR_ALERT]),
                textinfo="label+percent",
                textfont=dict(color=_COLOR_TEXT),
                hovertemplate="<b>%{label}</b><br>%{value} transactions<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_plotly_layout(
            height=300,
            showlegend=False,
            annotations=[
                dict(
                    text=f"{len(df)}",
                    x=0.5,
                    y=0.5,
                    font=dict(size=22, color="#FFFFFF", family="JetBrains Mono"),
                    showarrow=False,
                )
            ],
        )
    )
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_score_bars(df: pd.DataFrame) -> None:
    if df.empty or "fraud_score" not in df.columns:
        st.caption("Aucune donnée pour la distribution des scores.")
        return
    hist_df = df.copy()
    hist_df["tranche"] = pd.cut(
        hist_df["fraud_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["0–25 %", "25–50 %", "50–75 %", "75–100 %"],
        include_lowest=True,
    )
    counts = (
        hist_df["tranche"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"tranche": "Tranche", "count": "Transactions"})
    )
    fig = px.bar(
        counts,
        x="Tranche",
        y="Transactions",
        color_discrete_sequence=[_COLOR_BLUE],
    )
    threshold_x = 1.5
    fig.add_shape(
        type="line",
        x0=threshold_x,
        x1=threshold_x,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=_COLOR_ALERT, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=threshold_x,
        y=1,
        yref="paper",
        text=f"Seuil {SUSPICION_THRESHOLD:.0%}",
        showarrow=False,
        font=dict(color=_COLOR_TEXT_MUTED, size=11),
        yanchor="bottom",
    )
    fig = _apply_plotly_theme(fig, height=300)
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_country_bars(df: pd.DataFrame) -> None:
    if df.empty or "country" not in df.columns or "statut" not in df.columns:
        st.caption("Aucun pays renseigné.")
        return
    country_df = (
        df.assign(country=df["country"].fillna("Inconnu").astype(str))
        .groupby(["country", "statut"])
        .size()
        .reset_index(name="nombre")
    )
    if country_df.empty:
        st.caption("Aucun pays renseigné.")
        return
    color_map = {"CONFORME": _COLOR_OK, "ALERTE": _COLOR_ALERT}
    fig = px.bar(
        country_df,
        x="nombre",
        y="country",
        color="statut",
        orientation="h",
        color_discrete_map=color_map,
        category_orders={"statut": ["CONFORME", "ALERTE"]},
    )
    fig = _apply_plotly_theme(fig, height=300)
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_timeline(df: pd.DataFrame) -> None:
    if df.empty or "parsed_ts" not in df.columns:
        return
    timeline_df = df.dropna(subset=["parsed_ts"]).sort_values("parsed_ts")
    if timeline_df.empty:
        return

    st.markdown('<div class="section-label">Temporalité</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Chronologie des transactions</p>'
        '<p class="section-desc">Évolution temporelle des montants et signaux de risque.</p>',
        unsafe_allow_html=True,
    )

    color_map = {"CONFORME": _COLOR_OK, "ALERTE": _COLOR_ALERT}
    fig = px.scatter(
        timeline_df,
        x="parsed_ts",
        y="amount",
        color="statut",
        color_discrete_map=color_map,
        hover_data=["transaction_id", "fraud_score", "country", "reason"],
        category_orders={"statut": ["CONFORME", "ALERTE"]},
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig = _apply_plotly_theme(fig, height=340)
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_charts(df: pd.DataFrame, selected_country: str | None = None) -> None:
    st.markdown('<div class="section-label">Analytique</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Visualisations interactives</p>',
        unsafe_allow_html=True,
    )

    _render_fraud_heatmap_timeline(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<p class="section-desc" style="margin-top:0">Répartition conforme / alerte</p>',
            unsafe_allow_html=True,
        )
        _render_donut_chart(df)
    with c2:
        st.markdown(
            '<p class="section-desc" style="margin-top:0">Distribution des scores de risque</p>',
            unsafe_allow_html=True,
        )
        _render_score_bars(df)

    st.markdown(
        '<p class="section-desc">Volume par pays et statut</p>',
        unsafe_allow_html=True,
    )
    _render_country_bars(df)
    _render_timeline(df)


def _render_premium_table(view_df: pd.DataFrame) -> None:
    display_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "user_id",
            "amount",
            "currency",
            "merchant",
            "country",
            "fraud_score",
            "statut",
            "reason",
        ]
        if c in view_df.columns
    ]

    rows_html = []
    for _, row in view_df.iterrows():
        is_alert = bool(row.get("is_suspicious"))
        row_class = "alert-row" if is_alert else ""
        pill_class = "pill-alert" if is_alert else "pill-ok"
        pill_text = "ALERTE" if is_alert else "CONFORME"
        cells = []
        for col in display_cols:
            if col == "statut":
                cells.append(f'<td><span class="pill {pill_class}">{pill_text}</span></td>')
            elif col == "amount":
                cells.append(f"<td>{_format_amount(row.get(col))}</td>")
            elif col == "fraud_score":
                score = row.get(col, 0)
                cells.append(f"<td>{float(score):.2f}</td>" if pd.notna(score) else "<td>—</td>")
            else:
                val = row.get(col)
                cells.append(f"<td>{val if pd.notna(val) and val != '' else '—'}</td>")
        rows_html.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')

    headers = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in display_cols)
    table_html = f"""
    <div class="premium-table-wrap">
        <table class="premium-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{"".join(rows_html)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _score_meter_html(score: float) -> str:
    pct = min(max(score, 0.0), 1.0) * 100
    return f"""
    <div class="score-meter-wrap">
        <div class="score-meter-track">
            <div class="score-meter-fill" style="width: {pct:.1f}%;"></div>
        </div>
        <div class="score-meter-label">Indice de risque : {score:.0%} · Seuil {SUSPICION_THRESHOLD:.0%}</div>
    </div>
    """


def _render_suspicious_details(df: pd.DataFrame) -> None:
    if df.empty or "is_suspicious" not in df.columns:
        st.success("Aucune alerte sur ce lot. Toutes les transactions sont conformes.")
        return
    suspicious_df = df[df["is_suspicious"]].sort_values("fraud_score", ascending=False)
    if suspicious_df.empty:
        st.success("Aucune alerte sur ce lot. Toutes les transactions sont conformes.")
        return

    st.markdown('<div class="section-label">Investigation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-title">Détail des alertes ({len(suspicious_df)})</p>'
        '<p class="section-desc">Analyse transaction par transaction avec score et motif de déclenchement.</p>',
        unsafe_allow_html=True,
    )

    for _, row in suspicious_df.iterrows():
        tid = row.get("transaction_id", "—")
        score = float(row.get("fraud_score", 0))
        reason = row.get("reason", "Raison non précisée")
        amount_str = _format_amount(row.get("amount"))
        country = row.get("country") or "—"
        user = row.get("user_id") or "—"
        ts = row.get("timestamp") or "—"
        merchant = row.get("merchant") or "—"
        level = _risk_level(score)
        title = f"{tid}  ·  {score:.0%}  ·  {level}"

        with st.expander(title, expanded=score >= 0.75):
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"**Client**<br><span class='mono'>{user}</span>", unsafe_allow_html=True)
            m2.markdown(f"**Montant**<br><span class='mono'>{amount_str}</span>", unsafe_allow_html=True)
            m3.markdown(f"**Pays**<br><span class='mono'>{country}</span>", unsafe_allow_html=True)

            st.markdown(
                f"<span class='mono'>{merchant}</span> · <span class='mono'>{ts}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="pill pill-alert">ALERTE</span>'
                f'<div class="alert-reason">{reason}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_score_meter_html(score), unsafe_allow_html=True)


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    """Interface de visualisation des résultats pour le jury."""
    df = _build_dataframe(transactions, results)
    countries = _available_countries(df)
    all_users = (
        sorted(df["user_id"].dropna().astype(str).unique())
        if "user_id" in df.columns
        else []
    )
    amt_valid = df["amount"].dropna() if "amount" in df.columns else pd.Series(dtype=float)
    amt_min = float(amt_valid.min()) if not amt_valid.empty else 0.0
    amt_max = float(amt_valid.max()) if not amt_valid.empty else 10000.0
    if amt_max <= amt_min:
        amt_max = amt_min + 1.0

    ts_sorted = (
        df.dropna(subset=["parsed_ts"]).sort_values("parsed_ts")
        if "parsed_ts" in df.columns
        else pd.DataFrame()
    )
    timeline_labels: list[str] = []
    timeline_values: list[pd.Timestamp] = []
    for _, row in ts_sorted.iterrows():
        timeline_labels.append(str(row["transaction_id"]))
        timeline_values.append(row["parsed_ts"])

    presentation_mode = st.session_state.get("presentation_mode", False)
    reveal = st.session_state.get("dashboard_reveal", False)
    wrapper_cls = "presentation-mode" if presentation_mode else ""
    if reveal:
        wrapper_cls += " dashboard-reveal"
    st.markdown(f'<div class="{wrapper_cls.strip()}">', unsafe_allow_html=True)

    if presentation_mode:
        st.markdown(
            "<style>.block-container{max-width:100%!important;padding-top:0.5rem!important;}</style>",
            unsafe_allow_html=True,
        )

    _render_prestige_header(
        "Analyse comportementale en temps réel — montants, géographie, vélocité et authentification."
    )

    with st.sidebar:
        st.markdown("### Filtres & contrôles")
        st.session_state["presentation_mode"] = st.toggle(
            "Mode présentation",
            value=presentation_mode,
            help="Carte plein écran, interface épurée pour la démo jury.",
        )
        presentation_mode = st.session_state["presentation_mode"]
        st.markdown(
            f"<span class='mono'>Seuil d'alerte : {SUSPICION_THRESHOLD:.0%}</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        country_focus_options = [_ALL_COUNTRIES_LABEL] + countries
        current_focus = st.session_state.get("active_country", _ALL_COUNTRIES_LABEL)
        if current_focus not in country_focus_options:
            current_focus = _ALL_COUNTRIES_LABEL
            st.session_state["active_country"] = current_focus
        focus_choice = st.selectbox(
            "Focus pays (carte + données)",
            options=country_focus_options,
            index=country_focus_options.index(current_focus),
            help="Zoom carte et filtre synchronisé sur le pays sélectionné.",
        )
        if focus_choice != st.session_state.get("active_country"):
            st.session_state["active_country"] = focus_choice
            st.rerun()
        if "sidebar_countries_filter" not in st.session_state:
            st.session_state["sidebar_countries_filter"] = countries
        selected_countries = st.multiselect(
            "Pays (filtre avancé)",
            options=countries,
            placeholder="Tous les pays",
            help="Affinez sur plusieurs pays. Le focus ci-dessus prime si un seul pays est ciblé.",
            key="sidebar_countries_filter",
        )
        selected_users = st.multiselect(
            "Client (user_id)",
            options=all_users,
            default=[],
            placeholder="Tous les clients",
            key="sidebar_users_filter",
        )
        statut_filter = st.radio(
            "Statut",
            options=["Toutes", "Alertes uniquement", "Conformes uniquement"],
            horizontal=True,
            key="sidebar_statut_filter",
        )
        if "sidebar_amount_range" not in st.session_state:
            st.session_state["sidebar_amount_range"] = (amt_min, amt_max)
        else:
            lo, hi = st.session_state["sidebar_amount_range"]
            st.session_state["sidebar_amount_range"] = (
                max(amt_min, min(lo, amt_max)),
                max(amt_min, min(hi, amt_max)),
            )
        amount_range = st.slider(
            "Plage de montant",
            min_value=amt_min,
            max_value=amt_max,
            format="%.2f",
            key="sidebar_amount_range",
        )

        timeline_cutoff = None
        if timeline_values:
            timeline_idx = st.slider(
                "Chronologie",
                min_value=0,
                max_value=len(timeline_values) - 1,
                value=len(timeline_values) - 1,
                help="Scrubber temporel — synchronise carte, KPIs et graphiques.",
            )
            st.caption(
                f"Jusqu'à **{timeline_labels[timeline_idx]}** "
                f"({timeline_values[timeline_idx].strftime('%Y-%m-%d %H:%M')})"
            )
            timeline_cutoff = timeline_values[timeline_idx]

        st.divider()
        tx_options = ["— Aucune —"] + (
            df["transaction_id"].dropna().astype(str).tolist()
            if "transaction_id" in df.columns
            else []
        )
        tx_display = st.session_state.get("selected_tx_id") or "— Aucune —"
        if tx_display not in tx_options:
            tx_display = "— Aucune —"
        st.session_state["sidebar_tx_inspect"] = tx_display
        picked = st.selectbox(
            "Transaction à inspecter",
            options=tx_options,
            help="Zoom carte, parcours client et jauge de risque.",
            key="sidebar_tx_inspect",
        )
        st.session_state["selected_tx_id"] = None if picked == "— Aucune —" else picked

        if not presentation_mode:
            st.divider()
            st.markdown("**Signaux surveillés**")
            st.markdown(
                "- Déviation de montant\n"
                "- Incohérence géographique\n"
                "- Vélocité anormale\n"
                "- Carte absente à risque\n"
                "- Données invalides"
            )

    focus_country = st.session_state.get("active_country", _ALL_COUNTRIES_LABEL)
    if focus_country and focus_country != _ALL_COUNTRIES_LABEL:
        active_countries = [focus_country]
    else:
        active_countries = selected_countries if selected_countries else countries

    filtered_df = _apply_dashboard_filters(
        df,
        countries=active_countries,
        users=selected_users,
        statut_filter=statut_filter,
        amount_range=amount_range,
        timeline_cutoff=timeline_cutoff,
    )

    country_label = (
        focus_country
        if focus_country != _ALL_COUNTRIES_LABEL
        else (
            ", ".join(selected_countries)
            if selected_countries and len(selected_countries) < len(countries)
            else "tous les pays"
        )
    )
    map_focus = focus_country

    if not presentation_mode:
        st.markdown(
            '<div class="country-chips">'
            '<span class="country-chips-label">Raccourcis pays :</span></div>',
            unsafe_allow_html=True,
        )
        _render_country_chips(countries)

    kpi_col, gauge_col = st.columns([3, 1])
    with kpi_col:
        _render_kpi_cards(filtered_df, animated=reveal)
    with gauge_col:
        selected_tx_id = st.session_state.get("selected_tx_id")
        selected_row = None
        if selected_tx_id:
            match = filtered_df[filtered_df["transaction_id"].astype(str) == str(selected_tx_id)]
            if not match.empty:
                selected_row = match.iloc[0]
        if selected_row is not None:
            _render_risk_gauge(filtered_df, score_override=float(selected_row["fraud_score"]))
        else:
            _render_risk_gauge(filtered_df)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Géographie</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-title">Centre de contrôle géographique</p>'
        f'<p class="section-desc">Carte interactive plein écran — {country_label}. '
        "Bleu = conforme, rouge = alerte, taille = montant, arcs = fraude géographique.</p>",
        unsafe_allow_html=True,
    )

    journey_user = None
    if selected_row is not None:
        journey_user = str(selected_row.get("user_id", ""))

    map_height = MAP_HEIGHT_PRESENTATION if presentation_mode else MAP_HEIGHT

    if presentation_mode:
        _render_world_map(
            filtered_df,
            selected_country=map_focus,
            height=map_height,
            key="world_map_presentation",
            highlight_tx_id=selected_tx_id,
            journey_user_id=journey_user,
        )
        if selected_row is not None:
            _render_transaction_detail_panel(selected_row)
    else:
        with st.expander("Carte mondiale — Vue complète", expanded=True):
            map_col, detail_col = st.columns([2.5, 1])
            with map_col:
                _render_world_map(
                    filtered_df,
                    selected_country=map_focus,
                    height=MAP_HEIGHT_FULLSCREEN,
                    key="world_map_fullscreen",
                    highlight_tx_id=selected_tx_id,
                    journey_user_id=journey_user,
                )
            with detail_col:
                _render_transaction_detail_panel(selected_row)
                if selected_row is not None:
                    _render_risk_gauge(
                        filtered_df,
                        score_override=float(selected_row["fraud_score"]),
                    )

    if not presentation_mode:
        _render_alert_pulse_panel(filtered_df)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Registre</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-title">Transactions</p>'
            f'<p class="section-desc">{len(filtered_df)} enregistrement(s) — {country_label}</p>',
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("Aucune transaction ne correspond aux filtres.")
        else:
            _render_premium_table(filtered_df)

        st.markdown("<hr>", unsafe_allow_html=True)
        _render_charts(filtered_df, _ALL_COUNTRIES_LABEL)
        st.markdown("<hr>", unsafe_allow_html=True)
        _render_suspicious_details(filtered_df)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_preview_table(preview_df: pd.DataFrame) -> str:
    cols = list(preview_df.columns)[:8]
    subset = preview_df[cols]
    headers = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for _, row in subset.iterrows():
        cells = "".join(
            f"<td>{row[c] if pd.notna(row[c]) else '—'}</td>" for c in cols
        )
        rows += f"<tr>{cells}</tr>"
    return f"""
    <div class="preview-box">
        <div style="margin-bottom:0.5rem;font-family:Inter,sans-serif;font-size:0.72rem;
                    letter-spacing:0.08em;text-transform:uppercase;color:{_COLOR_TEXT_MUTED};">
            Aperçu — {len(subset)} premières lignes
        </div>
        <table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
    </div>
    """


def main() -> None:
    st.set_page_config(
        page_title="INTELO · Fraud Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _init_session_state()

    _render_prestige_header(
        "Importez vos transactions ou utilisez le jeu de démonstration pré-chargé pour lancer l'analyse."
    )

    col_sample, col_upload = st.columns(2)

    transactions: list[dict] = st.session_state.get("transactions") or []
    data_ready = bool(transactions)

    with col_sample:
        st.markdown(
            f"""
            <div class="glass-card">
                <p class="glass-card-title">Données d'exemple</p>
                <p class="glass-card-desc">
                    Jeu de démonstration prêt à l'emploi — un clic pour charger
                    <span class="mono">{SAMPLE_CSV.name}</span>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview_df = _load_sample_preview()
        st.markdown(_render_preview_table(preview_df), unsafe_allow_html=True)

        if st.button("Charger les données d'exemple", type="primary", key="load_sample"):
            st.session_state["transactions"] = load_transactions(str(SAMPLE_CSV))
            st.session_state["data_source"] = "sample"
            st.session_state.pop("results", None)
            st.session_state.pop("analyzed", None)
            st.rerun()

        if st.session_state.get("data_source") == "sample" and st.session_state.get("transactions"):
            transactions = st.session_state["transactions"]
            data_ready = True
            st.markdown(
                f"<p style='color:{_COLOR_ACCENT};font-size:0.85rem;margin-top:0.75rem;'>"
                f"<span class='mono'>{len(transactions)}</span> transactions d'exemple actives "
                f"(<span class='mono'>{SAMPLE_CSV.name}</span>)</p>",
                unsafe_allow_html=True,
            )

    with col_upload:
        st.markdown(
            """
            <div class="glass-card">
                <p class="glass-card-title">Import CSV</p>
                <p class="glass-card-desc">
                    Déposez votre fichier de transactions au format CSV standard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="main_csv_uploader",
        )
        if uploaded:
            tmp = Path(".streamlit_upload.csv")
            tmp.write_bytes(uploaded.getvalue())
            try:
                transactions = load_transactions(str(tmp))
                st.session_state["transactions"] = transactions
                st.session_state["data_source"] = "import"
                st.session_state.pop("results", None)
                st.session_state.pop("analyzed", None)
                data_ready = True
                st.markdown(
                    f"<p style='color:{_COLOR_ACCENT};font-size:0.85rem;'>"
                    f"<span class='mono'>{len(transactions)}</span> transactions — "
                    f"<span class='mono'>{uploaded.name}</span></p>",
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Erreur de lecture : {exc}")
            finally:
                tmp.unlink(missing_ok=True)
        elif st.session_state.get("data_source") == "import" and st.session_state.get("transactions"):
            transactions = st.session_state["transactions"]
            data_ready = True

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        analyze_clicked = st.button(
            "Lancer l'analyse",
            type="primary",
            use_container_width=True,
        )

    if not data_ready or not transactions:
        st.info("Chargez des données ou importez un CSV, puis lancez l'analyse.")
        return

    if analyze_clicked:
        progress = st.progress(0, text="Initialisation de l'analyse...")
        status = st.empty()
        try:
            for pct, label in _ANALYSIS_STEPS:
                status.markdown(f"**{label}**")
                progress.progress(pct, text=label)
                time.sleep(0.12)
            results = detect_fraud(transactions)
            progress.progress(1.0, text="Analyse terminée")
            st.session_state["results"] = results
            st.session_state["analyzed"] = True
            st.session_state["dashboard_reveal"] = True
            st.session_state["selected_tx_id"] = None
            time.sleep(0.1)
            progress.empty()
            status.empty()
        except NotImplementedError:
            progress.empty()
            status.empty()
            st.error("Implémentez d'abord detect_fraud dans fraud_detection.py.")
            return
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Erreur d'analyse : {exc}")
            return

    if st.session_state.get("analyzed") and st.session_state.get("results"):
        render_interface(
            st.session_state["transactions"],
            st.session_state["results"],
        )


if __name__ == "__main__":
    main()
