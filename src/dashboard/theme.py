"""Dashboard theme — F1 broadcast-style dark theme (CSS) + Plotly template."""
from __future__ import annotations

BG = "#0D0D0D"
PANEL = "#151515"
ACCENT = "#E10600"
TEXT = "#F5F5F5"

CSS = f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background: #0A0A0A; border-right: 2px solid {ACCENT}; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: 0.5px; }}
h1 {{ border-bottom: 3px solid {ACCENT}; padding-bottom: 6px; }}
.f1-title {{ font-size: 2.4rem; font-weight: 800; color: {TEXT};
    text-transform: uppercase; letter-spacing: 3px; }}
.f1-title span {{ color: {ACCENT}; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #222;
    border-left: 4px solid {ACCENT}; border-radius: 8px; padding: 12px; }}
.stButton>button {{ background: {ACCENT}; color: white; border: none; border-radius: 6px;
    font-weight: 700; }}
.stButton>button:hover {{ background: #ff1e10; color: white; }}
.stDataFrame {{ border: 1px solid #222; }}
.f1-chip {{ display:inline-block; background:{PANEL}; border:1px solid {ACCENT};
    border-radius:14px; padding:2px 12px; margin:2px; font-size:0.8rem; }}
</style>
"""

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": BG,
        "plot_bgcolor": PANEL,
        "font": {"color": TEXT, "family": "Helvetica, Arial, sans-serif"},
        "colorway": [ACCENT, "#00D2BE", "#FFFFFF", "#FF8700", "#0090FF",
                     "#FFD800", "#B6BABD", "#C00000", "#2293D1", "#358C75"],
        "xaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
        "yaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
    }
}


def header(subtitle: str = "") -> str:
    return (f'<div class="f1-title">F1<span>PREDICT</span></div>'
            f'<p style="color:#888;margin-top:-8px">{subtitle}</p>')
