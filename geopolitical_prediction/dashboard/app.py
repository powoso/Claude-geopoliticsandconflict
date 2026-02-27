"""Streamlit web dashboard for geopolitical prediction markets.

A polished, production-quality UI providing:
  - Executive overview with live probability gauges
  - Interactive world map with active conflict probabilities
  - Real-time escalation monitoring with heatmap and trend charts
  - Bilateral sentiment tracker with multi-scale comparison
  - Alliance network interactive visualization
  - Historical analog comparison panel
  - Commodity stress monitor
  - Market price vs model probability comparison
  - Historical accuracy tracking and calibration plots
  - Live alert feed with severity filtering

Launch with:  streamlit run geopolitical_prediction/dashboard/app.py
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme / style constants
# ---------------------------------------------------------------------------
DARK_BG = "#0E1117"
CARD_BG = "#1A1E2E"
ACCENT_BLUE = "#4E8CFF"
ACCENT_RED = "#FF4B4B"
ACCENT_GREEN = "#00CC96"
ACCENT_ORANGE = "#FF8C00"
ACCENT_YELLOW = "#FFD700"
TEXT_PRIMARY = "#FAFAFA"
TEXT_MUTED = "#8B949E"
BORDER = "#30363D"

CUSTOM_CSS = """
<style>
/* ---- Global overrides ---- */
[data-testid="stAppViewContainer"] {font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
section[data-testid="stSidebar"] {background: #131720; border-right: 1px solid #1E2433;}
section[data-testid="stSidebar"] h1 {font-size: 1.1rem; letter-spacing: 0.05em; text-transform: uppercase; color: #8B949E;}

/* ---- Cards ---- */
.geo-card {
    background: #1A1E2E; border: 1px solid #262D3D; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.geo-card:hover {border-color: #4E8CFF;}
.geo-card h3 {margin: 0 0 0.3rem 0; font-size: 1rem; color: #E6EDF3;}
.geo-card .probability {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.geo-card .subtitle {font-size: 0.82rem; color: #8B949E; margin-bottom: 0.5rem;}
.geo-card .meta {font-size: 0.75rem; color: #6E7681;}

/* ---- Probability badge colours ---- */
.prob-low    {color: #00CC96;}
.prob-medium {color: #FFD700;}
.prob-high   {color: #FF8C00;}
.prob-crit   {color: #FF4B4B;}

/* ---- Pill / tag ---- */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase;
}
.pill-invasion   {background: #3B1C2E; color: #FF4B4B;}
.pill-sanctions  {background: #2E2A1C; color: #FFD700;}
.pill-trade_deal {background: #1C2E1F; color: #00CC96;}
.pill-ceasefire  {background: #1C2530; color: #4E8CFF;}
.pill-escalating {background: #3B1C2E; color: #FF4B4B;}
.pill-stable     {background: #1C2530; color: #4E8CFF;}
.pill-de_escalating {background: #1C2E1F; color: #00CC96;}

/* ---- Alert rows ---- */
.alert-critical {border-left: 4px solid #FF4B4B; background: #2A1520; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;}
.alert-warning  {border-left: 4px solid #FF8C00; background: #2A2215; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;}
.alert-info     {border-left: 4px solid #4E8CFF; background: #151E2E; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;}
.alert-title {font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem;}
.alert-msg   {font-size: 0.82rem; color: #C9D1D9;}
.alert-time  {font-size: 0.72rem; color: #6E7681; margin-top: 0.3rem;}

/* ---- KPI row ---- */
.kpi-row {display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;}
.kpi-box {
    flex: 1; min-width: 160px; background: #1A1E2E; border: 1px solid #262D3D;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.kpi-value {font-size: 1.8rem; font-weight: 700; line-height: 1.1;}
.kpi-label {font-size: 0.78rem; color: #8B949E; margin-top: 0.25rem;}

/* ---- Section headers ---- */
.section-header {
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #8B949E; margin: 1.5rem 0 0.6rem 0;
    padding-bottom: 0.4rem; border-bottom: 1px solid #1E2433;
}

/* ---- Scrollable tables ---- */
.styled-table {width: 100%; border-collapse: collapse; font-size: 0.85rem;}
.styled-table th {
    text-align: left; padding: 0.6rem 0.8rem; color: #8B949E;
    font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.04em; border-bottom: 1px solid #262D3D;
}
.styled-table td {padding: 0.6rem 0.8rem; border-bottom: 1px solid #1E2433; color: #C9D1D9;}
.styled-table tr:hover td {background: #1E2433;}
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob_color_class(p: float) -> str:
    if p < 0.15:
        return "prob-low"
    if p < 0.40:
        return "prob-medium"
    if p < 0.70:
        return "prob-high"
    return "prob-crit"


def _prob_color_hex(p: float) -> str:
    if p < 0.15:
        return ACCENT_GREEN
    if p < 0.40:
        return ACCENT_YELLOW
    if p < 0.70:
        return ACCENT_ORANGE
    return ACCENT_RED


def _trend_pill(trend: str) -> str:
    css = "pill-stable"
    if "escalat" in trend and "de" not in trend:
        css = "pill-escalating"
    elif "de_escalat" in trend:
        css = "pill-de_escalating"
    label = trend.replace("_", " ").title()
    return f'<span class="pill {css}">{label}</span>'


def _svg_gauge(value: float, size: int = 110, stroke: int = 10) -> str:
    """Return an SVG donut-gauge for a probability value."""
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - value)
    color = _prob_color_hex(value)
    pct = f"{value:.0%}"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{size/2}" cy="{size/2}" r="{r}"
              fill="none" stroke="#262D3D" stroke-width="{stroke}"/>
      <circle cx="{size/2}" cy="{size/2}" r="{r}"
              fill="none" stroke="{color}" stroke-width="{stroke}"
              stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
              stroke-linecap="round"
              transform="rotate(-90 {size/2} {size/2})"/>
      <text x="50%" y="54%" text-anchor="middle" fill="{color}"
            font-size="{size*0.26}px" font-weight="700"
            font-family="Inter,sans-serif">{pct}</text>
    </svg>"""


def _mini_bar(value: float, max_val: float = 1.0, width: int = 80, height: int = 10) -> str:
    """Inline SVG progress bar."""
    pct = min(value / max_val, 1.0)
    color = _prob_color_hex(pct)
    filled = int(pct * width)
    return (
        f'<svg width="{width}" height="{height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" fill="#262D3D"/>'
        f'<rect x="0" y="0" width="{filled}" height="{height}" rx="4" fill="{color}"/>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# Demo / seed data generators
# ---------------------------------------------------------------------------

def _generate_demo_escalation_data() -> pd.DataFrame:
    np.random.seed(7)
    countries = [
        ("Ukraine", "UKR", 48.38, 31.17, 0.87, "stable", 480, 3100),
        ("Sudan", "SDN", 15.50, 32.56, 0.74, "escalating", 340, 1700),
        ("Myanmar", "MMR", 19.76, 96.08, 0.69, "escalating", 195, 420),
        ("Syria", "SYR", 34.80, 38.99, 0.56, "stable", 130, 310),
        ("Ethiopia", "ETH", 9.15, 40.49, 0.43, "de_escalating", 88, 190),
        ("DR Congo", "COD", -4.04, 21.76, 0.62, "escalating", 260, 850),
        ("Somalia", "SOM", 5.15, 46.20, 0.58, "stable", 145, 380),
        ("Yemen", "YEM", 15.55, 48.52, 0.71, "stable", 210, 620),
        ("Mali", "MLI", 17.57, -4.00, 0.47, "de_escalating", 72, 130),
        ("Nigeria", "NGA", 9.08, 7.49, 0.52, "stable", 165, 410),
        ("Israel / Palestine", "ISR", 31.05, 34.85, 0.83, "escalating", 550, 4200),
        ("Pakistan", "PAK", 30.38, 69.35, 0.38, "de_escalating", 62, 95),
    ]
    rows = []
    for name, iso, lat, lon, esc, trend, ev, fat in countries:
        rows.append({
            "Country": name, "ISO3": iso, "Latitude": lat, "Longitude": lon,
            "Escalation Score": esc, "Trend": trend,
            "Events (30d)": ev, "Fatalities (30d)": fat,
            "Battles": int(ev * 0.35), "Explosions": int(ev * 0.25),
            "Violence vs Civilians": int(ev * 0.15),
            "Protests": int(ev * 0.15), "Riots": int(ev * 0.10),
        })
    return pd.DataFrame(rows)


def _generate_demo_sentiment_series(pair: str, days: int = 90) -> pd.DataFrame:
    np.random.seed(hash(pair) % 2**31)
    dates = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
    base = np.random.choice([-3, -1, 0, 1], p=[0.3, 0.3, 0.2, 0.2])
    noise = np.random.normal(0, 1.5, days)
    trend_component = np.linspace(0, np.random.uniform(-2, 0.5), days)
    tone = base + noise + trend_component
    volume = np.random.poisson(50, days) + np.random.randint(10, 40, days)
    return pd.DataFrame({"date": dates, "avg_tone": tone, "article_count": volume})


def _generate_demo_commodity_data() -> pd.DataFrame:
    np.random.seed(11)
    days = 180
    dates = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
    commodities = {
        "Brent Crude ($/bbl)": (78, 5),
        "Gold ($/oz)": (2050, 80),
        "Wheat (c/bu)": (600, 40),
        "Natural Gas ($/MMBtu)": (2.8, 0.4),
    }
    rows = []
    for name, (base, vol) in commodities.items():
        prices = base + np.cumsum(np.random.normal(0, vol * 0.02, days))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "commodity": name, "price": p})
    return pd.DataFrame(rows)


def _generate_demo_alerts() -> list[dict[str, Any]]:
    now = datetime.utcnow()
    return [
        {"severity": "critical", "title": "Conflict Escalation",
         "country": "Sudan", "message": "Escalation score crossed 0.74 threshold. 340 events in 30d with accelerating fatality trend.",
         "time": now - timedelta(minutes=12)},
        {"severity": "critical", "title": "Conflict Escalation",
         "country": "Israel / Palestine", "message": "Escalation score at 0.83 with 550 events in 30d. Civilian targeting ratio above 25%.",
         "time": now - timedelta(minutes=35)},
        {"severity": "warning", "title": "Sentiment Deterioration",
         "country": "China-Taiwan", "message": "GDELT bilateral tone z-score fell to -2.4. Media coverage volume +180%.",
         "time": now - timedelta(hours=1, minutes=15)},
        {"severity": "warning", "title": "Commodity Stress",
         "country": "Global", "message": "Brent crude 5-day return z-score = +2.3. Russia-conflict stress index elevated.",
         "time": now - timedelta(hours=2, minutes=40)},
        {"severity": "info", "title": "Social Media Spike",
         "country": "North Korea", "message": "Keyword 'missile launch' volume spike: 840 tweets/hr (z=3.1).",
         "time": now - timedelta(hours=4)},
        {"severity": "info", "title": "Model-Market Divergence",
         "country": "Russia-Ukraine", "message": "Ceasefire model P (18%) diverges from Polymarket (32%) by +14pp.",
         "time": now - timedelta(hours=6)},
    ]


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def _page_overview(st: Any, go: Any, px: Any) -> None:
    """Executive overview with KPI row, world map, and top-line forecasts."""
    from geopolitical_prediction.models.ensemble import EnsembleProbabilityEngine
    from geopolitical_prediction.markets.questions import ACTIVE_QUESTIONS

    engine = EnsembleProbabilityEngine()

    alerts_data = _generate_demo_alerts()
    critical_count = sum(1 for a in alerts_data if a["severity"] == "critical")
    warning_count = sum(1 for a in alerts_data if a["severity"] == "warning")
    esc_data = _generate_demo_escalation_data()
    active_conflicts = len(esc_data[esc_data["Escalation Score"] >= 0.6])

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-box">
        <div class="kpi-value" style="color:{ACCENT_RED}">{critical_count}</div>
        <div class="kpi-label">Critical Alerts</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-value" style="color:{ACCENT_ORANGE}">{warning_count}</div>
        <div class="kpi-label">Warnings</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-value" style="color:{ACCENT_YELLOW}">{active_conflicts}</div>
        <div class="kpi-label">Active Conflicts</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-value" style="color:{ACCENT_BLUE}">{len(ACTIVE_QUESTIONS)}</div>
        <div class="kpi-label">Tracked Markets</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- World map ----
    st.markdown('<div class="section-header">Global Conflict Heatmap</div>', unsafe_allow_html=True)
    fig_map = px.scatter_geo(
        esc_data, lat="Latitude", lon="Longitude", size="Events (30d)",
        color="Escalation Score",
        color_continuous_scale=["#00CC96", "#FFD700", "#FF8C00", "#FF4B4B"],
        range_color=[0, 1],
        hover_name="Country",
        hover_data={
            "Escalation Score": ":.2f", "Events (30d)": True,
            "Fatalities (30d)": True, "Trend": True,
            "Latitude": False, "Longitude": False,
        },
        size_max=30, projection="natural earth",
    )
    fig_map.update_layout(
        geo=dict(
            bgcolor="rgba(0,0,0,0)", landcolor="#1A1E2E", oceancolor="#0E1117",
            showocean=True, showlakes=False,
            coastlinecolor="#30363D", countrycolor="#30363D", framecolor="#30363D",
        ),
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Escalation", thickness=12, len=0.6,
            tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_MUTED, size=11),
        ),
        font=dict(color=TEXT_PRIMARY),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ---- Top forecasts grid ----
    st.markdown('<div class="section-header">Active Market Forecasts</div>', unsafe_allow_html=True)
    estimates = [(q, engine.estimate_probability(q)) for q in ACTIVE_QUESTIONS]

    cols = st.columns(3)
    for idx, (q, est) in enumerate(estimates):
        col = cols[idx % 3]
        days_left = (q.resolution_date - datetime.utcnow()).days if q.resolution_date else 0
        pill_cls = f"pill-{q.question_type}"
        gauge = _svg_gauge(est.probability, size=90, stroke=8)
        with col:
            st.markdown(f"""
            <div class="geo-card">
              <div style="display:flex;align-items:flex-start;gap:0.9rem;">
                <div class="gauge-container">{gauge}</div>
                <div style="flex:1;">
                  <span class="pill {pill_cls}">{q.question_type.replace('_',' ')}</span>
                  <h3 style="margin-top:0.4rem;font-size:0.88rem;line-height:1.35;">{q.question_text}</h3>
                  <div class="meta">
                    Confidence {est.confidence:.0%} &nbsp;|&nbsp; Base rate {est.base_rate:.0%}
                    &nbsp;|&nbsp; {max(days_left,0)}d left
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


def _page_escalation(st: Any, go: Any, px: Any) -> None:
    """Conflict escalation monitor."""
    st.markdown('<div class="section-header">Conflict Escalation Monitor</div>', unsafe_allow_html=True)
    esc_df = _generate_demo_escalation_data()

    sort_col = st.selectbox("Sort by", ["Escalation Score", "Events (30d)", "Fatalities (30d)"], index=0)
    sorted_df = esc_df.sort_values(sort_col, ascending=False)

    table_rows = ""
    for _, row in sorted_df.iterrows():
        bar = _mini_bar(row["Escalation Score"], 1.0, 80, 8)
        trend_html = _trend_pill(row["Trend"])
        table_rows += f"""
        <tr>
          <td style="font-weight:600;">{row['Country']}</td>
          <td><span class="{_prob_color_class(row['Escalation Score'])}" style="font-weight:700;">{row['Escalation Score']:.2f}</span> {bar}</td>
          <td>{trend_html}</td>
          <td>{row['Events (30d)']:,}</td>
          <td>{row['Fatalities (30d)']:,}</td>
          <td>{row['Battles']}</td>
          <td>{row['Explosions']}</td>
          <td>{row['Violence vs Civilians']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>
        <th>Country</th><th>Escalation Score</th><th>Trend</th>
        <th>Events</th><th>Fatalities</th><th>Battles</th><th>Explosions</th><th>Viol. vs Civ.</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Country deep dive
    st.markdown('<div class="section-header">Country Deep Dive</div>', unsafe_allow_html=True)
    country = st.selectbox("Select country", sorted_df["Country"].tolist())
    row = sorted_df[sorted_df["Country"] == country].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="geo-card" style="text-align:center;">
          {_svg_gauge(row['Escalation Score'], size=130, stroke=12)}
          <div style="font-size:0.85rem;color:#8B949E;margin-top:0.3rem;">Escalation Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        event_types = ["Battles", "Explosions", "Violence vs Civilians", "Protests", "Riots"]
        values = [row[et] for et in event_types]
        fig_bar = go.Figure(go.Bar(
            y=event_types, x=values, orientation="h",
            marker_color=[ACCENT_RED, ACCENT_ORANGE, "#E040FB", ACCENT_BLUE, ACCENT_YELLOW],
        ))
        fig_bar.update_layout(
            height=240, margin=dict(l=0, r=10, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1E2433", color=TEXT_MUTED),
            yaxis=dict(color=TEXT_MUTED),
            title=dict(text="Event Breakdown (30d)", font=dict(size=13, color=TEXT_MUTED)),
            font=dict(color=TEXT_PRIMARY, size=11),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    with c3:
        np.random.seed(hash(country) % 2**31)
        days_range = pd.date_range(end=datetime.utcnow(), periods=90, freq="D")
        trend_vals = np.clip(row["Escalation Score"] + np.cumsum(np.random.normal(0, 0.01, 90)), 0, 1)
        color = _prob_color_hex(row["Escalation Score"])
        r_val, g_val, b_val = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig_spark = go.Figure(go.Scatter(
            x=days_range, y=trend_vals, mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=f"rgba({r_val},{g_val},{b_val},0.1)",
        ))
        fig_spark.update_layout(
            height=240, margin=dict(l=0, r=10, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1E2433", color=TEXT_MUTED),
            yaxis=dict(range=[0, 1], gridcolor="#1E2433", color=TEXT_MUTED),
            title=dict(text="90-Day Trend", font=dict(size=13, color=TEXT_MUTED)),
            font=dict(color=TEXT_PRIMARY, size=11), showlegend=False,
        )
        st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})


def _page_sentiment(st: Any, go: Any, px: Any) -> None:
    """Bilateral sentiment tracker."""
    st.markdown('<div class="section-header">Bilateral Sentiment Tracker</div>', unsafe_allow_html=True)

    pairs = [
        "Russia - Ukraine", "China - Taiwan", "India - Pakistan",
        "US - China", "US - Iran", "Israel - Iran",
        "Saudi Arabia - Iran", "North Korea - South Korea",
    ]
    selected_pairs = st.multiselect("Select country pairs", pairs, default=pairs[:3])
    if not selected_pairs:
        st.info("Select at least one country pair.")
        return

    fig = go.Figure()
    tone_colors = [ACCENT_BLUE, ACCENT_RED, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_YELLOW, "#E040FB", "#00BCD4", "#8BC34A"]
    for i, pair in enumerate(selected_pairs):
        df = _generate_demo_sentiment_series(pair)
        ma7 = df["avg_tone"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=ma7, mode="lines", name=pair,
            line=dict(color=tone_colors[i % len(tone_colors)], width=2),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#30363D")
    fig.add_hrect(y0=-10, y1=-3, fillcolor="rgba(255,75,75,0.05)", line_width=0)
    fig.add_hrect(y0=3, y1=10, fillcolor="rgba(0,204,150,0.05)", line_width=0)
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1E2433", color=TEXT_MUTED),
        yaxis=dict(title="Average Tone (7d MA)", gridcolor="#1E2433", color=TEXT_MUTED,
                   zeroline=True, zerolinecolor="#30363D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        font=dict(color=TEXT_PRIMARY, size=11),
        title=dict(text="GDELT Bilateral Tone Over Time", font=dict(size=14, color=TEXT_MUTED)),
    )
    fig.add_annotation(x=0.01, y=-5, text="Hostile", showarrow=False,
                       font=dict(color=ACCENT_RED, size=10), xref="paper", opacity=0.6)
    fig.add_annotation(x=0.01, y=5, text="Cooperative", showarrow=False,
                       font=dict(color=ACCENT_GREEN, size=10), xref="paper", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-header">Pair Detail</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(selected_pairs), 3))
    for i, pair in enumerate(selected_pairs):
        df = _generate_demo_sentiment_series(pair)
        current = df["avg_tone"].iloc[-1]
        avg_7d = df["avg_tone"].tail(7).mean()
        avg_30d = df["avg_tone"].tail(30).mean()
        trend = avg_7d - avg_30d
        vol_7d = int(df["article_count"].tail(7).sum())
        category = "hostile" if current < -5 else "negative" if current < -2 else "neutral" if current < 2 else "positive" if current < 5 else "cooperative"
        cat_color = ACCENT_RED if "hostile" in category else ACCENT_ORANGE if "negative" in category else TEXT_MUTED if "neutral" in category else ACCENT_GREEN
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="geo-card">
              <h3>{pair}</h3>
              <div style="font-size:1.5rem;font-weight:700;color:{cat_color};">{current:.1f}</div>
              <div class="subtitle">Current tone &bull; <span style="color:{cat_color};text-transform:uppercase;font-weight:600;">{category}</span></div>
              <div class="meta">7d avg: {avg_7d:.1f} | 30d avg: {avg_30d:.1f} | Trend: {trend:+.1f}</div>
              <div class="meta">{vol_7d:,} articles (7d)</div>
            </div>
            """, unsafe_allow_html=True)


def _page_alliances(st: Any, go: Any, px: Any) -> None:
    """Alliance network analysis."""
    from geopolitical_prediction.features.alliance import AllianceNetworkAnalyzer, MILITARY_SPENDING_RELATIVE

    st.markdown('<div class="section-header">Alliance Network Analysis</div>', unsafe_allow_html=True)
    analyzer = AllianceNetworkAnalyzer()
    tab1, tab2, tab3 = st.tabs(["Alliance Explorer", "Intervention Simulator", "Flash Points"])

    with tab1:
        country = st.selectbox("Select country", sorted(MILITARY_SPENDING_RELATIVE.keys()), index=0)
        allies = analyzer.get_allies(country)
        if allies:
            df = pd.DataFrame(allies).sort_values("military_weight", ascending=False)
            fig = go.Figure(go.Bar(
                y=df["ally"], x=df["military_weight"], orientation="h",
                marker_color=[ACCENT_BLUE if d else ACCENT_YELLOW for d in df["defense_pact"]],
                text=[f"{c:.0%}" for c in df["credibility"]], textposition="auto",
            ))
            fig.update_layout(
                height=max(300, len(df) * 28), margin=dict(l=0, r=20, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Relative Military Spending", gridcolor="#1E2433", color=TEXT_MUTED),
                yaxis=dict(color=TEXT_MUTED, autorange="reversed"),
                title=dict(text=f"Allies of {country}", font=dict(size=14, color=TEXT_MUTED)),
                font=dict(color=TEXT_PRIMARY, size=11),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Blue = defense pact | Yellow = entente/other")
        else:
            st.info(f"No modeled alliances for {country}")

    with tab2:
        st.markdown("**If Country A attacks Country B, how likely are B's allies to intervene?**")
        c1, c2 = st.columns(2)
        sorted_countries = sorted(MILITARY_SPENDING_RELATIVE.keys())
        with c1:
            defender = st.selectbox("Defender", sorted_countries, index=sorted_countries.index("Poland") if "Poland" in sorted_countries else 0)
        with c2:
            att_opts = [c for c in sorted_countries if c != defender]
            attacker = st.selectbox("Attacker", att_opts, index=att_opts.index("Russia") if "Russia" in att_opts else 0)

        analysis = analyzer.estimate_intervention_probability(defender, attacker)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""<div class="geo-card" style="text-align:center;">
              {_svg_gauge(analysis.estimated_intervention_probability, 100, 9)}
              <div style="font-size:0.8rem;color:#8B949E;">Intervention P</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.metric("Total Allies", analysis.total_allies)
        with mc3:
            st.metric("Defense Pacts", analysis.defense_pact_allies)
        with mc4:
            st.metric("Mil. Ratio", f"{analysis.collective_military_spending:.1f}x")
        if analysis.major_power_allies:
            st.markdown(f"**Major power allies:** {', '.join(analysis.major_power_allies)}")
        if analysis.weakest_link:
            st.markdown(f"**Weakest link:** {analysis.weakest_link}")

    with tab3:
        flash_points = analyzer.find_flash_points()[:10]
        if flash_points:
            fp_df = pd.DataFrame(flash_points)
            fp_df["label"] = fp_df.apply(lambda r: f"{r['country_a']} vs {r['country_b']}", axis=1)
            fig_fp = go.Figure(go.Bar(
                y=fp_df["label"], x=fp_df["tension_score"], orientation="h",
                marker=dict(color=fp_df["tension_score"], colorscale=["#FFD700", "#FF4B4B"]),
            ))
            fig_fp.update_layout(
                height=350, margin=dict(l=0, r=20, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Tension Score", gridcolor="#1E2433", color=TEXT_MUTED),
                yaxis=dict(color=TEXT_MUTED, autorange="reversed"),
                title=dict(text="Top Flash Points", font=dict(size=14, color=TEXT_MUTED)),
                font=dict(color=TEXT_PRIMARY, size=11), showlegend=False,
            )
            st.plotly_chart(fig_fp, use_container_width=True, config={"displayModeBar": False})


def _page_commodities(st: Any, go: Any, px: Any) -> None:
    """Commodity stress monitor."""
    st.markdown('<div class="section-header">Commodity Stress Monitor</div>', unsafe_allow_html=True)
    comm_df = _generate_demo_commodity_data()
    commodities = comm_df["commodity"].unique()
    colors = [ACCENT_ORANGE, ACCENT_YELLOW, "#8BC34A", ACCENT_BLUE]

    fig = go.Figure()
    for i, comm in enumerate(commodities):
        sub = comm_df[comm_df["commodity"] == comm]
        norm = (sub["price"] - sub["price"].iloc[0]) / sub["price"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=sub["date"], y=norm, mode="lines",
            name=comm, line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#30363D")
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1E2433", color=TEXT_MUTED),
        yaxis=dict(title="% Change (180d)", gridcolor="#1E2433", color=TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        font=dict(color=TEXT_PRIMARY, size=11),
        title=dict(text="Commodity Price Trends (Normalized)", font=dict(size=14, color=TEXT_MUTED)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-header">Geopolitical Scenario Stress Indices</div>', unsafe_allow_html=True)
    scenarios = {
        "Middle East Conflict": {"stress": 0.65, "commodities": "Oil, Gold, Gas"},
        "Russia Conflict": {"stress": 0.48, "commodities": "Oil, Gas, Wheat, Palladium"},
        "China-Taiwan": {"stress": 0.32, "commodities": "Copper, Gold, Oil"},
        "Food Crisis": {"stress": 0.41, "commodities": "Wheat, Corn"},
    }
    cols = st.columns(4)
    for i, (name, info) in enumerate(scenarios.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="geo-card" style="text-align:center;">
              {_svg_gauge(info['stress'], 90, 8)}
              <h3 style="font-size:0.85rem;margin-top:0.3rem;">{name}</h3>
              <div class="meta">{info['commodities']}</div>
            </div>
            """, unsafe_allow_html=True)


def _page_analogs(st: Any, go: Any, px: Any) -> None:
    """Historical analog comparison panel."""
    from geopolitical_prediction.models.analog_matching import AnalogMatcher

    st.markdown('<div class="section-header">Historical Analog Matching</div>', unsafe_allow_html=True)
    matcher = AnalogMatcher()

    st.markdown("**Configure the current crisis to find historical parallels:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trigger = st.selectbox("Trigger", ["territorial", "ethnic", "resource", "ideological"])
    with c2:
        regime_a = st.selectbox("Regime A", ["democracy", "autocracy", "hybrid"])
    with c3:
        regime_b = st.selectbox("Regime B", ["autocracy", "democracy", "hybrid"])
    with c4:
        nuclear = st.checkbox("Nuclear parties", value=False)

    c5, c6, c7 = st.columns(3)
    with c5:
        alliance = st.checkbox("Alliance involvement", value=True)
    with c6:
        econ_inter = st.slider("Economic interdependence", 0.0, 1.0, 0.15)
    with c7:
        prior = st.number_input("Prior conflicts", 0, 10, 1)

    esc_level = st.slider("Current escalation level", 0, 5, 3,
                          help="0=Normal, 1=Diplomatic, 2=Economic, 3=Military Posturing, 4=Limited Action, 5=Full Conflict")

    analogs = matcher.find_analogs(
        trigger=trigger, regime_type_a=regime_a, regime_type_b=regime_b,
        nuclear_parties=nuclear, alliance_involvement=alliance,
        economic_interdependence=econ_inter, prior_conflicts=prior,
        initial_escalation_level=esc_level, top_k=6,
    )
    forecast = matcher.forecast_from_analogs(analogs, "User-configured crisis")

    if analogs:
        c_left, c_right = st.columns([1, 2])
        with c_left:
            labels = list(forecast.outcome_probabilities.keys())
            values = list(forecast.outcome_probabilities.values())
            outcome_colors = {
                "war": ACCENT_RED, "limited_conflict": ACCENT_ORANGE,
                "sanctions": ACCENT_YELLOW, "negotiated_settlement": ACCENT_GREEN,
                "status_quo": ACCENT_BLUE, "de_escalation": "#00BCD4",
            }
            fig_donut = go.Figure(go.Pie(
                labels=[l.replace("_", " ").title() for l in labels],
                values=values, hole=0.55,
                marker=dict(colors=[outcome_colors.get(l, TEXT_MUTED) for l in labels]),
                textinfo="label+percent", textfont=dict(size=11),
            ))
            fig_donut.update_layout(
                height=300, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_PRIMARY, size=11),
                title=dict(text="Predicted Outcomes", font=dict(size=13, color=TEXT_MUTED)),
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div class="geo-card">
              <div class="meta">Confidence: <b>{forecast.confidence:.0%}</b></div>
              <div class="meta">Expected duration: <b>{forecast.weighted_duration_months:.0f} months</b></div>
              <div class="meta">Analogs matched: <b>{len(analogs)}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="section-header">Top Historical Analogs</div>', unsafe_allow_html=True)
            for match in analogs:
                c = match.crisis
                sim_bar = _mini_bar(match.similarity_score, 1.0, 60, 8)
                oc = outcome_colors.get(c.outcome, TEXT_MUTED)
                st.markdown(f"""
                <div class="geo-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <h3 style="margin:0;">{c.name} ({c.year})</h3>
                      <div class="meta">{' vs '.join(c.parties)} &bull; {c.trigger}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:0.75rem;color:#8B949E;">Similarity</div>
                      <div style="font-weight:700;color:{ACCENT_BLUE};">{match.similarity_score:.0%}</div>
                      {sim_bar}
                    </div>
                  </div>
                  <div style="margin-top:0.5rem;display:flex;gap:1rem;flex-wrap:wrap;">
                    <span class="meta">Outcome: <b style="color:{oc};">{c.outcome.replace('_',' ').title()}</b></span>
                    <span class="meta">Duration: <b>{c.duration_months}mo</b></span>
                    <span class="meta">Fatalities: <b>{c.fatalities:,}</b></span>
                    <span class="meta">Resolution: <b>{c.resolution_mechanism.replace('_',' ').title()}</b></span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No analogs matched. Try adjusting parameters.")


def _page_base_rates(st: Any, go: Any, px: Any) -> None:
    """Historical base rates reference."""
    from geopolitical_prediction.features.base_rates import BaseRateEngine

    st.markdown('<div class="section-header">Historical Base Rates</div>', unsafe_allow_html=True)
    engine = BaseRateEngine()
    rates = engine.list_all_rates()

    for r in rates:
        ci_lo, ci_hi = r["confidence_interval"]
        st.markdown(f"""
        <div class="geo-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="flex:1;">
              <h3>{r['event_type'].replace('_',' ').title()}</h3>
              <div class="subtitle">{r['description']}</div>
              <div class="meta">{r['time_period']} &bull; n={r['sample_size']} &bull; CI: [{ci_lo:.0%}, {ci_hi:.0%}]</div>
            </div>
            <div style="text-align:center;min-width:100px;">
              {_svg_gauge(r['base_rate'], 80, 7)}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Regime-Type Modifiers</div>', unsafe_allow_html=True)
    event = st.selectbox("Event type", [r["event_type"] for r in rates])
    base = engine.get_base_rate(event)
    if base:
        cols = st.columns(3)
        for i, regime in enumerate(["democracy", "autocracy", "hybrid_regime"]):
            adj = engine.get_adjusted_rate(event, regime_type=regime)
            label = regime.replace("_", " ").title()
            with cols[i]:
                st.markdown(f"""
                <div class="geo-card" style="text-align:center;">
                  {_svg_gauge(adj, 90, 8)}
                  <h3 style="font-size:0.85rem;">{label}</h3>
                  <div class="meta">Adjusted from {base.base_rate:.0%}</div>
                </div>
                """, unsafe_allow_html=True)


def _page_accuracy(st: Any, go: Any, px: Any) -> None:
    """Model calibration and accuracy."""
    st.markdown('<div class="section-header">Model Calibration & Accuracy</div>', unsafe_allow_html=True)

    np.random.seed(42)
    n = 200
    predicted = np.random.beta(2, 2, n)
    actuals = np.array([np.random.random() < p * 1.05 for p in predicted]).astype(float)
    brier = float(np.mean((predicted - actuals) ** 2))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="geo-card" style="text-align:center;">
          <div class="kpi-value" style="color:{ACCENT_GREEN};">{brier:.3f}</div>
          <div class="kpi-label">Brier Score (lower = better)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="geo-card" style="text-align:center;">
          <div class="kpi-value" style="color:{ACCENT_BLUE};">{n}</div>
          <div class="kpi-label">Resolved Predictions</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        accuracy = np.mean((predicted > 0.5) == actuals)
        st.markdown(f"""<div class="geo-card" style="text-align:center;">
          <div class="kpi-value" style="color:{ACCENT_YELLOW};">{accuracy:.1%}</div>
          <div class="kpi-label">Directional Accuracy</div>
        </div>""", unsafe_allow_html=True)

    # Calibration plot
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_actuals, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_actuals.append(actuals[mask].mean())
            bin_counts.append(int(mask.sum()))

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
        line=dict(dash="dash", color="#30363D", width=1)))
    fig_cal.add_trace(go.Scatter(
        x=bin_centers, y=bin_actuals, mode="lines+markers", name="Model",
        line=dict(color=ACCENT_BLUE, width=2.5), marker=dict(size=8, color=ACCENT_BLUE),
        text=[f"n={c}" for c in bin_counts],
        hovertemplate="Predicted: %{x:.0%}<br>Actual: %{y:.0%}<br>%{text}<extra></extra>",
    ))
    fig_cal.update_layout(
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted Probability", range=[0, 1], gridcolor="#1E2433", color=TEXT_MUTED, dtick=0.1),
        yaxis=dict(title="Observed Frequency", range=[0, 1], gridcolor="#1E2433", color=TEXT_MUTED, dtick=0.1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        font=dict(color=TEXT_PRIMARY, size=11),
        title=dict(text="Reliability Diagram", font=dict(size=14, color=TEXT_MUTED)),
    )
    st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

    fig_hist = go.Figure(go.Histogram(x=predicted, nbinsx=20, marker_color=ACCENT_BLUE, opacity=0.7))
    fig_hist.update_layout(
        height=250, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted Probability", gridcolor="#1E2433", color=TEXT_MUTED),
        yaxis=dict(title="Count", gridcolor="#1E2433", color=TEXT_MUTED),
        font=dict(color=TEXT_PRIMARY, size=11),
        title=dict(text="Prediction Distribution", font=dict(size=14, color=TEXT_MUTED)),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


def _page_alerts(st: Any, go: Any, px: Any) -> None:
    """Live alert feed."""
    st.markdown('<div class="section-header">Escalation Alerts</div>', unsafe_allow_html=True)
    alerts = _generate_demo_alerts()

    c1, _ = st.columns([1, 3])
    with c1:
        severity_filter = st.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning", "info"])

    filtered = [a for a in alerts if a["severity"] in severity_filter]
    if not filtered:
        st.markdown("""<div class="geo-card" style="text-align:center;padding:2rem;">
          <div style="font-size:1.3rem;color:#00CC96;font-weight:600;">All Clear</div>
          <div class="meta" style="margin-top:0.5rem;">No alerts matching filters</div>
        </div>""", unsafe_allow_html=True)
        return

    for a in filtered:
        css_class = f"alert-{a['severity']}"
        ago = datetime.utcnow() - a["time"]
        ago_str = f"{int(ago.total_seconds() / 60)}m ago" if ago.total_seconds() < 3600 else f"{int(ago.total_seconds() / 3600)}h ago"
        badge = a["severity"].upper()
        pill_cls = "pill-escalating" if a["severity"] == "critical" else "pill-sanctions" if a["severity"] == "warning" else "pill-stable"
        st.markdown(f"""
        <div class="{css_class}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="alert-title">{a['title']} — {a['country']}</div>
              <div class="alert-msg">{a['message']}</div>
            </div>
            <div style="text-align:right;min-width:80px;">
              <span class="pill {pill_cls}">{badge}</span>
              <div class="alert-time">{ago_str}</div>
            </div>
          </div>
          <div class="alert-time">{a['time'].strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_app() -> None:
    """Create and run the Streamlit dashboard."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Dashboard requires: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="GeoPred | Geopolitical Prediction Markets",
        page_icon="\U0001F310",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# \U0001F310 GeoPred")
        st.caption("Geopolitical Prediction Markets")
        st.divider()
        pages = {
            "\U0001F4CA Overview": "overview",
            "\u2694\uFE0F Escalation Monitor": "escalation",
            "\U0001F4AC Sentiment Tracker": "sentiment",
            "\U0001F6E1\uFE0F Alliance Network": "alliances",
            "\U0001F3ED Commodity Stress": "commodities",
            "\U0001F4DC Historical Analogs": "analogs",
            "\U0001F4C8 Base Rates": "base_rates",
            "\U0001F3AF Model Accuracy": "accuracy",
            "\U0001F514 Alerts": "alerts",
        }
        page_selection = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
        current_page = pages[page_selection]
        st.divider()
        st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Data: GDELT, ACLED, SIPRI, Polymarket")

    page_map = {
        "overview": _page_overview,
        "escalation": _page_escalation,
        "sentiment": _page_sentiment,
        "alliances": _page_alliances,
        "commodities": _page_commodities,
        "analogs": _page_analogs,
        "base_rates": _page_base_rates,
        "accuracy": _page_accuracy,
        "alerts": _page_alerts,
    }
    page_map[current_page](st, go, px)


if __name__ == "__main__":
    create_app()
