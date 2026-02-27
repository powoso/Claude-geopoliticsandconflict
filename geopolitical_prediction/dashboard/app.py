"""Streamlit dashboard for geopolitical prediction markets.

Provides:
  - Interactive world map with active conflict probabilities
  - Real-time escalation monitoring with alert feeds
  - Market price vs model probability comparison charts
  - Historical accuracy tracking and calibration plots
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def create_app() -> None:
    """Create and run the Streamlit dashboard application."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Dashboard requires streamlit and plotly: pip install streamlit plotly")
        return

    from geopolitical_prediction.dashboard.alerts import AlertEngine, AlertSeverity
    from geopolitical_prediction.features.base_rates import BaseRateEngine
    from geopolitical_prediction.features.alliance import AllianceNetworkAnalyzer
    from geopolitical_prediction.markets.questions import ACTIVE_QUESTIONS

    st.set_page_config(
        page_title="Geopolitical Prediction Markets",
        page_icon="🌍",
        layout="wide",
    )

    st.title("Geopolitical Prediction Markets Dashboard")

    # Initialize session state
    if "alert_engine" not in st.session_state:
        st.session_state.alert_engine = AlertEngine()
    if "base_rate_engine" not in st.session_state:
        st.session_state.base_rate_engine = BaseRateEngine()

    # Sidebar: Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Active Markets",
            "Escalation Monitor",
            "Sentiment Tracker",
            "Alliance Network",
            "Historical Accuracy",
            "Alerts",
        ],
    )

    if page == "Active Markets":
        _render_active_markets(st, ACTIVE_QUESTIONS)
    elif page == "Escalation Monitor":
        _render_escalation_monitor(st)
    elif page == "Sentiment Tracker":
        _render_sentiment_tracker(st)
    elif page == "Alliance Network":
        _render_alliance_network(st, px)
    elif page == "Historical Accuracy":
        _render_accuracy(st, go)
    elif page == "Alerts":
        _render_alerts(st, st.session_state.alert_engine)


def _render_active_markets(st: Any, questions: list) -> None:
    """Render active prediction market questions with current probabilities."""
    st.header("Active Prediction Markets")

    # Group by question type
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q.question_type, []).append(q)

    for qtype, qs in by_type.items():
        st.subheader(qtype.replace("_", " ").title())
        for q in qs:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(q.question_text)
            with col2:
                # Placeholder probability (would come from ensemble engine)
                base_engine = st.session_state.base_rate_engine
                base_rate = base_engine.get_adjusted_rate(q.base_rate_key) or 0.5
                st.metric("Model P", f"{base_rate:.1%}")
            with col3:
                if q.resolution_date:
                    days_left = (q.resolution_date - datetime.utcnow()).days
                    st.metric("Days Left", max(days_left, 0))


def _render_escalation_monitor(st: Any) -> None:
    """Render real-time escalation monitoring view."""
    st.header("Conflict Escalation Monitor")
    st.info(
        "Connect ACLED and GDELT data pipelines to see live escalation scores. "
        "Run the data update pipeline to populate this view."
    )

    # Demo data for layout
    demo_data = pd.DataFrame(
        {
            "Country": ["Ukraine", "Sudan", "Myanmar", "Ethiopia", "Syria"],
            "Escalation Score": [0.85, 0.72, 0.68, 0.45, 0.55],
            "Trend": ["stable", "escalating", "escalating", "de_escalating", "stable"],
            "Events (30d)": [450, 320, 180, 95, 120],
            "Fatalities (30d)": [2800, 1500, 350, 200, 280],
        }
    )
    st.dataframe(demo_data, use_container_width=True)


def _render_sentiment_tracker(st: Any) -> None:
    """Render bilateral sentiment tracking view."""
    st.header("Bilateral Sentiment Tracker")
    st.info(
        "Connect GDELT sentiment pipeline to see live bilateral sentiment. "
        "Configure watched country pairs in the pipeline settings."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Country A", value="Russia")
    with col2:
        st.text_input("Country B", value="Ukraine")

    st.write("Sentiment time series will appear here when data pipelines are active.")


def _render_alliance_network(st: Any, px: Any) -> None:
    """Render alliance network visualization."""
    st.header("Alliance Network Analysis")

    analyzer = AllianceNetworkAnalyzer()
    country = st.selectbox(
        "Select Country",
        ["United States", "China", "Russia", "United Kingdom", "France", "India", "Japan"],
    )

    allies = analyzer.get_allies(country)
    if allies:
        df = pd.DataFrame(allies)
        st.subheader(f"Allies of {country}")
        st.dataframe(df, use_container_width=True)

        # Intervention probability analysis
        adversary = st.selectbox("Potential Adversary", ["China", "Russia", "North Korea", "Iran"])
        if adversary != country:
            analysis = analyzer.estimate_intervention_probability(country, adversary)
            st.metric(
                "Estimated Intervention Probability",
                f"{analysis.estimated_intervention_probability:.1%}",
            )
            st.write(f"Defense pact allies: {analysis.defense_pact_allies}")
            st.write(f"Major power allies: {', '.join(analysis.major_power_allies)}")


def _render_accuracy(st: Any, go: Any) -> None:
    """Render historical accuracy and calibration metrics."""
    st.header("Model Calibration & Accuracy")
    st.info("Calibration data will populate as predictions resolve over time.")

    # Demo calibration chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            y=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title="Calibration Plot (Demo)",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_alerts(st: Any, alert_engine: Any) -> None:
    """Render alert feed."""
    st.header("Escalation Alerts")

    alerts = alert_engine.get_active_alerts()
    if not alerts:
        st.success("No active alerts")
        return

    for alert in alerts:
        severity_colors = {
            "info": "blue",
            "warning": "orange",
            "critical": "red",
        }
        color = severity_colors.get(alert.severity.value, "gray")
        st.markdown(
            f"**[{alert.severity.value.upper()}]** {alert.title} — {alert.country}  \n"
            f"{alert.message}  \n"
            f"*{alert.timestamp.strftime('%Y-%m-%d %H:%M UTC')}*"
        )


if __name__ == "__main__":
    create_app()
