"""Command-line interface for the geopolitical prediction system."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_update_data(args: argparse.Namespace) -> None:
    """Update data from all configured sources."""
    from geopolitical_prediction.data.gdelt import GDELTSentimentPipeline
    from geopolitical_prediction.data.acled import ACLEDConflictPipeline

    days = args.days

    if args.source in ("gdelt", "all"):
        logger.info("Updating GDELT sentiment data...")
        pipeline = GDELTSentimentPipeline()
        # Add default watch pairs
        for a, b in [("Russia", "Ukraine"), ("China", "Taiwan"), ("India", "Pakistan")]:
            pipeline.watch_country_pair(a, b)
        results = pipeline.update_all(days_back=days)
        for key, df in results.items():
            logger.info("  %s: %d records", key, len(df))

        shifts = pipeline.detect_all_shifts()
        if shifts:
            logger.warning("Sentiment shifts detected:")
            for s in shifts:
                logger.warning("  %s: z=%.2f (%s)", s["key"], s["z_score"], s["direction"])

    if args.source in ("acled", "all"):
        logger.info("Updating ACLED conflict data...")
        pipeline = ACLEDConflictPipeline()
        for country, iso3 in [("Ukraine", "UKR"), ("Sudan", "SDN"), ("Myanmar", "MMR")]:
            pipeline.watch_country(country, iso3)
        results = pipeline.update_all(days_back=days)
        for key, df in results.items():
            logger.info("  %s: %d events", key, len(df))

        escalations = pipeline.detect_all_escalations()
        for e in escalations:
            logger.info(
                "  %s: trend=%s, composite_ratio=%.2f",
                e["country_key"],
                e["trend"],
                e["composite_ratio"],
            )


def cmd_forecast(args: argparse.Namespace) -> None:
    """Generate probability forecasts for market questions."""
    from geopolitical_prediction.markets.questions import ACTIVE_QUESTIONS, get_question
    from geopolitical_prediction.models.ensemble import EnsembleProbabilityEngine

    engine = EnsembleProbabilityEngine()

    if args.question_id:
        questions = [get_question(args.question_id)]
        questions = [q for q in questions if q is not None]
        if not questions:
            logger.error("Question not found: %s", args.question_id)
            sys.exit(1)
    else:
        questions = ACTIVE_QUESTIONS

    for q in questions:
        estimate = engine.estimate_probability(q)
        print(f"\n{'=' * 60}")
        print(f"Q: {q.question_text}")
        print(f"Probability: {estimate.probability:.1%} (confidence: {estimate.confidence:.1%})")
        print(f"Base rate: {estimate.base_rate:.1%}")
        print(f"Escalation model: {estimate.escalation_model_prob:.1%}")
        print(f"Analog model: {estimate.analog_model_prob:.1%}")
        if estimate.market_price is not None:
            print(f"Market price: {estimate.market_price:.1%}")
            print(f"Model-market divergence: {estimate.model_market_divergence:+.1%}")
        print("Reasoning:")
        for r in estimate.reasoning:
            print(f"  - {r}")


def cmd_list_questions(args: argparse.Namespace) -> None:
    """List available prediction market questions."""
    from geopolitical_prediction.markets.questions import ACTIVE_QUESTIONS

    for q in ACTIVE_QUESTIONS:
        res_date = q.resolution_date.strftime("%Y-%m-%d") if q.resolution_date else "N/A"
        print(f"  [{q.question_id}] ({q.question_type}) {q.question_text} — resolves: {res_date}")


def cmd_base_rates(args: argparse.Namespace) -> None:
    """Display historical base rates."""
    from geopolitical_prediction.features.base_rates import BaseRateEngine

    engine = BaseRateEngine()
    rates = engine.list_all_rates()
    for r in rates:
        print(
            f"  {r['event_type']}: {r['base_rate']:.1%} "
            f"(n={r['sample_size']}, {r['time_period']})"
        )
        print(f"    {r['description']}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard_path = "geopolitical_prediction/dashboard/app.py"
    subprocess.run(
        ["streamlit", "run", dashboard_path, "--server.port", str(args.port)],
        check=True,
    )


def cmd_init_db(args: argparse.Namespace) -> None:
    """Initialize the database."""
    from geopolitical_prediction.storage.database import DatabaseManager

    db = DatabaseManager()
    db.create_tables()
    logger.info("Database initialized successfully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geopolitical Prediction Markets System",
        prog="geopred",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # update-data
    p_update = subparsers.add_parser("update-data", help="Update data from sources")
    p_update.add_argument(
        "--source",
        choices=["gdelt", "acled", "all"],
        default="all",
        help="Data source to update",
    )
    p_update.add_argument("--days", type=int, default=30, help="Days of history")
    p_update.set_defaults(func=cmd_update_data)

    # forecast
    p_forecast = subparsers.add_parser("forecast", help="Generate forecasts")
    p_forecast.add_argument("--question-id", help="Specific question to forecast")
    p_forecast.set_defaults(func=cmd_forecast)

    # list-questions
    p_list = subparsers.add_parser("list-questions", help="List market questions")
    p_list.set_defaults(func=cmd_list_questions)

    # base-rates
    p_rates = subparsers.add_parser("base-rates", help="Show historical base rates")
    p_rates.set_defaults(func=cmd_base_rates)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.set_defaults(func=cmd_dashboard)

    # init-db
    p_db = subparsers.add_parser("init-db", help="Initialize database")
    p_db.set_defaults(func=cmd_init_db)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
