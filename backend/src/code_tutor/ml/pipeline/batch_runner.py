"""Batch Runner for ML Data Pipeline

CLI script for running data aggregation jobs.
"""

import asyncio
from datetime import date, timedelta
from uuid import UUID

from code_tutor.ml.pipeline.daily_stats_service import DailyStatsService
from code_tutor.ml.pipeline.data_aggregator import DataAggregator
from code_tutor.shared.infrastructure.database import get_session_context
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


async def run_daily_aggregation(target_date: date | None = None) -> dict:
    """Run daily statistics aggregation.

    Args:
        target_date: Date to aggregate (defaults to yesterday)

    Returns:
        Summary of aggregation results
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    logger.info("starting_daily_aggregation", target_date=str(target_date))

    async with get_session_context() as session:
        # Aggregate daily stats
        stats_service = DailyStatsService(session)
        stats_count = await stats_service.aggregate_daily_stats(target_date)

        # Aggregate interactions
        aggregator = DataAggregator(session)
        interactions_count = await aggregator.aggregate_user_interactions()

        # Update streaks for all active users
        user_ids = await stats_service.get_all_user_ids_with_activity()
        for user_id in user_ids:
            await stats_service.update_streak(user_id)

    results = {
        "target_date": str(target_date),
        "daily_stats_records": stats_count,
        "interactions_updated": interactions_count,
        "users_processed": len(user_ids),
    }

    logger.info("daily_aggregation_completed", **results)
    return results


async def run_backfill(days_back: int = 30, user_id: str | None = None) -> dict:
    """Backfill historical data.

    Args:
        days_back: Number of days to backfill
        user_id: Optional specific user ID

    Returns:
        Summary of backfill results
    """
    logger.info("starting_backfill", days_back=days_back, user_id=user_id)

    uid = UUID(user_id) if user_id else None

    async with get_session_context() as session:
        stats_service = DailyStatsService(session)
        stats_count = await stats_service.backfill_stats(days_back, uid)

        aggregator = DataAggregator(session)
        interactions_count = await aggregator.aggregate_user_interactions(uid)

    results = {
        "days_back": days_back,
        "user_id": user_id,
        "daily_stats_records": stats_count,
        "interactions_updated": interactions_count,
    }

    logger.info("backfill_completed", **results)
    return results


async def prepare_ncf_training_data() -> dict:
    """Prepare data for NCF model training.

    Returns:
        Training data summary
    """
    logger.info("preparing_ncf_training_data")

    async with get_session_context() as session:
        aggregator = DataAggregator(session)
        (
            problems,
            user_histories,
            interactions,
        ) = await aggregator.get_ncf_training_data()

    results = {
        "problems_count": len(problems),
        "users_count": len(user_histories),
        "interactions_count": len(interactions),
        "positive_interactions": sum(1 for _, _, solved in interactions if solved),
        "negative_interactions": sum(1 for _, _, solved in interactions if not solved),
    }

    logger.info("ncf_training_data_prepared", **results)
    return results


async def prepare_lstm_training_data(sequence_length: int = 30) -> dict:
    """Prepare sequence data for LSTM model training.

    Args:
        sequence_length: Number of days in each sequence

    Returns:
        Training data summary
    """
    logger.info("preparing_lstm_training_data", sequence_length=sequence_length)

    async with get_session_context() as session:
        stats_service = DailyStatsService(session)
        user_ids = await stats_service.get_all_user_ids_with_activity()

        sequences = []
        for user_id in user_ids:
            seq = await stats_service.get_user_stats_sequence(user_id, sequence_length)
            if len(seq) >= sequence_length:
                sequences.append(seq)

    results = {
        "users_with_activity": len(user_ids),
        "valid_sequences": len(sequences),
        "sequence_length": sequence_length,
    }

    logger.info("lstm_training_data_prepared", **results)
    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ML Data Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Daily aggregation command
    daily_parser = subparsers.add_parser("daily", help="Run daily aggregation")
    daily_parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD), defaults to yesterday",
    )

    # Backfill command
    backfill_parser = subparsers.add_parser("backfill", help="Backfill historical data")
    backfill_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill (default: 30)",
    )
    backfill_parser.add_argument(
        "--user-id",
        type=str,
        help="Specific user ID to backfill",
    )

    # NCF data prep command
    subparsers.add_parser("ncf-data", help="Prepare NCF training data")

    # LSTM data prep command
    lstm_parser = subparsers.add_parser("lstm-data", help="Prepare LSTM training data")
    lstm_parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Sequence length (default: 30)",
    )

    args = parser.parse_args()

    if args.command == "daily":
        target_date = None
        if args.date:
            target_date = date.fromisoformat(args.date)
        asyncio.run(run_daily_aggregation(target_date))

    elif args.command == "backfill":
        asyncio.run(run_backfill(args.days, args.user_id))

    elif args.command == "ncf-data":
        asyncio.run(prepare_ncf_training_data())

    elif args.command == "lstm-data":
        asyncio.run(prepare_lstm_training_data(args.sequence_length))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
