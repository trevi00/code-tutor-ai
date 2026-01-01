"""ML Data Pipeline Module

Provides data aggregation, caching, and feature extraction for ML models.
"""

from code_tutor.ml.pipeline.cache import RecommendationCache
from code_tutor.ml.pipeline.daily_stats_service import DailyStatsService
from code_tutor.ml.pipeline.data_aggregator import DataAggregator
from code_tutor.ml.pipeline.models import (
    CodeQualityAnalysisModel,
    DailyStatsModel,
    ModelTrainingLogModel,
    QualityTrendModel,
    UserInteractionModel,
)

__all__ = [
    "DailyStatsService",
    "DataAggregator",
    "RecommendationCache",
    # Models
    "CodeQualityAnalysisModel",
    "DailyStatsModel",
    "ModelTrainingLogModel",
    "QualityTrendModel",
    "UserInteractionModel",
]
