"""Tests for ML Prediction Module."""

from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

import numpy as np
import pytest

from code_tutor.ml.prediction.learning_predictor import LearningPredictor


class TestLearningPredictor:
    """Tests for LearningPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return LearningPredictor()

    @pytest.fixture
    def sample_daily_stats(self):
        """Create sample daily stats data."""
        return [
            {
                "problems_attempted": 5,
                "problems_solved": 4,
                "success_rate": 80.0,
                "avg_time_to_solve": 300,
                "difficulty_easy": 2,
                "difficulty_medium": 2,
                "difficulty_hard": 0,
                "categories_attempted": 3,
                "streak_days": i + 1,
                "total_study_minutes": 60,
            }
            for i in range(35)
        ]

    @pytest.fixture
    def sparse_daily_stats(self):
        """Create sparse daily stats (less than sequence length)."""
        return [
            {
                "problems_attempted": 3,
                "problems_solved": 2,
                "success_rate": 66.7,
                "avg_time_to_solve": 400,
                "difficulty_easy": 2,
                "difficulty_medium": 0,
                "difficulty_hard": 0,
                "categories_attempted": 1,
                "streak_days": i + 1,
                "total_study_minutes": 30,
            }
            for i in range(10)
        ]

    def test_predictor_initialization(self, predictor):
        """Test predictor initial state."""
        assert predictor._is_initialized is False
        assert predictor._lstm_model is None
        assert predictor._feature_scalers == {}

    def test_features_definition(self, predictor):
        """Test features are defined correctly."""
        assert len(predictor.FEATURES) == 10
        assert "problems_attempted" in predictor.FEATURES
        assert "success_rate" in predictor.FEATURES
        assert "streak_days" in predictor.FEATURES

    def test_extract_features(self, predictor, sample_daily_stats):
        """Test feature extraction from daily stats."""
        features = predictor._extract_features(sample_daily_stats)

        assert isinstance(features, np.ndarray)
        assert features.shape == (len(sample_daily_stats), len(predictor.FEATURES))
        assert features.dtype == np.float32

    def test_extract_features_missing_keys(self, predictor):
        """Test feature extraction with missing keys."""
        stats = [{"problems_attempted": 5}]  # Missing most keys

        features = predictor._extract_features(stats)

        assert features.shape == (1, len(predictor.FEATURES))
        # Missing values should default to 0
        assert features[0, 1] == 0  # problems_solved

    def test_normalize_features(self, predictor, sample_daily_stats):
        """Test feature normalization."""
        features = predictor._extract_features(sample_daily_stats)
        normalized = predictor._normalize_features(features)

        assert normalized.shape == features.shape
        # Check that scalers were created
        assert len(predictor._feature_scalers) == len(predictor.FEATURES)

        # Normalized values should have mean ~0 and std ~1
        for i in range(len(predictor.FEATURES)):
            col_mean = normalized[:, i].mean()
            col_std = normalized[:, i].std()
            assert abs(col_mean) < 0.1 or col_std < 0.1  # Allow for edge cases

    def test_denormalize_value(self, predictor, sample_daily_stats):
        """Test value denormalization."""
        features = predictor._extract_features(sample_daily_stats)
        predictor._normalize_features(features)

        # Get original mean for success_rate
        original_mean = features[:, 2].mean()

        # Denormalize 0 (which should give back mean)
        denormalized = predictor._denormalize_value(0, "success_rate")

        assert abs(denormalized - original_mean) < 1.0

    def test_denormalize_value_unknown_feature(self, predictor):
        """Test denormalization of unknown feature returns original."""
        result = predictor._denormalize_value(0.5, "unknown_feature")
        assert result == 0.5

    def test_create_sequences(self, predictor, sample_daily_stats):
        """Test sequence creation for training."""
        features = predictor._extract_features(sample_daily_stats)
        normalized = predictor._normalize_features(features)

        X, y = predictor._create_sequences(normalized)

        seq_len = predictor.config.LSTM_SEQUENCE_LENGTH
        expected_samples = len(sample_daily_stats) - seq_len

        assert X.shape == (expected_samples, seq_len, len(predictor.FEATURES))
        assert y.shape == (expected_samples,)

    def test_simple_prediction_empty_history(self, predictor):
        """Test simple prediction with empty history."""
        result = predictor._simple_prediction([], days_ahead=7)

        assert result["current_success_rate"] == 0
        assert result["predicted_success_rate"] == 50
        assert result["confidence"] == 0.3
        assert result["trend"] == "neutral"

    def test_simple_prediction_with_history(self, predictor, sparse_daily_stats):
        """Test simple prediction with some history."""
        result = predictor._simple_prediction(sparse_daily_stats, days_ahead=7)

        assert result["current_success_rate"] == sparse_daily_stats[-1]["success_rate"]
        assert "predicted_success_rate" in result
        assert result["confidence"] == 0.5
        assert result["prediction_period"] == "next_7_days"

    def test_simple_prediction_trend_detection(self, predictor):
        """Test trend detection in simple prediction."""
        # Improving trend
        improving_stats = [
            {"success_rate": 50 + i * 5, "problems_attempted": 5}
            for i in range(7)
        ]
        result = predictor._simple_prediction(improving_stats, days_ahead=7)
        assert result["trend"] == "improving"

        # Declining trend
        declining_stats = [
            {"success_rate": 80 - i * 5, "problems_attempted": 5}
            for i in range(7)
        ]
        result = predictor._simple_prediction(declining_stats, days_ahead=7)
        assert result["trend"] == "declining"

    def test_predict_success_rate_not_initialized(self, predictor, sparse_daily_stats):
        """Test prediction when not initialized returns simple prediction."""
        result = predictor.predict_success_rate(sparse_daily_stats)

        # Should fallback to simple prediction
        assert "current_success_rate" in result
        assert "predicted_success_rate" in result
        assert result["confidence"] == 0.5

    def test_predict_success_rate_sparse_history(self, predictor, sparse_daily_stats):
        """Test prediction with sparse history."""
        predictor._is_initialized = True
        predictor._lstm_model = MagicMock()

        result = predictor.predict_success_rate(sparse_daily_stats)

        # Should fallback to simple prediction due to insufficient data
        assert result["confidence"] == 0.5

    def test_calculate_confidence(self, predictor):
        """Test confidence calculation."""
        # Base case
        features = np.random.randn(30, 10)
        predictions = np.array([0.5, 0.51, 0.52])
        confidence = predictor._calculate_confidence(features, predictions)
        assert 0.3 <= confidence <= 0.95

        # High variance predictions should lower confidence
        high_var_predictions = np.array([0.1, 0.9, 0.3])
        confidence_low = predictor._calculate_confidence(features, high_var_predictions)

        # Short history should lower confidence
        short_features = np.random.randn(10, 10)
        confidence_short = predictor._calculate_confidence(short_features, predictions)
        assert confidence_short <= confidence

        # Long history should increase confidence
        long_features = np.random.randn(100, 10)
        confidence_long = predictor._calculate_confidence(long_features, predictions)
        assert confidence_long >= confidence

    def test_determine_trend(self, predictor):
        """Test trend determination."""
        # Strongly improving
        assert predictor._determine_trend([50, 55, 60, 70]) == "strongly_improving"

        # Improving
        assert predictor._determine_trend([50, 51, 52, 54]) == "improving"

        # Stable
        assert predictor._determine_trend([50, 50.1, 50, 50.2]) == "stable"

        # Declining
        assert predictor._determine_trend([60, 58, 56, 55]) == "declining"

        # Strongly declining
        assert predictor._determine_trend([80, 70, 60, 50]) == "strongly_declining"

        # Single value
        assert predictor._determine_trend([50]) == "neutral"

    def test_analyze_learning_velocity_insufficient_data(self, predictor):
        """Test velocity analysis with insufficient data."""
        result = predictor.analyze_learning_velocity([])
        assert result["velocity"] == "unknown"

        result = predictor.analyze_learning_velocity([{"success_rate": 50}] * 3)
        assert result["velocity"] == "unknown"

    def test_analyze_learning_velocity_active_user(self, predictor, sample_daily_stats):
        """Test velocity analysis for active user."""
        result = predictor.analyze_learning_velocity(sample_daily_stats)

        assert result["velocity"] in ["accelerating", "steady", "slow", "inactive"]
        assert result["problems_per_day"] > 0
        assert result["solved_per_day"] > 0
        assert 0 <= result["consistency_score"] <= 100
        assert "active_days" in result
        assert "total_days" in result

    def test_analyze_learning_velocity_categories(self, predictor):
        """Test velocity categorization."""
        # Accelerating: >= 5 problems/day and improving > 5%
        accelerating_stats = [
            {"problems_attempted": 6, "problems_solved": 5, "success_rate": 50 + i * 2}
            for i in range(10)
        ]
        result = predictor.analyze_learning_velocity(accelerating_stats)
        assert result["velocity"] == "accelerating"

        # Steady: >= 3 problems/day
        steady_stats = [
            {"problems_attempted": 4, "problems_solved": 3, "success_rate": 70}
            for _ in range(10)
        ]
        result = predictor.analyze_learning_velocity(steady_stats)
        assert result["velocity"] == "steady"

        # Slow: >= 1 problem/day
        slow_stats = [
            {"problems_attempted": 1, "problems_solved": 1, "success_rate": 80}
            for _ in range(10)
        ]
        result = predictor.analyze_learning_velocity(slow_stats)
        assert result["velocity"] == "slow"

        # Inactive: < 1 problem/day
        inactive_stats = [
            {"problems_attempted": 0, "problems_solved": 0, "success_rate": 0}
            for _ in range(10)
        ]
        result = predictor.analyze_learning_velocity(inactive_stats)
        assert result["velocity"] == "inactive"

    def test_recommend_study_schedule_high_gap(self, predictor):
        """Test study recommendations with high gap to target."""
        # Create stats with low success rate to create high gap
        low_success_stats = [
            {
                "problems_attempted": 5,
                "problems_solved": 2,
                "success_rate": 40.0,  # Low success rate
                "avg_time_to_solve": 300,
                "difficulty_easy": 2,
                "difficulty_medium": 0,
                "difficulty_hard": 0,
                "categories_attempted": 2,
                "streak_days": i + 1,
                "total_study_minutes": 30,
            }
            for i in range(10)
        ]

        result = predictor.recommend_study_schedule(low_success_stats, target_success_rate=80)

        assert "current_success_rate" in result
        assert "target_success_rate" in result
        assert "gap_to_target" in result
        assert "recommendations" in result

        # Gap should be > 20 (80 - 40 = 40)
        assert result["gap_to_target"] > 20

        # Should have intensity recommendation for high gap
        rec_types = [r["type"] for r in result["recommendations"]]
        assert "intensity" in rec_types

    def test_recommend_study_schedule_low_consistency(self, predictor):
        """Test study recommendations for inconsistent user."""
        # Inconsistent user (many inactive days)
        inconsistent_stats = []
        for i in range(30):
            if i % 3 == 0:  # Only active every 3rd day
                inconsistent_stats.append({
                    "problems_attempted": 5,
                    "problems_solved": 4,
                    "success_rate": 80,
                })
            else:
                inconsistent_stats.append({
                    "problems_attempted": 0,
                    "problems_solved": 0,
                    "success_rate": 0,
                })

        result = predictor.recommend_study_schedule(inconsistent_stats)

        rec_types = [r["type"] for r in result["recommendations"]]
        assert "consistency" in rec_types

    def test_recommend_study_schedule_inactive_user(self, predictor):
        """Test study recommendations for inactive user."""
        inactive_stats = [
            {"problems_attempted": 0, "problems_solved": 0, "success_rate": 0}
            for _ in range(10)
        ]

        result = predictor.recommend_study_schedule(inactive_stats)

        rec_types = [r["type"] for r in result["recommendations"]]
        assert "engagement" in rec_types

    def test_get_insights_insufficient_data(self, predictor):
        """Test insights with insufficient data."""
        insights = predictor.get_insights([])
        assert len(insights) == 1
        assert insights[0]["type"] == "getting_started"

        insights = predictor.get_insights([{"success_rate": 50}])
        assert len(insights) == 1
        assert insights[0]["type"] == "getting_started"

    def test_get_insights_improving_trend(self, predictor):
        """Test insights for improving user."""
        improving_stats = [
            {
                "problems_attempted": 5,
                "problems_solved": i // 2 + 2,
                "success_rate": 50 + i * 3,
            }
            for i in range(15)
        ]

        insights = predictor.get_insights(improving_stats)

        types = [i["type"] for i in insights]
        sentiments = [i.get("sentiment") for i in insights]

        assert "trend" in types
        assert "positive" in sentiments

    def test_get_insights_high_consistency(self, predictor):
        """Test insights for consistent user."""
        consistent_stats = [
            {
                "problems_attempted": 5,  # Active every day
                "problems_solved": 4,
                "success_rate": 80,
            }
            for _ in range(30)
        ]

        insights = predictor.get_insights(consistent_stats)

        types = [i["type"] for i in insights]
        assert "consistency" in types

    def test_get_insights_high_success_rate(self, predictor):
        """Test insights for high performing user."""
        high_performer_stats = [
            {
                "problems_attempted": 5,
                "problems_solved": 5,
                "success_rate": 90,
            }
            for _ in range(10)
        ]

        insights = predictor.get_insights(high_performer_stats)

        types = [i["type"] for i in insights]
        assert "achievement" in types


class TestLearningPredictorTraining:
    """Tests for LearningPredictor training functionality."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return LearningPredictor()

    def test_initialize_without_model(self, predictor):
        """Test initialization without existing model."""
        # Mock model path to not exist
        predictor.config = MagicMock()
        predictor.config.LSTM_MODEL_PATH.exists.return_value = False
        predictor.config.LSTM_HIDDEN_SIZE = 64
        predictor.config.LSTM_NUM_LAYERS = 2
        predictor.config.LSTM_SEQUENCE_LENGTH = 30

        predictor.initialize(user_histories=None)

        assert predictor._is_initialized is True

    def test_train_model_insufficient_data(self, predictor):
        """Test training with insufficient data."""
        predictor.config = MagicMock()
        predictor.config.LSTM_HIDDEN_SIZE = 64
        predictor.config.LSTM_NUM_LAYERS = 2
        predictor.config.LSTM_SEQUENCE_LENGTH = 30

        # Short histories that won't meet minimum length
        user_histories = [
            {"daily_stats": [{"success_rate": 50}] * 10}
        ]

        predictor._train_model(user_histories)

        # Model should still be created
        assert predictor._lstm_model is not None


class TestInsightsServiceMocked:
    """Tests for InsightsService with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_insights_service_initialization(self):
        """Test insights service initialization."""
        from code_tutor.ml.prediction.insights_service import InsightsService

        mock_session = AsyncMock()
        service = InsightsService(mock_session)

        assert service.session == mock_session
        assert service._cache is None

    @pytest.mark.asyncio
    async def test_get_predictor_singleton(self):
        """Test that predictor is created as singleton."""
        from code_tutor.ml.prediction.insights_service import InsightsService

        mock_session = AsyncMock()
        service = InsightsService(mock_session)

        with patch("code_tutor.ml.prediction.insights_service.LearningPredictor") as MockPredictor:
            mock_predictor = MagicMock()
            MockPredictor.return_value = mock_predictor

            # Reset global
            import code_tutor.ml.prediction.insights_service as module
            module._predictor = None

            predictor1 = service._get_predictor()
            predictor2 = service._get_predictor()

            # Should be same instance
            assert predictor1 is predictor2


class TestLSTMModelMocked:
    """Tests for LSTM Model with mocked PyTorch."""

    def test_lstm_model_initialization(self):
        """Test LSTM model initialization without PyTorch."""
        from code_tutor.ml.prediction.lstm_model import LSTMPredictor

        model = LSTMPredictor(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            sequence_length=30,
        )

        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.sequence_length == 30
        assert model._model is None  # Not loaded yet

    def test_lstm_default_output_size(self):
        """Test LSTM model default output size."""
        from code_tutor.ml.prediction.lstm_model import LSTMPredictor

        model = LSTMPredictor(input_size=10, hidden_size=64)

        assert model.output_size == 1  # Default is 1 for regression
