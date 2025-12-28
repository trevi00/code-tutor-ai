"""Learning Success Predictor using LSTM"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
import numpy as np
import logging

from code_tutor.ml.prediction.lstm_model import LSTMPredictor
from code_tutor.ml.config import get_ml_config

logger = logging.getLogger(__name__)


class LearningPredictor:
    """
    Predicts learning success patterns for users.

    Features:
    - Success rate prediction for next week
    - Learning velocity trend analysis
    - Optimal study time recommendations
    - Skill progression forecasting
    """

    # Feature definitions for time-series
    FEATURES = [
        "problems_attempted",
        "problems_solved",
        "success_rate",
        "avg_time_to_solve",
        "difficulty_easy",
        "difficulty_medium",
        "difficulty_hard",
        "categories_attempted",
        "streak_days",
        "total_study_minutes"
    ]

    def __init__(self, config=None):
        self.config = config or get_ml_config()

        self._lstm_model: Optional[LSTMPredictor] = None
        self._is_initialized = False
        self._feature_scalers: Dict[str, tuple] = {}  # (mean, std) for normalization

    def initialize(
        self,
        user_histories: Optional[List[Dict]] = None,
        force_retrain: bool = False
    ):
        """
        Initialize predictor with historical data.

        Args:
            user_histories: List of user history dicts with daily stats
            force_retrain: Whether to retrain model
        """
        logger.info("Initializing learning predictor...")

        model_path = self.config.LSTM_MODEL_PATH
        if model_path.exists() and not force_retrain:
            try:
                self._load_model()
                self._is_initialized = True
                logger.info("Loaded pre-trained LSTM model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

        if user_histories:
            self._train_model(user_histories)

        self._is_initialized = True

    def _load_model(self):
        """Load pre-trained model"""
        self._lstm_model = LSTMPredictor(
            input_size=len(self.FEATURES),
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_NUM_LAYERS,
            sequence_length=self.config.LSTM_SEQUENCE_LENGTH
        )
        self._lstm_model.load(self.config.LSTM_MODEL_PATH)

    def _train_model(self, user_histories: List[Dict]):
        """Train LSTM model on user histories"""
        logger.info("Training LSTM model...")

        # Prepare training data
        all_sequences = []
        all_targets = []

        for history in user_histories:
            daily_data = history.get("daily_stats", [])
            if len(daily_data) < self.config.LSTM_SEQUENCE_LENGTH + 7:
                continue

            # Extract features
            features = self._extract_features(daily_data)

            # Normalize
            features = self._normalize_features(features)

            # Create sequences
            X, y = self._create_sequences(features)
            all_sequences.append(X)
            all_targets.append(y)

        if not all_sequences:
            logger.warning("Not enough data for training")
            self._lstm_model = LSTMPredictor(
                input_size=len(self.FEATURES),
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                sequence_length=self.config.LSTM_SEQUENCE_LENGTH
            )
            return

        X_train = np.vstack(all_sequences)
        y_train = np.concatenate(all_targets)

        # Create and train model
        self._lstm_model = LSTMPredictor(
            input_size=len(self.FEATURES),
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_NUM_LAYERS,
            sequence_length=self.config.LSTM_SEQUENCE_LENGTH
        )

        history = self._lstm_model.train(
            X_train, y_train,
            epochs=50,
            batch_size=32
        )

        # Save model
        self._lstm_model.save(self.config.LSTM_MODEL_PATH)

        logger.info(f"LSTM model trained. Final loss: {history['train_loss'][-1]:.4f}")

    def _extract_features(self, daily_stats: List[Dict]) -> np.ndarray:
        """Extract features from daily stats"""
        features = []

        for day in daily_stats:
            day_features = [
                day.get("problems_attempted", 0),
                day.get("problems_solved", 0),
                day.get("success_rate", 0),
                day.get("avg_time_to_solve", 0),
                day.get("difficulty_easy", 0),
                day.get("difficulty_medium", 0),
                day.get("difficulty_hard", 0),
                day.get("categories_attempted", 0),
                day.get("streak_days", 0),
                day.get("total_study_minutes", 0)
            ]
            features.append(day_features)

        return np.array(features, dtype=np.float32)

    def _normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using z-score"""
        normalized = features.copy()

        for i, name in enumerate(self.FEATURES):
            if fit or name not in self._feature_scalers:
                mean = features[:, i].mean()
                std = features[:, i].std() + 1e-8
                self._feature_scalers[name] = (mean, std)
            else:
                mean, std = self._feature_scalers[name]

            normalized[:, i] = (features[:, i] - mean) / std

        return normalized

    def _denormalize_value(self, value: float, feature_name: str) -> float:
        """Denormalize a single value"""
        if feature_name in self._feature_scalers:
            mean, std = self._feature_scalers[feature_name]
            return value * std + mean
        return value

    def _create_sequences(self, features: np.ndarray) -> tuple:
        """Create sequences for training"""
        seq_len = self.config.LSTM_SEQUENCE_LENGTH
        X, y = [], []

        # Target: success_rate (index 2)
        target_col = 2

        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(features[i + seq_len, target_col])

        return np.array(X), np.array(y)

    def predict_success_rate(
        self,
        user_history: List[Dict],
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predict future success rate.

        Args:
            user_history: User's daily stats history
            days_ahead: Number of days to predict

        Returns:
            Prediction results with confidence
        """
        if not self._is_initialized or self._lstm_model is None:
            return self._simple_prediction(user_history, days_ahead)

        # Extract and normalize features
        features = self._extract_features(user_history)

        if len(features) < self.config.LSTM_SEQUENCE_LENGTH:
            return self._simple_prediction(user_history, days_ahead)

        features = self._normalize_features(features, fit=False)

        # Get last sequence
        sequence = features[-self.config.LSTM_SEQUENCE_LENGTH:]

        # Predict
        predictions = self._lstm_model.predict_next(sequence, steps=days_ahead)

        # Denormalize predictions
        denormalized = [
            self._denormalize_value(p, "success_rate")
            for p in predictions
        ]

        # Clip to valid range
        denormalized = np.clip(denormalized, 0, 100)

        # Calculate confidence based on prediction variance
        confidence = self._calculate_confidence(features, predictions)

        return {
            "current_success_rate": float(user_history[-1].get("success_rate", 0)),
            "predicted_success_rate": float(np.mean(denormalized)),
            "daily_predictions": [float(p) for p in denormalized],
            "prediction_period": f"next_{days_ahead}_days",
            "confidence": confidence,
            "trend": self._determine_trend(denormalized)
        }

    def _simple_prediction(
        self,
        user_history: List[Dict],
        days_ahead: int
    ) -> Dict[str, Any]:
        """Simple fallback prediction without LSTM"""
        if not user_history:
            return {
                "current_success_rate": 0,
                "predicted_success_rate": 50,
                "prediction_period": f"next_{days_ahead}_days",
                "confidence": 0.3,
                "trend": "neutral"
            }

        recent = user_history[-min(7, len(user_history)):]
        current_rate = user_history[-1].get("success_rate", 0)

        # Simple moving average prediction
        recent_rates = [d.get("success_rate", 0) for d in recent]
        avg_rate = np.mean(recent_rates)

        # Calculate trend
        if len(recent_rates) >= 3:
            trend_slope = (recent_rates[-1] - recent_rates[0]) / len(recent_rates)
            predicted = avg_rate + trend_slope * days_ahead
        else:
            predicted = avg_rate

        predicted = np.clip(predicted, 0, 100)

        return {
            "current_success_rate": float(current_rate),
            "predicted_success_rate": float(predicted),
            "prediction_period": f"next_{days_ahead}_days",
            "confidence": 0.5,
            "trend": "improving" if predicted > current_rate else "declining"
        }

    def _calculate_confidence(
        self,
        features: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """Calculate prediction confidence"""
        # Base confidence
        confidence = 0.7

        # Reduce confidence for high variance in predictions
        pred_std = np.std(predictions)
        if pred_std > 0.5:
            confidence -= 0.1

        # Reduce confidence for sparse history
        if len(features) < 30:
            confidence -= 0.1
        elif len(features) > 90:
            confidence += 0.1

        return max(0.3, min(0.95, confidence))

    def _determine_trend(self, predictions: List[float]) -> str:
        """Determine trend from predictions"""
        if len(predictions) < 2:
            return "neutral"

        slope = (predictions[-1] - predictions[0]) / len(predictions)

        if slope > 2:
            return "strongly_improving"
        elif slope > 0.5:
            return "improving"
        elif slope < -2:
            return "strongly_declining"
        elif slope < -0.5:
            return "declining"
        else:
            return "stable"

    def analyze_learning_velocity(
        self,
        user_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze learning velocity and trends.

        Args:
            user_history: User's daily stats

        Returns:
            Learning velocity analysis
        """
        if not user_history or len(user_history) < 7:
            return {
                "velocity": "unknown",
                "problems_per_day": 0,
                "improvement_rate": 0,
                "consistency_score": 0
            }

        recent = user_history[-30:] if len(user_history) >= 30 else user_history

        # Calculate metrics
        problems_per_day = np.mean([d.get("problems_attempted", 0) for d in recent])
        solved_per_day = np.mean([d.get("problems_solved", 0) for d in recent])

        # Calculate improvement rate
        first_week = recent[:7] if len(recent) >= 7 else recent
        last_week = recent[-7:]

        first_rate = np.mean([d.get("success_rate", 0) for d in first_week])
        last_rate = np.mean([d.get("success_rate", 0) for d in last_week])
        improvement_rate = last_rate - first_rate

        # Calculate consistency
        active_days = sum(1 for d in recent if d.get("problems_attempted", 0) > 0)
        consistency_score = active_days / len(recent) * 100

        # Determine velocity category
        if problems_per_day >= 5 and improvement_rate > 5:
            velocity = "accelerating"
        elif problems_per_day >= 3:
            velocity = "steady"
        elif problems_per_day >= 1:
            velocity = "slow"
        else:
            velocity = "inactive"

        return {
            "velocity": velocity,
            "problems_per_day": float(problems_per_day),
            "solved_per_day": float(solved_per_day),
            "improvement_rate": float(improvement_rate),
            "consistency_score": float(consistency_score),
            "active_days": active_days,
            "total_days": len(recent)
        }

    def recommend_study_schedule(
        self,
        user_history: List[Dict],
        target_success_rate: float = 80
    ) -> Dict[str, Any]:
        """
        Recommend optimal study schedule.

        Args:
            user_history: User's daily stats
            target_success_rate: Target success rate

        Returns:
            Study schedule recommendations
        """
        velocity = self.analyze_learning_velocity(user_history)
        prediction = self.predict_success_rate(user_history)

        current_rate = prediction["current_success_rate"]
        predicted_rate = prediction["predicted_success_rate"]

        recommendations = []

        # Calculate required effort
        gap = target_success_rate - current_rate

        if gap > 20:
            recommendations.append({
                "type": "intensity",
                "message": "목표 달성을 위해 학습 강도를 높이세요.",
                "suggested_problems_per_day": max(5, velocity["problems_per_day"] + 2)
            })

        if velocity["consistency_score"] < 50:
            recommendations.append({
                "type": "consistency",
                "message": "일관된 학습이 중요합니다. 매일 조금씩 공부하세요.",
                "suggested_active_days": "5-7일/주"
            })

        if velocity["velocity"] == "inactive":
            recommendations.append({
                "type": "engagement",
                "message": "학습을 다시 시작하세요. 쉬운 문제부터 시작하는 것을 권장합니다.",
                "suggested_difficulty": "easy"
            })

        # Estimate days to reach target
        if velocity["improvement_rate"] > 0:
            days_to_target = int(gap / velocity["improvement_rate"])
            if days_to_target > 0 and days_to_target < 365:
                recommendations.append({
                    "type": "timeline",
                    "message": f"현재 속도로 약 {days_to_target}일 후 목표 달성 예상",
                    "estimated_days": days_to_target
                })

        return {
            "current_success_rate": current_rate,
            "target_success_rate": target_success_rate,
            "predicted_success_rate": predicted_rate,
            "gap_to_target": gap,
            "velocity": velocity["velocity"],
            "recommendations": recommendations
        }

    def get_insights(self, user_history: List[Dict]) -> List[Dict]:
        """
        Generate learning insights.

        Args:
            user_history: User's daily stats

        Returns:
            List of insight dicts
        """
        insights = []

        if not user_history or len(user_history) < 3:
            insights.append({
                "type": "getting_started",
                "message": "더 많은 문제를 풀어보세요. 충분한 데이터가 쌓이면 맞춤 분석을 제공합니다."
            })
            return insights

        velocity = self.analyze_learning_velocity(user_history)
        prediction = self.predict_success_rate(user_history)

        # Trend insight
        trend = prediction.get("trend", "neutral")
        if trend in ["strongly_improving", "improving"]:
            insights.append({
                "type": "trend",
                "message": "실력이 꾸준히 향상되고 있습니다! 이 페이스를 유지하세요.",
                "sentiment": "positive"
            })
        elif trend in ["strongly_declining", "declining"]:
            insights.append({
                "type": "trend",
                "message": "최근 성공률이 다소 하락했습니다. 어려운 문제에 도전하고 있다면 정상입니다.",
                "sentiment": "neutral"
            })

        # Velocity insight
        if velocity["velocity"] == "accelerating":
            insights.append({
                "type": "velocity",
                "message": "학습 속도가 빨라지고 있습니다. 훌륭합니다!",
                "sentiment": "positive"
            })

        # Consistency insight
        if velocity["consistency_score"] >= 80:
            insights.append({
                "type": "consistency",
                "message": f"지난 기간 동안 {velocity['consistency_score']:.0f}%의 일관성을 보였습니다. 꾸준함이 성공의 열쇠입니다!",
                "sentiment": "positive"
            })
        elif velocity["consistency_score"] < 40:
            insights.append({
                "type": "consistency",
                "message": "학습 일관성을 높이면 더 빠른 실력 향상이 가능합니다.",
                "sentiment": "suggestion"
            })

        # Achievement insight
        if prediction["current_success_rate"] >= 80:
            insights.append({
                "type": "achievement",
                "message": "80% 이상의 높은 성공률을 유지하고 있습니다. 더 어려운 문제에 도전해보세요!",
                "sentiment": "positive"
            })

        return insights
