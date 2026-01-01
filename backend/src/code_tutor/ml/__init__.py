"""Machine Learning module for Code Tutor AI

Provides:
- RAG (Retrieval-Augmented Generation) with FAISS and 25 algorithm patterns
- NCF (Neural Collaborative Filtering) for problem recommendation
- LSTM for learning success prediction
- CodeBERT for code analysis and pattern detection
"""

from code_tutor.ml.config import MLConfig, get_ml_config


# Lazy imports to avoid loading heavy models at startup
def get_rag_engine():
    """Get RAG engine singleton"""
    from code_tutor.ml.rag import RAGEngine

    return RAGEngine()


def get_recommender():
    """Get problem recommender singleton"""
    from code_tutor.ml.recommendation import ProblemRecommender

    return ProblemRecommender()


def get_learning_predictor():
    """Get learning predictor singleton"""
    from code_tutor.ml.prediction import LearningPredictor

    return LearningPredictor()


def get_code_analyzer():
    """Get code analyzer singleton"""
    from code_tutor.ml.analysis import CodeAnalyzer

    return CodeAnalyzer()


def get_code_classifier():
    """Get Transformer-based code quality classifier singleton"""
    from code_tutor.ml.analysis import CodeQualityClassifier

    return CodeQualityClassifier()


__all__ = [
    "MLConfig",
    "get_ml_config",
    "get_rag_engine",
    "get_recommender",
    "get_learning_predictor",
    "get_code_analyzer",
    "get_code_classifier",
]
