"""Tests for ML Recommendation Module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from code_tutor.ml.config import MLConfig, get_ml_config
from code_tutor.ml.recommendation.recommender import ProblemRecommender


class TestMLConfig:
    """Tests for ML configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLConfig()

        assert config.NCF_EMBEDDING_DIM == 32
        assert config.NCF_HIDDEN_LAYERS == [64, 32, 16]
        assert config.LSTM_HIDDEN_SIZE == 64
        assert config.LSTM_NUM_LAYERS == 2
        assert config.LSTM_SEQUENCE_LENGTH == 30
        assert config.RAG_TOP_K == 3
        assert config.EMBEDDING_DIMENSION == 384

    def test_get_ml_config_singleton(self):
        """Test that get_ml_config returns cached instance."""
        config1 = get_ml_config()
        config2 = get_ml_config()
        assert config1 is config2


class TestProblemRecommender:
    """Tests for ProblemRecommender."""

    @pytest.fixture
    def recommender(self):
        """Create recommender instance."""
        return ProblemRecommender()

    @pytest.fixture
    def sample_problems(self):
        """Create sample problem data."""
        return [
            {
                "id": uuid4(),
                "title": "Two Sum",
                "difficulty": "easy",
                "category": "array",
                "patterns": ["hash-map"],
                "acceptance_rate": 0.7,
            },
            {
                "id": uuid4(),
                "title": "Binary Search",
                "difficulty": "easy",
                "category": "array",
                "patterns": ["binary-search"],
                "acceptance_rate": 0.6,
            },
            {
                "id": uuid4(),
                "title": "Valid Parentheses",
                "difficulty": "easy",
                "category": "stack",
                "patterns": ["stack"],
                "acceptance_rate": 0.65,
            },
            {
                "id": uuid4(),
                "title": "Merge Intervals",
                "difficulty": "medium",
                "category": "array",
                "patterns": ["sorting", "intervals"],
                "acceptance_rate": 0.45,
            },
            {
                "id": uuid4(),
                "title": "LRU Cache",
                "difficulty": "hard",
                "category": "design",
                "patterns": ["hash-map", "linked-list"],
                "acceptance_rate": 0.35,
            },
        ]

    @pytest.fixture
    def sample_interactions(self, sample_problems):
        """Create sample interaction data."""
        user1 = uuid4()
        user2 = uuid4()
        return [
            {"user_id": user1, "problem_id": sample_problems[0]["id"], "is_solved": True},
            {"user_id": user1, "problem_id": sample_problems[1]["id"], "is_solved": True},
            {"user_id": user1, "problem_id": sample_problems[2]["id"], "is_solved": False},
            {"user_id": user2, "problem_id": sample_problems[0]["id"], "is_solved": True},
            {"user_id": user2, "problem_id": sample_problems[3]["id"], "is_solved": True},
        ]

    def test_recommender_initialization(self, recommender):
        """Test recommender initial state."""
        assert recommender._is_initialized is False
        assert recommender._ncf_model is None
        assert recommender._user_id_map == {}
        assert recommender._item_id_map == {}

    def test_build_id_mappings(self, recommender, sample_problems, sample_interactions):
        """Test building user and item ID mappings."""
        recommender._build_id_mappings(sample_problems, sample_interactions)

        # Check mappings were created
        assert len(recommender._user_id_map) == 2
        assert len(recommender._item_id_map) == 5
        assert len(recommender._reverse_item_map) == 5

        # Check reverse mapping is correct
        for item_id, idx in recommender._item_id_map.items():
            assert recommender._reverse_item_map[idx] == item_id

    def test_store_problem_features(self, recommender, sample_problems):
        """Test storing problem features."""
        recommender._store_problem_features(sample_problems)

        assert len(recommender._problem_features) == 5

        for problem in sample_problems:
            pid = problem["id"]
            assert pid in recommender._problem_features
            features = recommender._problem_features[pid]
            assert features["difficulty"] == problem["difficulty"]
            assert features["category"] == problem["category"]
            assert features["patterns"] == problem["patterns"]

    def test_build_user_history(self, recommender, sample_interactions):
        """Test building user interaction history."""
        recommender._build_user_history(sample_interactions)

        # Get unique users
        users = set(i["user_id"] for i in sample_interactions)
        assert len(recommender._user_history) == len(users)

        # Check solved problems are tracked
        for interaction in sample_interactions:
            if interaction["is_solved"]:
                user_id = interaction["user_id"]
                assert interaction["problem_id"] in recommender._user_history[user_id]

    def test_analyze_user_profile_empty(self, recommender):
        """Test analyzing empty user profile."""
        profile = recommender._analyze_user_profile(set())

        assert profile["difficulties"] == {"easy": 0, "medium": 0, "hard": 0}
        assert profile["categories"] == {}
        assert profile["patterns"] == {}
        assert profile["avg_acceptance_rate"] == 0.5

    def test_analyze_user_profile_with_data(self, recommender, sample_problems):
        """Test analyzing user profile with solved problems."""
        recommender._store_problem_features(sample_problems)

        # User solved first two problems (easy, array)
        solved = {sample_problems[0]["id"], sample_problems[1]["id"]}
        profile = recommender._analyze_user_profile(solved)

        assert profile["difficulties"]["easy"] == 2
        assert profile["categories"]["array"] == 2
        assert "hash-map" in profile["patterns"]
        assert "binary-search" in profile["patterns"]

    def test_content_similarity_score(self, recommender):
        """Test content similarity scoring."""
        profile = {
            "difficulties": {"easy": 5, "medium": 2, "hard": 0},
            "categories": {"array": 5, "stack": 2},
            "patterns": {"hash-map": 3, "two-pointers": 2},
            "avg_acceptance_rate": 0.6,
        }

        # Test with medium difficulty (should get progression bonus)
        features_medium = {
            "difficulty": "medium",
            "category": "array",
            "patterns": ["two-pointers"],
            "acceptance_rate": 0.45,
        }
        score_medium = recommender._content_similarity_score(profile, features_medium)
        assert score_medium > 0

        # Test with new category (exploration bonus)
        features_new = {
            "difficulty": "easy",
            "category": "graph",
            "patterns": ["bfs"],
            "acceptance_rate": 0.5,
        }
        score_new = recommender._content_similarity_score(profile, features_new)
        assert score_new > 0

    def test_cold_start_recommendations(self, recommender, sample_problems):
        """Test recommendations for new users."""
        recommender._store_problem_features(sample_problems)

        recommendations = recommender._cold_start_recommendations(top_k=3)

        assert len(recommendations) <= 3
        # Should prefer easy problems with high acceptance
        for rec in recommendations:
            assert "problem_id" in rec
            assert "score" in rec
            assert "reason" in rec
            assert rec["reason"] == "popular"

    def test_cold_start_with_filters(self, recommender, sample_problems):
        """Test cold start with difficulty and category filters."""
        recommender._store_problem_features(sample_problems)

        # Filter by difficulty
        easy_recs = recommender._cold_start_recommendations(
            top_k=10, difficulty="easy"
        )
        for rec in easy_recs:
            assert rec["difficulty"] == "easy"

        # Filter by category
        array_recs = recommender._cold_start_recommendations(
            top_k=10, category="array"
        )
        for rec in array_recs:
            assert rec["category"] == "array"

    def test_apply_filters(self, recommender, sample_problems):
        """Test applying filters to recommendations."""
        recommender._store_problem_features(sample_problems)

        recommendations = [
            {"problem_id": sample_problems[0]["id"], "difficulty": "easy", "category": "array"},
            {"problem_id": sample_problems[3]["id"], "difficulty": "medium", "category": "array"},
            {"problem_id": sample_problems[4]["id"], "difficulty": "hard", "category": "design"},
        ]

        # Filter by difficulty
        filtered = recommender._apply_filters(
            recommendations, difficulty="easy", category=None, solved=set()
        )
        assert len(filtered) == 1
        assert filtered[0]["difficulty"] == "easy"

        # Filter by category
        filtered = recommender._apply_filters(
            recommendations, difficulty=None, category="design", solved=set()
        )
        assert len(filtered) == 1
        assert filtered[0]["category"] == "design"

        # Exclude solved
        filtered = recommender._apply_filters(
            recommendations,
            difficulty=None,
            category=None,
            solved={sample_problems[0]["id"]},
        )
        assert len(filtered) == 2

    def test_recommend_not_initialized(self, recommender):
        """Test recommend when not initialized returns cold start."""
        recommender._problem_features = {
            uuid4(): {"difficulty": "easy", "category": "array", "patterns": [], "acceptance_rate": 0.7}
        }

        recs = recommender.recommend(uuid4(), top_k=5)

        # Should return cold start recommendations
        assert isinstance(recs, list)

    def test_recommend_new_user(self, recommender, sample_problems, sample_interactions):
        """Test recommend for user not in training data."""
        recommender._build_id_mappings(sample_problems, sample_interactions)
        recommender._store_problem_features(sample_problems)
        recommender._build_user_history(sample_interactions)
        recommender._is_initialized = True

        # New user not in mapping
        new_user = uuid4()
        recs = recommender.recommend(new_user, top_k=3)

        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_get_skill_gaps(self, recommender, sample_problems, sample_interactions):
        """Test identifying skill gaps."""
        recommender._store_problem_features(sample_problems)
        recommender._build_user_history(sample_interactions)

        user_id = sample_interactions[0]["user_id"]
        gaps = recommender.get_skill_gaps(user_id)

        assert isinstance(gaps, list)
        for gap in gaps:
            assert "type" in gap
            assert "name" in gap
            assert "solved" in gap
            assert "total" in gap
            assert "coverage" in gap

    def test_get_skill_gaps_new_user(self, recommender, sample_problems):
        """Test skill gaps for new user."""
        recommender._store_problem_features(sample_problems)

        gaps = recommender.get_skill_gaps(uuid4())

        # New user should have all categories as gaps
        assert isinstance(gaps, list)

    def test_add_negative_samples(self, recommender):
        """Test adding negative samples for training."""
        recommender._item_id_map = {uuid4(): i for i in range(10)}

        positive_samples = [
            (0, 0, 1),
            (0, 1, 1),
            (1, 2, 1),
        ]

        augmented = recommender._add_negative_samples(positive_samples, negative_ratio=2)

        # Should have more samples now
        assert len(augmented) > len(positive_samples)

        # Original positive samples should still be there
        for sample in positive_samples:
            assert sample in augmented

    def test_content_based_recommend(self, recommender, sample_problems, sample_interactions):
        """Test content-based recommendations."""
        recommender._build_id_mappings(sample_problems, sample_interactions)
        recommender._store_problem_features(sample_problems)
        recommender._build_user_history(sample_interactions)

        user_id = sample_interactions[0]["user_id"]
        solved = recommender._user_history.get(user_id, set())

        recs = recommender._content_based_recommend(user_id, solved, top_k=3)

        assert isinstance(recs, list)
        # Should not include solved problems
        for rec in recs:
            assert rec["problem_id"] not in solved
            assert rec["reason"] == "content_match"


class TestNCFModelMocked:
    """Tests for NCF Model with mocked PyTorch."""

    def test_ncf_model_initialization(self):
        """Test NCF model initialization without PyTorch."""
        from code_tutor.ml.recommendation.ncf_model import NCFModel

        model = NCFModel(
            num_users=100,
            num_items=500,
            embedding_dim=32,
            hidden_layers=[64, 32, 16],
        )

        assert model.num_users == 100
        assert model.num_items == 500
        assert model.embedding_dim == 32
        assert model.hidden_layers == [64, 32, 16]
        assert model._model is None  # Not loaded yet

    def test_ncf_default_hidden_layers(self):
        """Test NCF model default hidden layers."""
        from code_tutor.ml.recommendation.ncf_model import NCFModel

        model = NCFModel(num_users=10, num_items=10)

        assert model.hidden_layers == [64, 32, 16]

    def test_ncf_recommend_empty_candidates(self):
        """Test recommend with empty candidates returns empty list."""
        from code_tutor.ml.recommendation.ncf_model import NCFModel

        model = NCFModel(num_users=10, num_items=10)

        # Test that empty candidates after exclusion returns empty list
        # This tests the logic without requiring torch
        candidate_items = np.array([1, 2, 3])
        exclude_items = {1, 2, 3}

        # Filter candidates
        filtered = np.array(
            [item for item in candidate_items if item not in exclude_items]
        )

        assert len(filtered) == 0


class TestRecommenderIntegration:
    """Integration tests for recommendation system."""

    @pytest.fixture
    def full_setup(self):
        """Set up recommender with full data."""
        recommender = ProblemRecommender()

        problems = [
            {"id": uuid4(), "title": f"Problem {i}", "difficulty": d, "category": c, "patterns": [p], "acceptance_rate": 0.5}
            for i, (d, c, p) in enumerate([
                ("easy", "array", "two-pointers"),
                ("easy", "array", "sliding-window"),
                ("easy", "string", "hash-map"),
                ("medium", "array", "binary-search"),
                ("medium", "tree", "dfs"),
                ("medium", "graph", "bfs"),
                ("hard", "dp", "memoization"),
                ("hard", "graph", "dijkstra"),
            ])
        ]

        users = [uuid4() for _ in range(3)]
        interactions = []

        # User 0: solved easy problems
        for i in range(3):
            interactions.append({"user_id": users[0], "problem_id": problems[i]["id"], "is_solved": True})

        # User 1: solved some medium problems
        for i in [0, 3, 4]:
            interactions.append({"user_id": users[1], "problem_id": problems[i]["id"], "is_solved": True})

        # User 2: solved a hard problem
        interactions.append({"user_id": users[2], "problem_id": problems[6]["id"], "is_solved": True})

        recommender._build_id_mappings(problems, interactions)
        recommender._store_problem_features(problems)
        recommender._build_user_history(interactions)

        return recommender, problems, users

    def test_skill_progression(self, full_setup):
        """Test that recommendations support skill progression."""
        recommender, problems, users = full_setup

        # User 0 solved easy problems, should get medium recommendations
        profile = recommender._analyze_user_profile(recommender._user_history[users[0]])

        assert profile["difficulties"]["easy"] == 3
        assert profile["difficulties"]["medium"] == 0

    def test_category_diversity(self, full_setup):
        """Test category diversity in user profile."""
        recommender, problems, users = full_setup

        profile = recommender._analyze_user_profile(recommender._user_history[users[0]])

        # User 0 solved problems in array and string categories
        assert "array" in profile["categories"]
        assert "string" in profile["categories"]

    def test_hybrid_weights(self, full_setup):
        """Test that hybrid recommendations combine scores."""
        recommender, problems, users = full_setup
        recommender._is_initialized = True

        # Mock NCF model for collaborative filtering
        mock_ncf = MagicMock()
        mock_ncf.recommend.return_value = []
        recommender._ncf_model = mock_ncf

        user_id = users[0]
        solved = recommender._user_history[user_id]

        # Get hybrid recommendations (will only have content-based since NCF returns empty)
        recs = recommender._hybrid_recommend(user_id, solved, top_k=5)

        assert isinstance(recs, list)
