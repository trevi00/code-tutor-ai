"""Problem Recommender using NCF and content-based features"""

import logging
from uuid import UUID

import numpy as np

from code_tutor.ml.config import get_ml_config
from code_tutor.ml.recommendation.ncf_model import NCFModel

logger = logging.getLogger(__name__)


class ProblemRecommender:
    """
    Problem recommendation system combining:
    - NCF (Neural Collaborative Filtering) for collaborative signals
    - Content-based features (difficulty, category, patterns)
    - Cold-start handling for new users

    Recommendation strategies:
    - Similar users' solved problems
    - Next difficulty level progression
    - Category/pattern completion
    - Skill gap identification
    """

    def __init__(self, config=None):
        self.config = config or get_ml_config()

        self._ncf_model: NCFModel | None = None
        self._user_id_map: dict[UUID, int] = {}
        self._item_id_map: dict[UUID, int] = {}
        self._reverse_item_map: dict[int, UUID] = {}

        self._problem_features: dict[UUID, dict] = {}
        self._user_history: dict[UUID, set[UUID]] = {}

        self._is_initialized = False

    def initialize(
        self,
        problems: list[dict],
        interactions: list[dict],
        force_retrain: bool = False,
    ):
        """
        Initialize recommender with problem and interaction data.

        Args:
            problems: List of problem dicts with id, difficulty, category, patterns
            interactions: List of interaction dicts with user_id, problem_id, is_solved
            force_retrain: Whether to retrain model even if exists
        """
        logger.info("Initializing problem recommender...")

        # Build ID mappings
        self._build_id_mappings(problems, interactions)

        # Store problem features
        self._store_problem_features(problems)

        # Build user history
        self._build_user_history(interactions)

        # Initialize or load NCF model
        model_path = self.config.NCF_MODEL_PATH
        if model_path.exists() and not force_retrain:
            try:
                self._load_model()
                self._is_initialized = True
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, retraining...")

        # Train new model
        self._train_model(interactions)
        self._is_initialized = True

    def _build_id_mappings(self, problems: list[dict], interactions: list[dict]):
        """Build user and item ID mappings"""
        # Collect unique IDs
        user_ids = set()
        item_ids = set()

        for interaction in interactions:
            user_ids.add(interaction["user_id"])
            item_ids.add(interaction["problem_id"])

        for problem in problems:
            item_ids.add(problem["id"])

        # Create mappings
        self._user_id_map = {
            uid: idx for idx, uid in enumerate(sorted(user_ids, key=str))
        }
        self._item_id_map = {
            iid: idx for idx, iid in enumerate(sorted(item_ids, key=str))
        }
        self._reverse_item_map = {idx: iid for iid, idx in self._item_id_map.items()}

        logger.info(
            f"Built mappings: {len(self._user_id_map)} users, {len(self._item_id_map)} items"
        )

    def _store_problem_features(self, problems: list[dict]):
        """Store problem features for content-based filtering"""
        for problem in problems:
            self._problem_features[problem["id"]] = {
                "difficulty": problem.get("difficulty", "medium"),
                "category": problem.get("category", "unknown"),
                "patterns": problem.get("patterns", []),
                "title": problem.get("title", ""),
                "acceptance_rate": problem.get("acceptance_rate", 0.5),
            }

    def _build_user_history(self, interactions: list[dict]):
        """Build user interaction history"""
        self._user_history = {}

        for interaction in interactions:
            user_id = interaction["user_id"]
            if user_id not in self._user_history:
                self._user_history[user_id] = set()

            if interaction.get("is_solved", False):
                self._user_history[user_id].add(interaction["problem_id"])

    def _train_model(self, interactions: list[dict]):
        """Train NCF model on interaction data"""
        logger.info("Training NCF model...")

        # Prepare training data
        train_data = []
        for interaction in interactions:
            user_idx = self._user_id_map.get(interaction["user_id"])
            item_idx = self._item_id_map.get(interaction["problem_id"])

            if user_idx is not None and item_idx is not None:
                label = 1 if interaction.get("is_solved", False) else 0
                train_data.append((user_idx, item_idx, label))

        if not train_data:
            logger.warning("No training data available")
            return

        # Add negative samples
        train_data = self._add_negative_samples(train_data)

        # Split train/val
        np.random.shuffle(train_data)
        split_idx = int(len(train_data) * 0.9)
        train_split = train_data[:split_idx]
        val_split = train_data[split_idx:]

        # Create and train model
        self._ncf_model = NCFModel(
            num_users=len(self._user_id_map),
            num_items=len(self._item_id_map),
            embedding_dim=self.config.NCF_EMBEDDING_DIM,
            hidden_layers=self.config.NCF_HIDDEN_LAYERS,
        )

        history = self._ncf_model.train(
            train_data=train_split,
            val_data=val_split if val_split else None,
            epochs=20,
            batch_size=256,
        )

        # Save model
        self._ncf_model.save(self.config.NCF_MODEL_PATH)

        logger.info(f"NCF model trained. Final loss: {history['train_loss'][-1]:.4f}")

    def _add_negative_samples(
        self, positive_samples: list, negative_ratio: int = 4
    ) -> list:
        """Add negative samples for training"""
        user_positive_items = {}
        for user_idx, item_idx, _ in positive_samples:
            if user_idx not in user_positive_items:
                user_positive_items[user_idx] = set()
            user_positive_items[user_idx].add(item_idx)

        all_items = set(range(len(self._item_id_map)))
        augmented = list(positive_samples)

        for user_idx, item_idx, _ in positive_samples:
            negative_items = list(all_items - user_positive_items.get(user_idx, set()))

            n_neg = min(negative_ratio, len(negative_items))
            if n_neg > 0:
                neg_samples = np.random.choice(negative_items, n_neg, replace=False)
                for neg_item in neg_samples:
                    augmented.append((user_idx, neg_item, 0))

        return augmented

    def _load_model(self):
        """Load pre-trained NCF model"""
        self._ncf_model = NCFModel(
            num_users=len(self._user_id_map),
            num_items=len(self._item_id_map),
            embedding_dim=self.config.NCF_EMBEDDING_DIM,
            hidden_layers=self.config.NCF_HIDDEN_LAYERS,
        )
        self._ncf_model.load(self.config.NCF_MODEL_PATH)
        logger.info("Loaded pre-trained NCF model")

    def recommend(
        self,
        user_id: UUID,
        top_k: int = 10,
        strategy: str = "hybrid",
        difficulty_filter: str | None = None,
        category_filter: str | None = None,
    ) -> list[dict]:
        """
        Recommend problems for a user.

        Args:
            user_id: User UUID
            top_k: Number of recommendations
            strategy: "collaborative", "content", "hybrid"
            difficulty_filter: Filter by difficulty level
            category_filter: Filter by category

        Returns:
            List of recommended problems with scores
        """
        if not self._is_initialized:
            logger.warning("Recommender not initialized")
            return self._cold_start_recommendations(
                top_k, difficulty_filter, category_filter
            )

        # Get user's solved problems
        solved_problems = self._user_history.get(user_id, set())

        # Handle cold-start (new user)
        if user_id not in self._user_id_map:
            return self._cold_start_recommendations(
                top_k, difficulty_filter, category_filter, solved_problems
            )

        # Get recommendations based on strategy
        if strategy == "collaborative":
            recommendations = self._collaborative_recommend(
                user_id, solved_problems, top_k * 2
            )
        elif strategy == "content":
            recommendations = self._content_based_recommend(
                user_id, solved_problems, top_k * 2
            )
        else:  # hybrid
            recommendations = self._hybrid_recommend(
                user_id, solved_problems, top_k * 2
            )

        # Apply filters
        filtered = self._apply_filters(
            recommendations, difficulty_filter, category_filter, solved_problems
        )

        return filtered[:top_k]

    def _collaborative_recommend(
        self, user_id: UUID, solved: set[UUID], top_k: int
    ) -> list[dict]:
        """Collaborative filtering recommendations using NCF"""
        if self._ncf_model is None:
            return []

        user_idx = self._user_id_map.get(user_id)
        if user_idx is None:
            return []

        # Exclude solved problems
        exclude_indices = {
            self._item_id_map[pid] for pid in solved if pid in self._item_id_map
        }

        # Get NCF recommendations
        ncf_recs = self._ncf_model.recommend(
            user_idx, top_k=top_k, exclude_items=exclude_indices
        )

        recommendations = []
        for item_idx, score in ncf_recs:
            problem_id = self._reverse_item_map.get(item_idx)
            if problem_id and problem_id in self._problem_features:
                features = self._problem_features[problem_id]
                recommendations.append(
                    {
                        "problem_id": problem_id,
                        "score": score,
                        "reason": "similar_users",
                        **features,
                    }
                )

        return recommendations

    def _content_based_recommend(
        self, user_id: UUID, solved: set[UUID], top_k: int
    ) -> list[dict]:
        """Content-based recommendations based on user profile"""
        # Analyze user's solved problem patterns
        user_profile = self._analyze_user_profile(solved)

        # Score unsolved problems
        candidates = []
        for problem_id, features in self._problem_features.items():
            if problem_id in solved:
                continue

            score = self._content_similarity_score(user_profile, features)
            candidates.append(
                {
                    "problem_id": problem_id,
                    "score": score,
                    "reason": "content_match",
                    **features,
                }
            )

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _hybrid_recommend(
        self, user_id: UUID, solved: set[UUID], top_k: int
    ) -> list[dict]:
        """Hybrid recommendations combining collaborative and content-based"""
        collab_recs = self._collaborative_recommend(user_id, solved, top_k)
        content_recs = self._content_based_recommend(user_id, solved, top_k)

        # Merge with weighted scores
        combined = {}
        collab_weight = 0.6
        content_weight = 0.4

        for rec in collab_recs:
            pid = rec["problem_id"]
            combined[pid] = {
                **rec,
                "score": rec["score"] * collab_weight,
                "reason": "hybrid",
            }

        for rec in content_recs:
            pid = rec["problem_id"]
            if pid in combined:
                combined[pid]["score"] += rec["score"] * content_weight
            else:
                combined[pid] = {
                    **rec,
                    "score": rec["score"] * content_weight,
                    "reason": "hybrid",
                }

        # Sort and return
        result = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return result[:top_k]

    def _analyze_user_profile(self, solved: set[UUID]) -> dict:
        """Analyze user's solved problems to build profile"""
        profile = {
            "difficulties": {"easy": 0, "medium": 0, "hard": 0},
            "categories": {},
            "patterns": {},
            "avg_acceptance_rate": 0.5,
        }

        if not solved:
            return profile

        total_acceptance = 0
        for pid in solved:
            if pid not in self._problem_features:
                continue

            features = self._problem_features[pid]

            # Count difficulties
            diff = features.get("difficulty", "medium")
            profile["difficulties"][diff] = profile["difficulties"].get(diff, 0) + 1

            # Count categories
            cat = features.get("category", "unknown")
            profile["categories"][cat] = profile["categories"].get(cat, 0) + 1

            # Count patterns
            for pattern in features.get("patterns", []):
                profile["patterns"][pattern] = profile["patterns"].get(pattern, 0) + 1

            total_acceptance += features.get("acceptance_rate", 0.5)

        profile["avg_acceptance_rate"] = total_acceptance / len(solved)

        return profile

    def _content_similarity_score(self, profile: dict, features: dict) -> float:
        """Calculate content-based similarity score"""
        score = 0.0

        # Difficulty progression score
        difficulty = features.get("difficulty", "medium")
        user_difficulties = profile["difficulties"]
        total_solved = sum(user_difficulties.values())

        if total_solved > 0:
            # Prefer next difficulty level
            if difficulty == "medium" and user_difficulties.get("easy", 0) > 0:
                score += 0.3
            elif difficulty == "hard" and user_difficulties.get("medium", 0) > 0:
                score += 0.3
            elif difficulty == "easy":
                score += 0.1

        # Category familiarity score
        category = features.get("category", "unknown")
        if category in profile["categories"]:
            score += 0.2  # Familiar category
        else:
            score += 0.3  # New category to explore

        # Pattern progression score
        problem_patterns = features.get("patterns", [])
        for pattern in problem_patterns:
            if pattern in profile["patterns"]:
                score += 0.1  # Reinforce learned pattern
            else:
                score += 0.15  # Learn new pattern

        # Acceptance rate adjustment
        acceptance = features.get("acceptance_rate", 0.5)
        user_level = profile["avg_acceptance_rate"]

        # Prefer slightly challenging problems
        if 0.3 <= acceptance <= user_level + 0.1:
            score += 0.2

        return min(score, 1.0)

    def _cold_start_recommendations(
        self,
        top_k: int,
        difficulty: str | None = None,
        category: str | None = None,
        solved: set[UUID] | None = None,
    ) -> list[dict]:
        """Recommendations for new users or when no model"""
        solved = solved or set()

        candidates = []
        for problem_id, features in self._problem_features.items():
            if problem_id in solved:
                continue

            # Default: recommend popular easy problems
            score = features.get("acceptance_rate", 0.5)

            if features.get("difficulty") == "easy":
                score += 0.3

            candidates.append(
                {
                    "problem_id": problem_id,
                    "score": score,
                    "reason": "popular",
                    **features,
                }
            )

        # Apply filters
        filtered = self._apply_filters(candidates, difficulty, category, solved)

        return filtered[:top_k]

    def _apply_filters(
        self,
        recommendations: list[dict],
        difficulty: str | None,
        category: str | None,
        solved: set[UUID],
    ) -> list[dict]:
        """Apply difficulty and category filters"""
        filtered = []

        for rec in recommendations:
            # Skip solved
            if rec["problem_id"] in solved:
                continue

            # Difficulty filter
            if difficulty and rec.get("difficulty") != difficulty:
                continue

            # Category filter
            if category and rec.get("category") != category:
                continue

            filtered.append(rec)

        return filtered

    def get_skill_gaps(self, user_id: UUID) -> list[dict]:
        """
        Identify skill gaps for a user.

        Returns:
            List of categories/patterns with low coverage
        """
        solved = self._user_history.get(user_id, set())
        profile = self._analyze_user_profile(solved)

        # Find categories with few solved problems
        all_categories = set()
        for features in self._problem_features.values():
            all_categories.add(features.get("category", "unknown"))

        gaps = []
        for cat in all_categories:
            solved_count = profile["categories"].get(cat, 0)
            total_in_cat = sum(
                1 for f in self._problem_features.values() if f.get("category") == cat
            )

            if total_in_cat > 0:
                coverage = solved_count / total_in_cat
                if coverage < 0.3:  # Less than 30% solved
                    gaps.append(
                        {
                            "type": "category",
                            "name": cat,
                            "solved": solved_count,
                            "total": total_in_cat,
                            "coverage": coverage,
                        }
                    )

        gaps.sort(key=lambda x: x["coverage"])
        return gaps[:5]

    def get_next_challenge(self, user_id: UUID) -> dict | None:
        """
        Get next challenge problem for user.

        Returns:
            Next appropriate challenge problem or None
        """
        solved = self._user_history.get(user_id, set())
        profile = self._analyze_user_profile(solved)

        # Determine appropriate difficulty
        difficulties = profile["difficulties"]
        if difficulties.get("hard", 0) > 5:
            target_diff = "hard"  # Expert level
        elif difficulties.get("medium", 0) > 10:
            target_diff = "hard"  # Ready for hard
        elif difficulties.get("easy", 0) > 5:
            target_diff = "medium"  # Ready for medium
        else:
            target_diff = "easy"  # Still building basics

        # Find unsolved problem at target difficulty
        candidates = []
        for pid, features in self._problem_features.items():
            if pid in solved:
                continue
            if features.get("difficulty") == target_diff:
                candidates.append({"problem_id": pid, **features})

        if candidates:
            # Return random from candidates
            return np.random.choice(candidates)

        return None
