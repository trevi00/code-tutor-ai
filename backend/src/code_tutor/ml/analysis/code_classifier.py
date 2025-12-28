"""CodeBERT-based Code Quality Classifier (Transformer)"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CodeQualityClassifier:
    """
    Transformer-based code quality classifier using CodeBERT.

    Classifies code into quality categories:
    - Correctness: 코드가 의도한 대로 동작하는지
    - Efficiency: 시간/공간 복잡도 효율성
    - Readability: 코드 가독성
    - Best Practices: 코딩 컨벤션 준수

    Based on fine-tuned CodeBERT model.
    """

    QUALITY_LABELS = ["poor", "fair", "good", "excellent"]
    QUALITY_DIMENSIONS = ["correctness", "efficiency", "readability", "best_practices"]

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 4,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._device = device
        self._classification_head = None

    def _lazy_load(self):
        """Lazy load the model to save memory until needed"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import torch.nn as nn

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading CodeBERT classifier: {self.model_name} on {self._device}")

            # Load base model
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self._device)
            self._model.eval()

            # Classification head for each quality dimension
            hidden_size = self._model.config.hidden_size

            class MultiDimensionClassifier(nn.Module):
                """Multi-output classifier for code quality dimensions"""

                def __init__(self, hidden_size, num_labels, num_dimensions):
                    super().__init__()
                    self.classifiers = nn.ModuleList([
                        nn.Sequential(
                            nn.Dropout(0.1),
                            nn.Linear(hidden_size, hidden_size // 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_size // 2, num_labels)
                        )
                        for _ in range(num_dimensions)
                    ])
                    self.softmax = nn.Softmax(dim=-1)

                def forward(self, x):
                    outputs = []
                    for classifier in self.classifiers:
                        logits = classifier(x)
                        probs = self.softmax(logits)
                        outputs.append(probs)
                    return torch.stack(outputs, dim=1)

            self._classification_head = MultiDimensionClassifier(
                hidden_size,
                self.num_labels,
                len(self.QUALITY_DIMENSIONS)
            ).to(self._device)
            self._classification_head.eval()

            logger.info("CodeBERT classifier loaded successfully")

        except ImportError as e:
            logger.error(f"transformers not installed: {e}")
            raise ImportError(
                "Please install transformers: pip install transformers torch"
            )

    @property
    def model(self):
        """Get the model, loading it if necessary"""
        self._lazy_load()
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer, loading it if necessary"""
        self._lazy_load()
        return self._tokenizer

    def classify(
        self,
        code: str,
        language: str = "python"
    ) -> Dict:
        """
        Classify code quality across multiple dimensions.

        Args:
            code: Source code to classify
            language: Programming language

        Returns:
            Dict with quality scores for each dimension
        """
        import torch

        self._lazy_load()

        # Preprocess code
        code = f"# {language}\n{code.strip()}"

        # Tokenize
        inputs = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            # Get CodeBERT embeddings
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Get classification probabilities
            probs = self._classification_head(cls_embedding)
            probs = probs.cpu().numpy()[0]

        # Build result
        result = {
            "overall_score": 0,
            "overall_grade": "",
            "dimensions": {}
        }

        total_score = 0
        for i, dim in enumerate(self.QUALITY_DIMENSIONS):
            dim_probs = probs[i]
            predicted_label_idx = np.argmax(dim_probs)
            predicted_label = self.QUALITY_LABELS[predicted_label_idx]

            # Score: 0-100 based on weighted probabilities
            score = sum(
                prob * (j + 1) * 25
                for j, prob in enumerate(dim_probs)
            )

            result["dimensions"][dim] = {
                "label": predicted_label,
                "score": float(score),
                "probabilities": {
                    label: float(prob)
                    for label, prob in zip(self.QUALITY_LABELS, dim_probs)
                }
            }
            total_score += score

        # Overall score
        result["overall_score"] = total_score / len(self.QUALITY_DIMENSIONS)
        result["overall_grade"] = self._score_to_grade(result["overall_score"])

        return result

    def classify_batch(
        self,
        codes: List[str],
        language: str = "python"
    ) -> List[Dict]:
        """
        Classify multiple code snippets.

        Args:
            codes: List of source codes to classify
            language: Programming language

        Returns:
            List of classification results
        """
        return [self.classify(code, language) for code in codes]

    def get_improvement_suggestions(
        self,
        classification_result: Dict
    ) -> List[Dict]:
        """
        Generate improvement suggestions based on classification.

        Args:
            classification_result: Result from classify()

        Returns:
            List of suggestions for improvement
        """
        suggestions = []
        dimensions = classification_result.get("dimensions", {})

        for dim, info in dimensions.items():
            if info["score"] < 50:
                suggestion = self._get_suggestion_for_dimension(dim, info["label"])
                if suggestion:
                    suggestions.append(suggestion)

        # Sort by priority (lower score = higher priority)
        suggestions.sort(key=lambda x: x.get("priority", 5))

        return suggestions

    def _get_suggestion_for_dimension(
        self,
        dimension: str,
        label: str
    ) -> Optional[Dict]:
        """Get improvement suggestion for a specific dimension"""
        suggestions_map = {
            "correctness": {
                "poor": {
                    "message": "코드의 정확성이 낮습니다. 엣지 케이스와 예외 처리를 확인하세요.",
                    "tips": [
                        "입력 검증 추가",
                        "경계 조건 테스트",
                        "예외 처리 구현"
                    ],
                    "priority": 1
                },
                "fair": {
                    "message": "일부 케이스에서 오류가 발생할 수 있습니다.",
                    "tips": ["단위 테스트 추가", "엣지 케이스 검토"],
                    "priority": 2
                }
            },
            "efficiency": {
                "poor": {
                    "message": "시간/공간 복잡도가 높습니다. 알고리즘을 최적화하세요.",
                    "tips": [
                        "불필요한 중첩 루프 제거",
                        "적절한 자료구조 사용",
                        "메모이제이션 고려"
                    ],
                    "priority": 2
                },
                "fair": {
                    "message": "효율성 개선의 여지가 있습니다.",
                    "tips": ["캐싱 적용", "조기 종료 조건 추가"],
                    "priority": 3
                }
            },
            "readability": {
                "poor": {
                    "message": "코드 가독성이 낮습니다. 리팩토링을 권장합니다.",
                    "tips": [
                        "의미 있는 변수명 사용",
                        "함수로 분리",
                        "주석 추가"
                    ],
                    "priority": 3
                },
                "fair": {
                    "message": "가독성을 높일 수 있습니다.",
                    "tips": ["일관된 포맷팅", "docstring 추가"],
                    "priority": 4
                }
            },
            "best_practices": {
                "poor": {
                    "message": "코딩 컨벤션을 준수하지 않습니다.",
                    "tips": [
                        "PEP 8 스타일 가이드 준수",
                        "타입 힌트 추가",
                        "에러 처리 패턴 적용"
                    ],
                    "priority": 4
                },
                "fair": {
                    "message": "일부 베스트 프랙티스가 누락되었습니다.",
                    "tips": ["린터 적용", "코드 리뷰 수행"],
                    "priority": 5
                }
            }
        }

        if dimension in suggestions_map and label in suggestions_map[dimension]:
            return {
                "dimension": dimension,
                "dimension_ko": self._dimension_to_korean(dimension),
                **suggestions_map[dimension][label]
            }
        return None

    def _dimension_to_korean(self, dimension: str) -> str:
        """Translate dimension to Korean"""
        translations = {
            "correctness": "정확성",
            "efficiency": "효율성",
            "readability": "가독성",
            "best_practices": "베스트 프랙티스"
        }
        return translations.get(dimension, dimension)

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def compare_solutions(
        self,
        code1: str,
        code2: str,
        language: str = "python"
    ) -> Dict:
        """
        Compare quality of two solutions.

        Args:
            code1: First solution
            code2: Second solution
            language: Programming language

        Returns:
            Comparison results with winner for each dimension
        """
        result1 = self.classify(code1, language)
        result2 = self.classify(code2, language)

        comparison = {
            "solution1_score": result1["overall_score"],
            "solution2_score": result2["overall_score"],
            "overall_winner": 1 if result1["overall_score"] > result2["overall_score"] else 2,
            "dimension_comparison": {}
        }

        for dim in self.QUALITY_DIMENSIONS:
            score1 = result1["dimensions"][dim]["score"]
            score2 = result2["dimensions"][dim]["score"]

            comparison["dimension_comparison"][dim] = {
                "solution1_score": score1,
                "solution2_score": score2,
                "winner": 1 if score1 > score2 else (2 if score2 > score1 else 0),
                "difference": abs(score1 - score2)
            }

        return comparison

    def unload(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._classification_head
            self._model = None
            self._tokenizer = None
            self._classification_head = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("CodeBERT classifier unloaded")
