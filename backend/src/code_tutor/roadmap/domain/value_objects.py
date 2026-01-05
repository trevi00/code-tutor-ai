"""Value objects for Learning Roadmap domain."""

from enum import Enum


class PathLevel(str, Enum):
    """Learning path difficulty levels."""

    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

    @property
    def display_name(self) -> str:
        """Get Korean display name."""
        names = {
            PathLevel.BEGINNER: "입문",
            PathLevel.ELEMENTARY: "초급",
            PathLevel.INTERMEDIATE: "중급",
            PathLevel.ADVANCED: "고급",
        }
        return names[self]

    @property
    def order(self) -> int:
        """Get order for sorting."""
        orders = {
            PathLevel.BEGINNER: 1,
            PathLevel.ELEMENTARY: 2,
            PathLevel.INTERMEDIATE: 3,
            PathLevel.ADVANCED: 4,
        }
        return orders[self]


class LessonType(str, Enum):
    """Types of lessons in a learning path."""

    CONCEPT = "concept"      # 개념 설명 (텍스트/마크다운)
    PROBLEM = "problem"      # 문제 풀이
    TYPING = "typing"        # 받아쓰기 연습
    PATTERN = "pattern"      # 알고리즘 패턴 학습
    QUIZ = "quiz"            # 퀴즈 (객관식)

    @property
    def display_name(self) -> str:
        """Get Korean display name."""
        names = {
            LessonType.CONCEPT: "개념",
            LessonType.PROBLEM: "문제",
            LessonType.TYPING: "타이핑",
            LessonType.PATTERN: "패턴",
            LessonType.QUIZ: "퀴즈",
        }
        return names[self]


class ProgressStatus(str, Enum):
    """Status of user progress."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

    @property
    def display_name(self) -> str:
        """Get Korean display name."""
        names = {
            ProgressStatus.NOT_STARTED: "시작 전",
            ProgressStatus.IN_PROGRESS: "진행 중",
            ProgressStatus.COMPLETED: "완료",
        }
        return names[self]
