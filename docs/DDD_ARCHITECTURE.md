# Code Tutor AI - DDD 아키텍처 설계서

## Domain-Driven Design & Hexagonal Architecture

**버전**: 1.0
**작성일**: 2025-12-26
**설계 원칙**: 높은 응집도(High Cohesion), 낮은 결합도(Loose Coupling)

---

## 목차

1. [전략적 설계 (Strategic Design)](#1-전략적-설계-strategic-design)
2. [전술적 설계 (Tactical Design)](#2-전술적-설계-tactical-design)
3. [헥사고날 아키텍처](#3-헥사고날-아키텍처)
4. [패키지 구조](#4-패키지-구조)
5. [의존성 규칙](#5-의존성-규칙)
6. [환경 설정](#6-환경-설정)

---

# 1. 전략적 설계 (Strategic Design)

## 1.1 도메인 분석

### 핵심 도메인 (Core Domain)
> 비즈니스 경쟁력의 핵심. 직접 개발 및 투자 집중

| 도메인 | 설명 | 우선순위 |
|--------|------|----------|
| **AI 튜터링** | LLM 기반 대화형 학습 | P0 |
| **코드 리뷰** | AI 기반 코드 분석 및 피드백 | P0 |
| **맞춤형 추천** | 딥러닝 기반 문제 추천 | P1 |

### 지원 도메인 (Supporting Domain)
> 핵심 도메인을 지원. 커스텀 개발 필요

| 도메인 | 설명 | 우선순위 |
|--------|------|----------|
| **학습 관리** | 문제, 제출, 진도 추적 | P1 |
| **사용자 관리** | 회원, 프로필, 통계 | P1 |

### 일반 도메인 (Generic Domain)
> 비즈니스 특화 X. 외부 솔루션 활용 가능

| 도메인 | 설명 | 구현 방식 |
|--------|------|-----------|
| **인증/인가** | JWT, OAuth | 라이브러리 활용 |
| **코드 실행** | 샌드박스 실행 | Docker 활용 |
| **알림** | 이메일, 푸시 | 외부 서비스 |

---

## 1.2 Bounded Context 정의

```
┌─────────────────────────────────────────────────────────────────┐
│                     Code Tutor AI System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   Identity       │    │   Learning       │                   │
│  │   Context        │    │   Context        │                   │
│  │                  │    │                  │                   │
│  │  • User          │    │  • Problem       │                   │
│  │  • Auth          │    │  • Submission    │                   │
│  │  • Profile       │    │  • Progress      │                   │
│  │  • Session       │    │  • Statistics    │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                             │
│           │    Upstream           │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   AI Tutor       │◄───│   Code           │                   │
│  │   Context        │    │   Execution      │                   │
│  │                  │    │   Context        │                   │
│  │  • Conversation  │    │                  │                   │
│  │  • CodeReview    │    │  • Sandbox       │                   │
│  │  • Recommendation│    │  • TestRunner    │                   │
│  │  • Prediction    │    │  • Result        │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Bounded Context 상세

#### 1. Identity Context (신원 컨텍스트)
**책임**: 사용자 신원 관리 및 인증

| 구성요소 | 설명 |
|----------|------|
| **User** | 사용자 계정 정보 |
| **Authentication** | 로그인/로그아웃, 토큰 관리 |
| **Profile** | 닉네임, 설정 등 개인 정보 |
| **Session** | 활성 세션 관리 |

**Ubiquitous Language**:
- `User`: 시스템에 등록된 사용자
- `Authenticate`: 자격 증명 검증
- `Token`: 인증 상태를 나타내는 JWT
- `Session`: 사용자의 활성 접속 상태

---

#### 2. Learning Context (학습 컨텍스트)
**책임**: 학습 콘텐츠 및 진도 관리

| 구성요소 | 설명 |
|----------|------|
| **Problem** | 알고리즘 문제 |
| **Submission** | 사용자 코드 제출 |
| **Progress** | 학습 진도 |
| **Statistics** | 학습 통계 |

**Ubiquitous Language**:
- `Problem`: 풀어야 할 알고리즘 문제
- `Submission`: 문제에 대한 코드 제출
- `Solve`: 문제를 성공적으로 해결함
- `Progress`: 카테고리별 학습 진행 상황
- `Streak`: 연속 학습 일수

---

#### 3. AI Tutor Context (AI 튜터 컨텍스트)
**책임**: AI 기반 학습 지원

| 구성요소 | 설명 |
|----------|------|
| **Conversation** | 대화형 Q&A |
| **CodeReview** | AI 코드 분석 |
| **Recommendation** | 맞춤형 문제 추천 |
| **Prediction** | 학습 성과 예측 |

**Ubiquitous Language**:
- `Conversation`: 사용자와 AI 간의 대화 세션
- `Message`: 대화 내 개별 메시지
- `Review`: 코드에 대한 AI 분석 결과
- `Feedback`: 코드 개선을 위한 제안
- `Recommendation`: AI가 추천하는 다음 학습 문제

---

#### 4. Code Execution Context (코드 실행 컨텍스트)
**책임**: 안전한 코드 실행 환경 제공

| 구성요소 | 설명 |
|----------|------|
| **Sandbox** | 격리된 실행 환경 |
| **TestRunner** | 테스트 케이스 실행 |
| **Result** | 실행 결과 |

**Ubiquitous Language**:
- `Sandbox`: 격리된 Docker 컨테이너 실행 환경
- `Execute`: 사용자 코드 실행
- `TestCase`: 입력과 기대 출력의 쌍
- `Verdict`: 실행 결과 판정 (Pass/Fail/Error/Timeout)

---

## 1.3 Context Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                      Context Map                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Identity Context                                              │
│        │                                                        │
│        │ [U/D] Published Language                               │
│        │ (UserDTO: id, email, nickname)                         │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Anti-Corruption Layer                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                    │                    │              │
│        ▼                    ▼                    ▼              │
│   Learning Context    AI Tutor Context    Code Execution       │
│        │                    │                                   │
│        │ [Partnership]      │ [Conformist]                      │
│        └────────────────────┘                                   │
│                             │                                   │
│                             │ [Customer/Supplier]               │
│                             ▼                                   │
│                    Code Execution Context                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Context 관계 유형

| 관계 | Upstream | Downstream | 설명 |
|------|----------|------------|------|
| **Published Language** | Identity | All | 표준화된 사용자 DTO 제공 |
| **Partnership** | Learning | AI Tutor | 양방향 협력, 동시 발전 |
| **Customer/Supplier** | AI Tutor | Code Execution | AI가 코드 실행 결과 소비 |
| **Anti-Corruption Layer** | All | Identity | 도메인 모델 보호 |

---

# 2. 전술적 설계 (Tactical Design)

## 2.1 Identity Context

### Aggregate: User

```python
# ========================
# Value Objects
# ========================

@dataclass(frozen=True)
class Email:
    """이메일 값 객체"""
    value: str

    def __post_init__(self):
        if not self._is_valid(self.value):
            raise ValueError(f"Invalid email: {self.value}")

    @staticmethod
    def _is_valid(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))


@dataclass(frozen=True)
class Password:
    """비밀번호 값 객체 (해시된 상태)"""
    hashed_value: str

    @classmethod
    def create(cls, plain_password: str) -> 'Password':
        if len(plain_password) < 8:
            raise ValueError("Password must be at least 8 characters")
        from passlib.hash import bcrypt
        return cls(bcrypt.hash(plain_password))

    def verify(self, plain_password: str) -> bool:
        from passlib.hash import bcrypt
        return bcrypt.verify(plain_password, self.hashed_value)


@dataclass(frozen=True)
class UserId:
    """사용자 ID 값 객체"""
    value: UUID

    @classmethod
    def generate(cls) -> 'UserId':
        return cls(uuid4())


@dataclass(frozen=True)
class Nickname:
    """닉네임 값 객체"""
    value: str

    def __post_init__(self):
        if not 2 <= len(self.value) <= 20:
            raise ValueError("Nickname must be 2-20 characters")


# ========================
# Entity
# ========================

@dataclass
class UserProfile:
    """사용자 프로필 엔티티"""
    nickname: Nickname
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

    def update_nickname(self, new_nickname: Nickname) -> None:
        self.nickname = new_nickname


# ========================
# Aggregate Root
# ========================

class User:
    """사용자 Aggregate Root"""

    def __init__(
        self,
        user_id: UserId,
        email: Email,
        password: Password,
        profile: UserProfile,
        created_at: datetime,
        is_active: bool = True
    ):
        self._user_id = user_id
        self._email = email
        self._password = password
        self._profile = profile
        self._created_at = created_at
        self._is_active = is_active
        self._domain_events: List[DomainEvent] = []

    @classmethod
    def register(
        cls,
        email: str,
        password: str,
        nickname: str
    ) -> 'User':
        """팩토리 메서드: 새 사용자 등록"""
        user = cls(
            user_id=UserId.generate(),
            email=Email(email),
            password=Password.create(password),
            profile=UserProfile(nickname=Nickname(nickname)),
            created_at=datetime.utcnow()
        )
        user._add_event(UserRegistered(user._user_id))
        return user

    def authenticate(self, password: str) -> bool:
        """인증 검증"""
        if not self._is_active:
            raise UserInactiveError()
        return self._password.verify(password)

    def change_password(self, old_password: str, new_password: str) -> None:
        """비밀번호 변경"""
        if not self.authenticate(old_password):
            raise InvalidPasswordError()
        self._password = Password.create(new_password)
        self._add_event(PasswordChanged(self._user_id))

    def update_profile(self, nickname: Optional[str] = None) -> None:
        """프로필 업데이트"""
        if nickname:
            self._profile.update_nickname(Nickname(nickname))
        self._add_event(ProfileUpdated(self._user_id))

    def deactivate(self) -> None:
        """계정 비활성화"""
        self._is_active = False
        self._add_event(UserDeactivated(self._user_id))

    def _add_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)

    def collect_events(self) -> List[DomainEvent]:
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    # Properties
    @property
    def id(self) -> UserId:
        return self._user_id

    @property
    def email(self) -> Email:
        return self._email

    @property
    def nickname(self) -> str:
        return self._profile.nickname.value
```

### Domain Events

```python
@dataclass(frozen=True)
class UserRegistered(DomainEvent):
    user_id: UserId
    occurred_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(frozen=True)
class PasswordChanged(DomainEvent):
    user_id: UserId
    occurred_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(frozen=True)
class ProfileUpdated(DomainEvent):
    user_id: UserId
    occurred_at: datetime = field(default_factory=datetime.utcnow)
```

---

## 2.2 Learning Context

### Aggregate: Problem

```python
# ========================
# Value Objects
# ========================

@dataclass(frozen=True)
class ProblemId:
    value: UUID

    @classmethod
    def generate(cls) -> 'ProblemId':
        return cls(uuid4())


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def score(self) -> int:
        return {"easy": 10, "medium": 25, "hard": 50}[self.value]


class Category(Enum):
    ARRAY = "array"
    LINKED_LIST = "linked_list"
    STACK_QUEUE = "stack_queue"
    TREE = "tree"
    GRAPH = "graph"
    SORTING = "sorting"
    DP = "dynamic_programming"
    STRING = "string"


@dataclass(frozen=True)
class TestCase:
    """테스트 케이스 값 객체"""
    input: str
    expected_output: str
    is_hidden: bool = False

    def matches(self, actual_output: str) -> bool:
        return self.expected_output.strip() == actual_output.strip()


@dataclass(frozen=True)
class Hint:
    """힌트 값 객체"""
    order: int
    content: str
    penalty: int = 5  # 힌트 사용 시 감점


# ========================
# Aggregate Root
# ========================

class Problem:
    """문제 Aggregate Root"""

    def __init__(
        self,
        problem_id: ProblemId,
        title: str,
        description: str,
        difficulty: Difficulty,
        category: Category,
        test_cases: List[TestCase],
        hints: List[Hint],
        time_limit_ms: int = 5000,
        memory_limit_mb: int = 256
    ):
        self._problem_id = problem_id
        self._title = title
        self._description = description
        self._difficulty = difficulty
        self._category = category
        self._test_cases = test_cases
        self._hints = sorted(hints, key=lambda h: h.order)
        self._time_limit_ms = time_limit_ms
        self._memory_limit_mb = memory_limit_mb

    def get_visible_test_cases(self) -> List[TestCase]:
        """공개 테스트 케이스만 반환"""
        return [tc for tc in self._test_cases if not tc.is_hidden]

    def get_all_test_cases(self) -> List[TestCase]:
        """모든 테스트 케이스 반환 (채점용)"""
        return self._test_cases.copy()

    def get_hint(self, order: int) -> Optional[Hint]:
        """특정 순서의 힌트 반환"""
        for hint in self._hints:
            if hint.order == order:
                return hint
        return None

    def get_available_hints_count(self) -> int:
        return len(self._hints)

    @property
    def id(self) -> ProblemId:
        return self._problem_id

    @property
    def score(self) -> int:
        return self._difficulty.score
```

### Aggregate: Submission

```python
# ========================
# Value Objects
# ========================

@dataclass(frozen=True)
class SubmissionId:
    value: UUID

    @classmethod
    def generate(cls) -> 'SubmissionId':
        return cls(uuid4())


class Verdict(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    WRONG_ANSWER = "wrong_answer"
    TIME_LIMIT = "time_limit"
    MEMORY_LIMIT = "memory_limit"
    RUNTIME_ERROR = "runtime_error"
    COMPILE_ERROR = "compile_error"


@dataclass(frozen=True)
class ExecutionResult:
    """실행 결과 값 객체"""
    verdict: Verdict
    runtime_ms: int
    memory_kb: int
    passed_tests: int
    total_tests: int
    error_message: Optional[str] = None


@dataclass(frozen=True)
class Code:
    """코드 값 객체"""
    content: str
    language: str = "python"

    def __post_init__(self):
        if len(self.content) > 50000:
            raise ValueError("Code too long (max 50000 chars)")


# ========================
# Aggregate Root
# ========================

class Submission:
    """제출 Aggregate Root"""

    def __init__(
        self,
        submission_id: SubmissionId,
        user_id: UserId,
        problem_id: ProblemId,
        code: Code,
        submitted_at: datetime,
        result: Optional[ExecutionResult] = None,
        hints_used: int = 0
    ):
        self._submission_id = submission_id
        self._user_id = user_id
        self._problem_id = problem_id
        self._code = code
        self._submitted_at = submitted_at
        self._result = result
        self._hints_used = hints_used
        self._domain_events: List[DomainEvent] = []

    @classmethod
    def create(
        cls,
        user_id: UserId,
        problem_id: ProblemId,
        code: str,
        hints_used: int = 0
    ) -> 'Submission':
        submission = cls(
            submission_id=SubmissionId.generate(),
            user_id=user_id,
            problem_id=problem_id,
            code=Code(code),
            submitted_at=datetime.utcnow(),
            hints_used=hints_used
        )
        submission._add_event(SubmissionCreated(submission._submission_id))
        return submission

    def record_result(self, result: ExecutionResult) -> None:
        """실행 결과 기록"""
        self._result = result

        if result.verdict == Verdict.ACCEPTED:
            self._add_event(ProblemSolved(
                submission_id=self._submission_id,
                user_id=self._user_id,
                problem_id=self._problem_id
            ))
        else:
            self._add_event(SubmissionFailed(
                submission_id=self._submission_id,
                verdict=result.verdict
            ))

    def calculate_score(self, base_score: int) -> int:
        """점수 계산 (힌트 사용 시 감점)"""
        if not self.is_accepted:
            return 0
        penalty = self._hints_used * 5
        return max(0, base_score - penalty)

    @property
    def is_accepted(self) -> bool:
        return self._result and self._result.verdict == Verdict.ACCEPTED

    @property
    def id(self) -> SubmissionId:
        return self._submission_id

    def _add_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)
```

### Aggregate: Progress

```python
@dataclass(frozen=True)
class CategoryProgress:
    """카테고리별 진도 값 객체"""
    category: Category
    solved_count: int
    total_count: int

    @property
    def percentage(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.solved_count / self.total_count) * 100


class LearningProgress:
    """학습 진도 Aggregate Root"""

    def __init__(
        self,
        user_id: UserId,
        category_progress: Dict[Category, CategoryProgress],
        total_score: int = 0,
        streak_days: int = 0,
        last_activity: Optional[datetime] = None
    ):
        self._user_id = user_id
        self._category_progress = category_progress
        self._total_score = total_score
        self._streak_days = streak_days
        self._last_activity = last_activity

    def record_solve(self, category: Category, score: int) -> None:
        """문제 해결 기록"""
        current = self._category_progress.get(category)
        if current:
            self._category_progress[category] = CategoryProgress(
                category=category,
                solved_count=current.solved_count + 1,
                total_count=current.total_count
            )

        self._total_score += score
        self._update_streak()

    def _update_streak(self) -> None:
        """연속 학습 일수 업데이트"""
        today = datetime.utcnow().date()

        if self._last_activity is None:
            self._streak_days = 1
        elif self._last_activity.date() == today:
            pass  # 오늘 이미 활동함
        elif (today - self._last_activity.date()).days == 1:
            self._streak_days += 1
        else:
            self._streak_days = 1

        self._last_activity = datetime.utcnow()

    def get_weak_categories(self, threshold: float = 30.0) -> List[Category]:
        """취약 카테고리 식별 (진도 30% 미만)"""
        return [
            cp.category
            for cp in self._category_progress.values()
            if cp.percentage < threshold
        ]
```

---

## 2.3 AI Tutor Context

### Aggregate: Conversation

```python
# ========================
# Value Objects
# ========================

@dataclass(frozen=True)
class ConversationId:
    value: UUID

    @classmethod
    def generate(cls) -> 'ConversationId':
        return cls(uuid4())


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    """메시지 값 객체"""
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_llm_format(self) -> dict:
        return {"role": self.role.value, "content": self.content}


# ========================
# Aggregate Root
# ========================

class Conversation:
    """대화 Aggregate Root"""

    MAX_CONTEXT_MESSAGES = 10

    def __init__(
        self,
        conversation_id: ConversationId,
        user_id: UserId,
        messages: List[Message],
        created_at: datetime,
        topic: Optional[str] = None
    ):
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._messages = messages
        self._created_at = created_at
        self._topic = topic

    @classmethod
    def start(cls, user_id: UserId, system_prompt: str) -> 'Conversation':
        """새 대화 시작"""
        return cls(
            conversation_id=ConversationId.generate(),
            user_id=user_id,
            messages=[Message(MessageRole.SYSTEM, system_prompt)],
            created_at=datetime.utcnow()
        )

    def add_user_message(self, content: str) -> None:
        """사용자 메시지 추가"""
        self._messages.append(Message(MessageRole.USER, content))
        self._infer_topic_if_needed()

    def add_assistant_message(self, content: str) -> None:
        """AI 응답 추가"""
        self._messages.append(Message(MessageRole.ASSISTANT, content))

    def get_context_for_llm(self) -> List[dict]:
        """LLM에 전달할 컨텍스트 (최근 N개)"""
        # 시스템 프롬프트 + 최근 메시지
        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        recent = other_messages[-self.MAX_CONTEXT_MESSAGES:]

        return [m.to_llm_format() for m in system_messages + recent]

    def _infer_topic_if_needed(self) -> None:
        """첫 사용자 메시지로 토픽 추론"""
        if self._topic is None:
            user_messages = [m for m in self._messages if m.role == MessageRole.USER]
            if user_messages:
                # 첫 메시지의 앞 30자를 토픽으로
                self._topic = user_messages[0].content[:30]

    @property
    def id(self) -> ConversationId:
        return self._conversation_id

    @property
    def message_count(self) -> int:
        return len([m for m in self._messages if m.role != MessageRole.SYSTEM])
```

### Aggregate: CodeReview

```python
@dataclass(frozen=True)
class ReviewId:
    value: UUID


@dataclass(frozen=True)
class ComplexityAnalysis:
    """복잡도 분석 값 객체"""
    time_complexity: str  # "O(n)", "O(n^2)", etc.
    space_complexity: str
    explanation: str


@dataclass(frozen=True)
class CodeSuggestion:
    """코드 개선 제안 값 객체"""
    line_number: Optional[int]
    severity: str  # "info", "warning", "error"
    message: str
    suggested_code: Optional[str] = None


class CodeReview:
    """코드 리뷰 Aggregate Root"""

    def __init__(
        self,
        review_id: ReviewId,
        user_id: UserId,
        code: Code,
        complexity: ComplexityAnalysis,
        suggestions: List[CodeSuggestion],
        overall_score: float,  # 0-100
        created_at: datetime
    ):
        self._review_id = review_id
        self._user_id = user_id
        self._code = code
        self._complexity = complexity
        self._suggestions = suggestions
        self._overall_score = overall_score
        self._created_at = created_at

    @classmethod
    def create(
        cls,
        user_id: UserId,
        code: str,
        ai_analysis: dict
    ) -> 'CodeReview':
        """AI 분석 결과로 리뷰 생성"""
        return cls(
            review_id=ReviewId(uuid4()),
            user_id=user_id,
            code=Code(code),
            complexity=ComplexityAnalysis(
                time_complexity=ai_analysis["time_complexity"],
                space_complexity=ai_analysis["space_complexity"],
                explanation=ai_analysis["explanation"]
            ),
            suggestions=[
                CodeSuggestion(**s) for s in ai_analysis["suggestions"]
            ],
            overall_score=ai_analysis["score"],
            created_at=datetime.utcnow()
        )

    def get_critical_suggestions(self) -> List[CodeSuggestion]:
        """심각도가 높은 제안만 반환"""
        return [s for s in self._suggestions if s.severity == "error"]

    @property
    def needs_improvement(self) -> bool:
        return self._overall_score < 70
```

### Aggregate: Recommendation

```python
@dataclass(frozen=True)
class RecommendedProblem:
    """추천 문제 값 객체"""
    problem_id: ProblemId
    reason: str
    confidence: float  # 0-1
    predicted_success_rate: float


class ProblemRecommendation:
    """문제 추천 Aggregate Root"""

    def __init__(
        self,
        user_id: UserId,
        recommendations: List[RecommendedProblem],
        generated_at: datetime,
        model_version: str
    ):
        self._user_id = user_id
        self._recommendations = recommendations
        self._generated_at = generated_at
        self._model_version = model_version

    @classmethod
    def generate(
        cls,
        user_id: UserId,
        ml_predictions: List[dict],
        model_version: str
    ) -> 'ProblemRecommendation':
        """ML 모델 예측으로 추천 생성"""
        recommendations = [
            RecommendedProblem(
                problem_id=ProblemId(UUID(p["problem_id"])),
                reason=p["reason"],
                confidence=p["confidence"],
                predicted_success_rate=p["success_rate"]
            )
            for p in ml_predictions
        ]

        return cls(
            user_id=user_id,
            recommendations=recommendations,
            generated_at=datetime.utcnow(),
            model_version=model_version
        )

    def get_top_recommendations(self, n: int = 5) -> List[RecommendedProblem]:
        """상위 N개 추천"""
        sorted_recs = sorted(
            self._recommendations,
            key=lambda r: r.confidence,
            reverse=True
        )
        return sorted_recs[:n]
```

---

## 2.4 Code Execution Context

### Aggregate: ExecutionRequest

```python
@dataclass(frozen=True)
class ExecutionId:
    value: UUID


@dataclass(frozen=True)
class SandboxConfig:
    """샌드박스 설정 값 객체"""
    timeout_seconds: int = 5
    memory_limit_mb: int = 256
    cpu_limit: float = 0.5
    network_disabled: bool = True


@dataclass(frozen=True)
class ExecutionOutput:
    """실행 출력 값 객체"""
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    memory_used_kb: int


class ExecutionRequest:
    """코드 실행 요청 Aggregate Root"""

    def __init__(
        self,
        execution_id: ExecutionId,
        code: Code,
        test_cases: List[TestCase],
        config: SandboxConfig,
        status: str = "pending"
    ):
        self._execution_id = execution_id
        self._code = code
        self._test_cases = test_cases
        self._config = config
        self._status = status
        self._outputs: List[ExecutionOutput] = []

    @classmethod
    def create(
        cls,
        code: str,
        test_cases: List[TestCase],
        config: Optional[SandboxConfig] = None
    ) -> 'ExecutionRequest':
        return cls(
            execution_id=ExecutionId(uuid4()),
            code=Code(code),
            test_cases=test_cases,
            config=config or SandboxConfig()
        )

    def record_output(self, output: ExecutionOutput) -> None:
        """테스트 케이스 실행 결과 기록"""
        self._outputs.append(output)

    def complete(self) -> ExecutionResult:
        """실행 완료 및 결과 집계"""
        self._status = "completed"

        passed = sum(
            1 for output, tc in zip(self._outputs, self._test_cases)
            if tc.matches(output.stdout) and output.exit_code == 0
        )

        # 가장 나쁜 결과를 verdict로
        if any(o.exit_code != 0 for o in self._outputs):
            verdict = Verdict.RUNTIME_ERROR
        elif passed < len(self._test_cases):
            verdict = Verdict.WRONG_ANSWER
        else:
            verdict = Verdict.ACCEPTED

        max_time = max(o.execution_time_ms for o in self._outputs)
        max_memory = max(o.memory_used_kb for o in self._outputs)

        return ExecutionResult(
            verdict=verdict,
            runtime_ms=max_time,
            memory_kb=max_memory,
            passed_tests=passed,
            total_tests=len(self._test_cases)
        )
```

---

# 3. 헥사고날 아키텍처

## 3.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Primary Adapters                            │
│                    (Driving / Input Adapters)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │   REST API  │  │  WebSocket  │  │    CLI      │  │   GraphQL   ││
│  │  (FastAPI)  │  │  (Chat)     │  │  (Admin)    │  │  (Future)   ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘│
└─────────┼────────────────┼────────────────┼────────────────┼────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Ports                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Application Services                         ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    ││
│  │  │UserService│  │AIService  │  │CodeService│  │LearnService│    ││
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Domain Core                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Domain Models                              ││
│  │  ┌────────┐  ┌────────────┐  ┌───────────┐  ┌───────────────┐  ││
│  │  │  User  │  │Conversation│  │Submission │  │ExecutionRequest│  ││
│  │  └────────┘  └────────────┘  └───────────┘  └───────────────┘  ││
│  │  ┌────────┐  ┌────────────┐  ┌───────────┐  ┌───────────────┐  ││
│  │  │Problem │  │ CodeReview │  │ Progress  │  │Recommendation │  ││
│  │  └────────┘  └────────────┘  └───────────┘  └───────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Domain Services                              ││
│  │  ┌────────────────────┐  ┌────────────────────────────────────┐││
│  │  │SubmissionEvaluator │  │ RecommendationEngine               │││
│  │  └────────────────────┘  └────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Output Ports                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Port Interfaces                             ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    ││
│  │  │UserRepo   │  │LLMPort    │  │SandboxPort│  │CachePort  │    ││
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Secondary Adapters                             │
│                     (Driven / Output Adapters)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ PostgreSQL  │  │   Redis     │  │   Docker    │  │HuggingFace  ││
│  │  Adapter    │  │  Adapter    │  │  Sandbox    │  │  LLM        ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   FAISS     │  │   Email     │  │  External   │                 │
│  │   Vector    │  │   Service   │  │   API       │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Port 정의

### Input Ports (Use Cases)

```python
# ========================
# Identity Context Ports
# ========================

class RegisterUserUseCase(Protocol):
    async def execute(self, command: RegisterUserCommand) -> UserDTO: ...

class AuthenticateUserUseCase(Protocol):
    async def execute(self, command: LoginCommand) -> TokenDTO: ...

class UpdateProfileUseCase(Protocol):
    async def execute(self, command: UpdateProfileCommand) -> UserDTO: ...


# ========================
# Learning Context Ports
# ========================

class GetProblemsUseCase(Protocol):
    async def execute(self, query: GetProblemsQuery) -> List[ProblemDTO]: ...

class SubmitSolutionUseCase(Protocol):
    async def execute(self, command: SubmitCommand) -> SubmissionResultDTO: ...

class GetProgressUseCase(Protocol):
    async def execute(self, user_id: UUID) -> ProgressDTO: ...


# ========================
# AI Tutor Context Ports
# ========================

class ChatWithTutorUseCase(Protocol):
    async def execute(self, command: ChatCommand) -> ChatResponseDTO: ...

class ReviewCodeUseCase(Protocol):
    async def execute(self, command: ReviewCommand) -> CodeReviewDTO: ...

class GetRecommendationsUseCase(Protocol):
    async def execute(self, user_id: UUID) -> List[RecommendationDTO]: ...


# ========================
# Code Execution Context Ports
# ========================

class ExecuteCodeUseCase(Protocol):
    async def execute(self, command: ExecuteCommand) -> ExecutionResultDTO: ...
```

### Output Ports (Repository & External Services)

```python
# ========================
# Repository Ports
# ========================

class UserRepository(Protocol):
    async def save(self, user: User) -> None: ...
    async def find_by_id(self, user_id: UserId) -> Optional[User]: ...
    async def find_by_email(self, email: Email) -> Optional[User]: ...
    async def delete(self, user_id: UserId) -> None: ...


class ProblemRepository(Protocol):
    async def save(self, problem: Problem) -> None: ...
    async def find_by_id(self, problem_id: ProblemId) -> Optional[Problem]: ...
    async def find_all(
        self,
        filters: ProblemFilters,
        pagination: Pagination
    ) -> List[Problem]: ...
    async def count(self, filters: ProblemFilters) -> int: ...


class SubmissionRepository(Protocol):
    async def save(self, submission: Submission) -> None: ...
    async def find_by_id(self, submission_id: SubmissionId) -> Optional[Submission]: ...
    async def find_by_user(
        self,
        user_id: UserId,
        pagination: Pagination
    ) -> List[Submission]: ...


class ConversationRepository(Protocol):
    async def save(self, conversation: Conversation) -> None: ...
    async def find_by_id(self, conversation_id: ConversationId) -> Optional[Conversation]: ...
    async def find_by_user(self, user_id: UserId) -> List[Conversation]: ...


# ========================
# External Service Ports
# ========================

class LLMPort(Protocol):
    """LLM 서비스 포트"""
    async def generate(
        self,
        messages: List[dict],
        max_tokens: int = 512
    ) -> str: ...


class CodeSandboxPort(Protocol):
    """코드 실행 샌드박스 포트"""
    async def execute(
        self,
        code: str,
        input_data: str,
        config: SandboxConfig
    ) -> ExecutionOutput: ...


class VectorSearchPort(Protocol):
    """벡터 검색 포트"""
    async def index(self, id: str, embedding: List[float]) -> None: ...
    async def search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[str]: ...


class RecommendationModelPort(Protocol):
    """추천 모델 포트"""
    async def predict(
        self,
        user_id: UUID,
        problem_ids: List[UUID]
    ) -> List[float]: ...


class CachePort(Protocol):
    """캐시 포트"""
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...


class EventPublisherPort(Protocol):
    """도메인 이벤트 발행 포트"""
    async def publish(self, event: DomainEvent) -> None: ...
```

---

## 3.3 Application Services

```python
class SubmitSolutionService:
    """제출 유스케이스 구현"""

    def __init__(
        self,
        submission_repo: SubmissionRepository,
        problem_repo: ProblemRepository,
        sandbox: CodeSandboxPort,
        event_publisher: EventPublisherPort
    ):
        self._submission_repo = submission_repo
        self._problem_repo = problem_repo
        self._sandbox = sandbox
        self._event_publisher = event_publisher

    async def execute(self, command: SubmitCommand) -> SubmissionResultDTO:
        # 1. 문제 조회
        problem = await self._problem_repo.find_by_id(
            ProblemId(command.problem_id)
        )
        if not problem:
            raise ProblemNotFoundError(command.problem_id)

        # 2. 제출 생성
        submission = Submission.create(
            user_id=UserId(command.user_id),
            problem_id=problem.id,
            code=command.code,
            hints_used=command.hints_used
        )

        # 3. 코드 실행
        test_cases = problem.get_all_test_cases()
        execution_request = ExecutionRequest.create(
            code=command.code,
            test_cases=test_cases
        )

        for tc in test_cases:
            output = await self._sandbox.execute(
                code=command.code,
                input_data=tc.input,
                config=execution_request._config
            )
            execution_request.record_output(output)

        # 4. 결과 기록
        result = execution_request.complete()
        submission.record_result(result)

        # 5. 저장
        await self._submission_repo.save(submission)

        # 6. 도메인 이벤트 발행
        for event in submission.collect_events():
            await self._event_publisher.publish(event)

        # 7. DTO 반환
        return SubmissionResultDTO(
            submission_id=str(submission.id.value),
            verdict=result.verdict.value,
            runtime_ms=result.runtime_ms,
            memory_kb=result.memory_kb,
            passed_tests=result.passed_tests,
            total_tests=result.total_tests,
            score=submission.calculate_score(problem.score)
        )
```

---

# 4. 패키지 구조

## 4.1 Backend 구조 (Python)

```
backend/
├── pyproject.toml              # 의존성 관리 (uv)
├── alembic/                    # DB 마이그레이션
│   └── versions/
├── src/
│   └── code_tutor/
│       ├── __init__.py
│       │
│       ├── identity/           # Identity Context
│       │   ├── domain/
│       │   │   ├── model/
│       │   │   │   ├── user.py           # User Aggregate
│       │   │   │   ├── value_objects.py  # Email, Password, etc.
│       │   │   │   └── events.py         # Domain Events
│       │   │   └── service/
│       │   │       └── authentication.py # Domain Service
│       │   ├── application/
│       │   │   ├── ports/
│       │   │   │   ├── input/           # Use Cases
│       │   │   │   │   ├── register_user.py
│       │   │   │   │   ├── authenticate.py
│       │   │   │   │   └── update_profile.py
│       │   │   │   └── output/          # Repository Interfaces
│       │   │   │       └── user_repository.py
│       │   │   ├── services/            # Application Services
│       │   │   │   └── user_service.py
│       │   │   └── dto/
│       │   │       ├── commands.py
│       │   │       └── responses.py
│       │   └── adapters/
│       │       ├── inbound/
│       │       │   └── rest/
│       │       │       ├── router.py
│       │       │       └── schemas.py
│       │       └── outbound/
│       │           └── persistence/
│       │               ├── sqlalchemy_user_repo.py
│       │               └── models.py
│       │
│       ├── learning/           # Learning Context
│       │   ├── domain/
│       │   │   ├── model/
│       │   │   │   ├── problem.py
│       │   │   │   ├── submission.py
│       │   │   │   ├── progress.py
│       │   │   │   └── value_objects.py
│       │   │   └── service/
│       │   │       └── submission_evaluator.py
│       │   ├── application/
│       │   │   ├── ports/
│       │   │   ├── services/
│       │   │   └── dto/
│       │   └── adapters/
│       │       ├── inbound/rest/
│       │       └── outbound/persistence/
│       │
│       ├── ai_tutor/           # AI Tutor Context
│       │   ├── domain/
│       │   │   ├── model/
│       │   │   │   ├── conversation.py
│       │   │   │   ├── code_review.py
│       │   │   │   └── recommendation.py
│       │   │   └── service/
│       │   │       └── recommendation_engine.py
│       │   ├── application/
│       │   │   ├── ports/
│       │   │   │   ├── input/
│       │   │   │   └── output/
│       │   │   │       ├── llm_port.py
│       │   │   │       └── recommendation_model_port.py
│       │   │   ├── services/
│       │   │   └── dto/
│       │   └── adapters/
│       │       ├── inbound/
│       │       │   ├── rest/
│       │       │   └── websocket/    # 실시간 채팅
│       │       └── outbound/
│       │           ├── llm/
│       │           │   └── eeve_adapter.py
│       │           ├── ml/
│       │           │   ├── ncf_adapter.py
│       │           │   └── lstm_adapter.py
│       │           └── vector/
│       │               └── faiss_adapter.py
│       │
│       ├── code_execution/     # Code Execution Context
│       │   ├── domain/
│       │   │   └── model/
│       │   │       └── execution.py
│       │   ├── application/
│       │   │   ├── ports/
│       │   │   │   └── output/
│       │   │   │       └── sandbox_port.py
│       │   │   └── services/
│       │   └── adapters/
│       │       └── outbound/
│       │           └── docker_sandbox.py
│       │
│       ├── shared/             # Shared Kernel
│       │   ├── domain/
│       │   │   ├── base.py            # Base classes
│       │   │   └── events.py          # DomainEvent base
│       │   ├── infrastructure/
│       │   │   ├── database.py        # DB connection
│       │   │   ├── redis.py           # Cache
│       │   │   ├── event_bus.py       # Event publishing
│       │   │   └── logging.py
│       │   └── utils/
│       │       └── pagination.py
│       │
│       └── main.py             # FastAPI entry point
│
└── tests/
    ├── unit/
    │   ├── identity/
    │   ├── learning/
    │   ├── ai_tutor/
    │   └── code_execution/
    ├── integration/
    └── e2e/
```

## 4.2 Frontend 구조 (React)

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── src/
    ├── main.tsx
    ├── App.tsx
    │
    ├── features/               # Feature-Sliced Design
    │   ├── auth/
    │   │   ├── api/
    │   │   │   └── authApi.ts
    │   │   ├── model/
    │   │   │   └── authStore.ts      # Zustand store
    │   │   ├── ui/
    │   │   │   ├── LoginForm.tsx
    │   │   │   └── RegisterForm.tsx
    │   │   └── index.ts
    │   │
    │   ├── problems/
    │   │   ├── api/
    │   │   │   └── problemApi.ts
    │   │   ├── model/
    │   │   │   └── problemStore.ts
    │   │   ├── ui/
    │   │   │   ├── ProblemList.tsx
    │   │   │   ├── ProblemDetail.tsx
    │   │   │   └── ProblemFilters.tsx
    │   │   └── index.ts
    │   │
    │   ├── code-editor/
    │   │   ├── api/
    │   │   ├── model/
    │   │   ├── ui/
    │   │   │   ├── CodeEditor.tsx     # Monaco Editor
    │   │   │   ├── TestResults.tsx
    │   │   │   └── SubmitButton.tsx
    │   │   └── index.ts
    │   │
    │   ├── ai-chat/
    │   │   ├── api/
    │   │   │   └── chatApi.ts
    │   │   ├── model/
    │   │   │   └── chatStore.ts
    │   │   ├── ui/
    │   │   │   ├── ChatWindow.tsx
    │   │   │   ├── MessageList.tsx
    │   │   │   └── ChatInput.tsx
    │   │   └── index.ts
    │   │
    │   └── dashboard/
    │       ├── api/
    │       ├── model/
    │       └── ui/
    │           ├── ProgressChart.tsx
    │           └── Statistics.tsx
    │
    ├── shared/
    │   ├── api/
    │   │   ├── client.ts             # Axios instance
    │   │   └── types.ts
    │   ├── ui/
    │   │   ├── Button.tsx
    │   │   ├── Card.tsx
    │   │   ├── Modal.tsx
    │   │   └── Loading.tsx
    │   ├── hooks/
    │   │   ├── useAuth.ts
    │   │   └── useDebounce.ts
    │   └── lib/
    │       └── utils.ts
    │
    ├── pages/
    │   ├── HomePage.tsx
    │   ├── LoginPage.tsx
    │   ├── ProblemsPage.tsx
    │   ├── ProblemSolvePage.tsx
    │   ├── ChatPage.tsx
    │   └── DashboardPage.tsx
    │
    └── app/
        ├── router.tsx
        ├── providers.tsx
        └── layouts/
            └── MainLayout.tsx
```

---

# 5. 의존성 규칙

## 5.1 의존성 방향

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dependency Rule                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Adapters (REST, DB, LLM, Docker)                              │
│       │                                                         │
│       │  depends on                                             │
│       ▼                                                         │
│   Application (Use Cases, DTOs)                                 │
│       │                                                         │
│       │  depends on                                             │
│       ▼                                                         │
│   Domain (Aggregates, Entities, Value Objects, Domain Services) │
│                                                                 │
│   ❌ Domain NEVER depends on Application or Adapters            │
│   ❌ Application NEVER depends on Adapters                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 의존성 주입 설정

```python
# src/code_tutor/shared/infrastructure/container.py

from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """의존성 주입 컨테이너"""

    # Configuration
    config = providers.Configuration()

    # Infrastructure
    database = providers.Singleton(
        Database,
        url=config.database.url
    )

    redis = providers.Singleton(
        RedisClient,
        url=config.redis.url
    )

    # Repositories
    user_repository = providers.Factory(
        SQLAlchemyUserRepository,
        session_factory=database.provided.session_factory
    )

    problem_repository = providers.Factory(
        SQLAlchemyProblemRepository,
        session_factory=database.provided.session_factory
    )

    submission_repository = providers.Factory(
        SQLAlchemySubmissionRepository,
        session_factory=database.provided.session_factory
    )

    # External Services
    llm_adapter = providers.Singleton(
        EEVEAdapter,
        model_path=config.llm.model_path,
        device=config.llm.device
    )

    sandbox_adapter = providers.Factory(
        DockerSandboxAdapter,
        docker_host=config.docker.host
    )

    # Application Services
    user_service = providers.Factory(
        UserService,
        user_repo=user_repository,
        event_publisher=event_publisher
    )

    submit_service = providers.Factory(
        SubmitSolutionService,
        submission_repo=submission_repository,
        problem_repo=problem_repository,
        sandbox=sandbox_adapter,
        event_publisher=event_publisher
    )

    chat_service = providers.Factory(
        ChatService,
        conversation_repo=conversation_repository,
        llm=llm_adapter
    )
```

## 5.3 모듈 간 의존성 매트릭스

| From \ To | Identity | Learning | AI Tutor | Code Exec | Shared |
|-----------|----------|----------|----------|-----------|--------|
| **Identity** | - | ❌ | ❌ | ❌ | ✅ |
| **Learning** | ✅ (UserDTO) | - | ❌ | ✅ (SandboxPort) | ✅ |
| **AI Tutor** | ✅ (UserDTO) | ✅ (ProblemDTO) | - | ❌ | ✅ |
| **Code Exec** | ❌ | ❌ | ❌ | - | ✅ |
| **Shared** | ❌ | ❌ | ❌ | ❌ | - |

**규칙**:
- ✅ 허용된 의존성
- ❌ 금지된 의존성
- Shared는 어떤 Context에도 의존하지 않음
- Context 간 통신은 DTO 또는 Domain Event를 통해서만

---

# 6. 환경 설정

## 6.1 개발 환경

### Python 환경 (uv 사용)

```toml
# pyproject.toml
[project]
name = "code-tutor-ai"
version = "1.0.0"
description = "AI-based Python Algorithm Learning Platform"
requires-python = ">=3.11"

dependencies = [
    # Web Framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",

    # Database
    "sqlalchemy>=2.0.25",
    "alembic>=1.13.1",
    "asyncpg>=0.29.0",
    "redis>=5.0.1",

    # Auth
    "passlib[bcrypt]>=1.7.4",
    "python-jose[cryptography]>=3.3.0",

    # ML/AI
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "bitsandbytes>=0.41.0",
    "faiss-cpu>=1.7.4",

    # Utils
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "dependency-injector>=4.41.0",
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.26.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
strict = true
```

### 환경 변수

```bash
# .env.example
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/code_tutor
REDIS_URL=redis://localhost:6379/0

# Auth
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# LLM
LLM_MODEL_PATH=yanolja/EEVE-Korean-2.8B-v1.0
LLM_DEVICE=cuda
LLM_MAX_TOKENS=512

# Docker Sandbox
DOCKER_HOST=unix:///var/run/docker.sock
SANDBOX_TIMEOUT_SECONDS=5
SANDBOX_MEMORY_LIMIT_MB=256

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 6.2 Docker 구성

```yaml
# docker-compose.yml
version: '3.9'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/code_tutor
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./backend:/app
      - /var/run/docker.sock:/var/run/docker.sock  # Sandbox용
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: code_tutor
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

## 6.3 Backend Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Run migrations and start server
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn src.code_tutor.main:app --host 0.0.0.0 --port 8000"]
```

---

## 부록: 용어 사전 (Glossary)

| 용어 | 정의 | Context |
|------|------|---------|
| **User** | 시스템에 등록된 사용자 | Identity |
| **Problem** | 해결해야 할 알고리즘 문제 | Learning |
| **Submission** | 문제에 대한 코드 제출 | Learning |
| **Verdict** | 제출 결과 판정 | Learning |
| **Conversation** | 사용자와 AI 간의 대화 | AI Tutor |
| **Review** | 코드에 대한 AI 분석 | AI Tutor |
| **Recommendation** | AI가 추천하는 문제 | AI Tutor |
| **Sandbox** | 격리된 코드 실행 환경 | Code Exec |
| **Aggregate** | 일관성 경계를 가진 엔티티 클러스터 | DDD |
| **Port** | 의존성 역전을 위한 인터페이스 | Hexagonal |
| **Adapter** | Port의 구체적 구현체 | Hexagonal |
