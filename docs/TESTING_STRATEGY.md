# Code Tutor AI - 테스트 전략

## Testing Strategy & Quality Assurance

**버전**: 1.0
**최종 수정**: 2025-12-26

---

## 1. 테스트 피라미드

```
                    ┌─────────┐
                    │   E2E   │  10%
                    │  Tests  │
                  ┌─┴─────────┴─┐
                  │ Integration │  20%
                  │   Tests     │
                ┌─┴─────────────┴─┐
                │   Unit Tests    │  70%
                └─────────────────┘
```

### 테스트 비율 목표

| 계층 | 비율 | 실행 시간 | 목적 |
|------|------|-----------|------|
| **Unit** | 70% | 빠름 (< 1분) | 개별 함수/클래스 로직 검증 |
| **Integration** | 20% | 중간 (< 5분) | 컴포넌트 간 상호작용 |
| **E2E** | 10% | 느림 (< 15분) | 전체 사용자 시나리오 |

---

## 2. Backend 테스트

### 2.1 디렉토리 구조

```
backend/tests/
├── conftest.py              # 공통 fixture
├── unit/
│   ├── identity/
│   │   ├── test_user.py
│   │   ├── test_email_vo.py
│   │   └── test_password_vo.py
│   ├── learning/
│   │   ├── test_problem.py
│   │   ├── test_submission.py
│   │   └── test_progress.py
│   ├── ai_tutor/
│   │   ├── test_conversation.py
│   │   └── test_code_review.py
│   └── code_execution/
│       └── test_execution_request.py
├── integration/
│   ├── test_user_repository.py
│   ├── test_problem_repository.py
│   ├── test_llm_adapter.py
│   └── test_sandbox_adapter.py
└── e2e/
    ├── test_auth_flow.py
    ├── test_problem_solving_flow.py
    └── test_ai_chat_flow.py
```

### 2.2 Unit Tests

#### Domain Model 테스트

```python
# tests/unit/identity/test_user.py

import pytest
from code_tutor.identity.domain.model.user import User
from code_tutor.identity.domain.model.value_objects import Email, Password

class TestUser:
    """User Aggregate 테스트"""

    def test_register_creates_user_with_valid_data(self):
        # Given
        email = "test@example.com"
        password = "SecurePass123"
        nickname = "테스터"

        # When
        user = User.register(email, password, nickname)

        # Then
        assert user.email.value == email
        assert user.nickname == nickname
        assert user._is_active is True

    def test_register_raises_on_invalid_email(self):
        # Given
        invalid_email = "not-an-email"

        # When/Then
        with pytest.raises(ValueError, match="Invalid email"):
            User.register(invalid_email, "SecurePass123", "테스터")

    def test_authenticate_returns_true_for_correct_password(self):
        # Given
        user = User.register("test@example.com", "SecurePass123", "테스터")

        # When/Then
        assert user.authenticate("SecurePass123") is True
        assert user.authenticate("WrongPassword") is False

    def test_change_password_updates_password(self):
        # Given
        user = User.register("test@example.com", "OldPass123", "테스터")

        # When
        user.change_password("OldPass123", "NewPass456")

        # Then
        assert user.authenticate("NewPass456") is True
        assert user.authenticate("OldPass123") is False

    def test_change_password_raises_on_wrong_current(self):
        # Given
        user = User.register("test@example.com", "OldPass123", "테스터")

        # When/Then
        with pytest.raises(InvalidPasswordError):
            user.change_password("WrongPassword", "NewPass456")

    def test_deactivate_prevents_authentication(self):
        # Given
        user = User.register("test@example.com", "SecurePass123", "테스터")

        # When
        user.deactivate()

        # Then
        with pytest.raises(UserInactiveError):
            user.authenticate("SecurePass123")

    def test_register_emits_user_registered_event(self):
        # Given/When
        user = User.register("test@example.com", "SecurePass123", "테스터")
        events = user.collect_events()

        # Then
        assert len(events) == 1
        assert isinstance(events[0], UserRegistered)
        assert events[0].user_id == user.id
```

#### Value Object 테스트

```python
# tests/unit/identity/test_email_vo.py

import pytest
from code_tutor.identity.domain.model.value_objects import Email

class TestEmail:
    """Email Value Object 테스트"""

    @pytest.mark.parametrize("valid_email", [
        "user@example.com",
        "test.user@domain.co.kr",
        "user+tag@gmail.com",
    ])
    def test_valid_emails_are_accepted(self, valid_email):
        email = Email(valid_email)
        assert email.value == valid_email

    @pytest.mark.parametrize("invalid_email", [
        "not-an-email",
        "@nodomain.com",
        "missing@.com",
        "",
        "spaces in@email.com",
    ])
    def test_invalid_emails_raise_error(self, invalid_email):
        with pytest.raises(ValueError, match="Invalid email"):
            Email(invalid_email)

    def test_emails_are_immutable(self):
        email = Email("test@example.com")
        with pytest.raises(AttributeError):
            email.value = "new@example.com"

    def test_equal_emails_are_equal(self):
        email1 = Email("test@example.com")
        email2 = Email("test@example.com")
        assert email1 == email2

    def test_different_emails_are_not_equal(self):
        email1 = Email("test1@example.com")
        email2 = Email("test2@example.com")
        assert email1 != email2
```

#### Application Service 테스트

```python
# tests/unit/learning/test_submit_solution_service.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from code_tutor.learning.application.services.submit_service import (
    SubmitSolutionService
)
from code_tutor.learning.application.dto.commands import SubmitCommand
from code_tutor.learning.domain.model.value_objects import Verdict

class TestSubmitSolutionService:
    """제출 서비스 테스트"""

    @pytest.fixture
    def mock_dependencies(self):
        return {
            "submission_repo": AsyncMock(),
            "problem_repo": AsyncMock(),
            "sandbox": AsyncMock(),
            "event_publisher": AsyncMock(),
        }

    @pytest.fixture
    def service(self, mock_dependencies):
        return SubmitSolutionService(**mock_dependencies)

    @pytest.mark.asyncio
    async def test_submit_accepted_solution(
        self, service, mock_dependencies
    ):
        # Given
        problem_id = uuid4()
        user_id = uuid4()

        mock_problem = MagicMock()
        mock_problem.get_all_test_cases.return_value = [
            MagicMock(input="1\n2", expected_output="3", matches=lambda x: x == "3")
        ]
        mock_problem.score = 10

        mock_dependencies["problem_repo"].find_by_id.return_value = mock_problem
        mock_dependencies["sandbox"].execute.return_value = MagicMock(
            stdout="3",
            stderr="",
            exit_code=0,
            execution_time_ms=50,
            memory_used_kb=10000
        )

        command = SubmitCommand(
            user_id=user_id,
            problem_id=problem_id,
            code="print(int(input()) + int(input()))"
        )

        # When
        result = await service.execute(command)

        # Then
        assert result.verdict == "accepted"
        assert result.score == 10
        mock_dependencies["submission_repo"].save.assert_called_once()
        mock_dependencies["event_publisher"].publish.assert_called()

    @pytest.mark.asyncio
    async def test_submit_wrong_answer(self, service, mock_dependencies):
        # Given
        mock_problem = MagicMock()
        mock_problem.get_all_test_cases.return_value = [
            MagicMock(input="1\n2", expected_output="3", matches=lambda x: x == "3")
        ]

        mock_dependencies["problem_repo"].find_by_id.return_value = mock_problem
        mock_dependencies["sandbox"].execute.return_value = MagicMock(
            stdout="5",  # Wrong answer
            stderr="",
            exit_code=0,
            execution_time_ms=50,
            memory_used_kb=10000
        )

        command = SubmitCommand(
            user_id=uuid4(),
            problem_id=uuid4(),
            code="print(5)"  # Wrong code
        )

        # When
        result = await service.execute(command)

        # Then
        assert result.verdict == "wrong_answer"
        assert result.score == 0

    @pytest.mark.asyncio
    async def test_submit_problem_not_found(self, service, mock_dependencies):
        # Given
        mock_dependencies["problem_repo"].find_by_id.return_value = None

        command = SubmitCommand(
            user_id=uuid4(),
            problem_id=uuid4(),
            code="print('hello')"
        )

        # When/Then
        with pytest.raises(ProblemNotFoundError):
            await service.execute(command)
```

### 2.3 Integration Tests

```python
# tests/integration/test_user_repository.py

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.adapters.outbound.persistence.sqlalchemy_user_repo import (
    SQLAlchemyUserRepository
)
from code_tutor.identity.domain.model.user import User

@pytest.mark.integration
class TestSQLAlchemyUserRepository:
    """User Repository 통합 테스트"""

    @pytest.fixture
    async def repository(self, db_session: AsyncSession):
        return SQLAlchemyUserRepository(session=db_session)

    @pytest.mark.asyncio
    async def test_save_and_find_by_id(self, repository):
        # Given
        user = User.register("test@example.com", "SecurePass123", "테스터")

        # When
        await repository.save(user)
        found = await repository.find_by_id(user.id)

        # Then
        assert found is not None
        assert found.email.value == "test@example.com"
        assert found.nickname == "테스터"

    @pytest.mark.asyncio
    async def test_find_by_email(self, repository):
        # Given
        user = User.register("find@example.com", "SecurePass123", "테스터")
        await repository.save(user)

        # When
        found = await repository.find_by_email(Email("find@example.com"))

        # Then
        assert found is not None
        assert found.id == user.id

    @pytest.mark.asyncio
    async def test_find_nonexistent_returns_none(self, repository):
        # When
        found = await repository.find_by_id(UserId(uuid4()))

        # Then
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_removes_user(self, repository):
        # Given
        user = User.register("delete@example.com", "SecurePass123", "테스터")
        await repository.save(user)

        # When
        await repository.delete(user.id)
        found = await repository.find_by_id(user.id)

        # Then
        assert found is None
```

```python
# tests/integration/test_llm_adapter.py

import pytest
from code_tutor.ai_tutor.adapters.outbound.llm.eeve_adapter import EEVEAdapter

@pytest.mark.integration
@pytest.mark.slow
class TestEEVEAdapter:
    """LLM Adapter 통합 테스트 (GPU 필요)"""

    @pytest.fixture(scope="class")
    def adapter(self):
        # 테스트용 소형 모델 사용
        return EEVEAdapter(
            model_path="yanolja/EEVE-Korean-2.8B-v1.0",
            device="cuda",
            load_in_4bit=True
        )

    @pytest.mark.asyncio
    async def test_generate_response(self, adapter):
        # Given
        messages = [
            {"role": "system", "content": "당신은 Python 튜터입니다."},
            {"role": "user", "content": "리스트란 무엇인가요?"}
        ]

        # When
        response = await adapter.generate(messages, max_tokens=100)

        # Then
        assert isinstance(response, str)
        assert len(response) > 0
        assert "리스트" in response or "list" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_handles_empty_messages(self, adapter):
        # When/Then
        with pytest.raises(ValueError):
            await adapter.generate([], max_tokens=100)
```

### 2.4 E2E Tests

```python
# tests/e2e/test_auth_flow.py

import pytest
from httpx import AsyncClient

@pytest.mark.e2e
class TestAuthFlow:
    """인증 E2E 테스트"""

    @pytest.mark.asyncio
    async def test_register_login_logout_flow(self, client: AsyncClient):
        # 1. 회원가입
        register_response = await client.post("/api/v1/auth/register", json={
            "email": "e2e@example.com",
            "password": "SecurePass123",
            "nickname": "E2E테스터"
        })
        assert register_response.status_code == 201
        tokens = register_response.json()["tokens"]
        access_token = tokens["access_token"]

        # 2. 토큰으로 내 정보 조회
        me_response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert me_response.status_code == 200
        assert me_response.json()["data"]["email"] == "e2e@example.com"

        # 3. 로그아웃
        logout_response = await client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert logout_response.status_code == 200

        # 4. 로그아웃 후 토큰 무효화 확인
        invalid_response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert invalid_response.status_code == 401


# tests/e2e/test_problem_solving_flow.py

@pytest.mark.e2e
class TestProblemSolvingFlow:
    """문제 풀이 E2E 테스트"""

    @pytest.mark.asyncio
    async def test_solve_problem_flow(
        self, client: AsyncClient, auth_headers: dict
    ):
        # 1. 문제 목록 조회
        problems_response = await client.get(
            "/api/v1/problems",
            headers=auth_headers
        )
        assert problems_response.status_code == 200
        problems = problems_response.json()["data"]["problems"]
        assert len(problems) > 0

        # 2. 첫 번째 문제 상세 조회
        problem_id = problems[0]["problem_id"]
        detail_response = await client.get(
            f"/api/v1/problems/{problem_id}",
            headers=auth_headers
        )
        assert detail_response.status_code == 200

        # 3. 코드 제출
        submit_response = await client.post(
            f"/api/v1/problems/{problem_id}/submit",
            headers=auth_headers,
            json={
                "code": "def solution(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target-n], i]\n        seen[n] = i"
            }
        )
        assert submit_response.status_code == 202
        submission_id = submit_response.json()["data"]["submission_id"]

        # 4. 결과 확인 (폴링)
        import asyncio
        for _ in range(10):
            result_response = await client.get(
                f"/api/v1/submissions/{submission_id}",
                headers=auth_headers
            )
            if result_response.json()["data"]["status"] == "completed":
                break
            await asyncio.sleep(1)

        assert result_response.json()["data"]["status"] == "completed"
```

---

## 3. Frontend 테스트

### 3.1 디렉토리 구조

```
frontend/src/
├── features/
│   └── auth/
│       ├── __tests__/
│       │   ├── LoginForm.test.tsx
│       │   └── authStore.test.ts
│       └── ...
└── shared/
    └── ui/
        └── __tests__/
            └── Button.test.tsx
```

### 3.2 컴포넌트 테스트

```typescript
// features/auth/__tests__/LoginForm.test.tsx

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LoginForm } from '../ui/LoginForm';
import { authApi } from '../api/authApi';

jest.mock('../api/authApi');

describe('LoginForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders email and password inputs', () => {
    render(<LoginForm />);

    expect(screen.getByLabelText(/이메일/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/비밀번호/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /로그인/i })).toBeInTheDocument();
  });

  it('shows validation error for invalid email', async () => {
    render(<LoginForm />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/이메일/i), 'invalid-email');
    await user.click(screen.getByRole('button', { name: /로그인/i }));

    expect(await screen.findByText(/유효한 이메일/i)).toBeInTheDocument();
  });

  it('shows validation error for short password', async () => {
    render(<LoginForm />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/이메일/i), 'test@example.com');
    await user.type(screen.getByLabelText(/비밀번호/i), 'short');
    await user.click(screen.getByRole('button', { name: /로그인/i }));

    expect(await screen.findByText(/8자 이상/i)).toBeInTheDocument();
  });

  it('calls login API on valid submission', async () => {
    const mockLogin = authApi.login as jest.Mock;
    mockLogin.mockResolvedValue({
      user: { email: 'test@example.com' },
      tokens: { access_token: 'token' }
    });

    render(<LoginForm />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/이메일/i), 'test@example.com');
    await user.type(screen.getByLabelText(/비밀번호/i), 'SecurePass123');
    await user.click(screen.getByRole('button', { name: /로그인/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'SecurePass123'
      });
    });
  });

  it('shows error message on login failure', async () => {
    const mockLogin = authApi.login as jest.Mock;
    mockLogin.mockRejectedValue(new Error('이메일 또는 비밀번호가 올바르지 않습니다.'));

    render(<LoginForm />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/이메일/i), 'test@example.com');
    await user.type(screen.getByLabelText(/비밀번호/i), 'WrongPassword');
    await user.click(screen.getByRole('button', { name: /로그인/i }));

    expect(await screen.findByText(/올바르지 않습니다/i)).toBeInTheDocument();
  });

  it('disables button while submitting', async () => {
    const mockLogin = authApi.login as jest.Mock;
    mockLogin.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<LoginForm />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText(/이메일/i), 'test@example.com');
    await user.type(screen.getByLabelText(/비밀번호/i), 'SecurePass123');
    await user.click(screen.getByRole('button', { name: /로그인/i }));

    expect(screen.getByRole('button', { name: /로그인/i })).toBeDisabled();
  });
});
```

### 3.3 Store 테스트

```typescript
// features/auth/__tests__/authStore.test.ts

import { useAuthStore } from '../model/authStore';
import { authApi } from '../api/authApi';

jest.mock('../api/authApi');

describe('authStore', () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
    jest.clearAllMocks();
  });

  it('starts with no user', () => {
    const { user, isAuthenticated } = useAuthStore.getState();

    expect(user).toBeNull();
    expect(isAuthenticated).toBe(false);
  });

  it('login sets user and tokens', async () => {
    const mockUser = { user_id: '1', email: 'test@example.com', nickname: 'Test' };
    const mockTokens = { access_token: 'token', refresh_token: 'refresh' };

    (authApi.login as jest.Mock).mockResolvedValue({
      user: mockUser,
      tokens: mockTokens
    });

    await useAuthStore.getState().login('test@example.com', 'password');

    const { user, isAuthenticated, accessToken } = useAuthStore.getState();
    expect(user).toEqual(mockUser);
    expect(isAuthenticated).toBe(true);
    expect(accessToken).toBe('token');
  });

  it('logout clears user and tokens', async () => {
    // First login
    (authApi.login as jest.Mock).mockResolvedValue({
      user: { email: 'test@example.com' },
      tokens: { access_token: 'token' }
    });
    await useAuthStore.getState().login('test@example.com', 'password');

    // Then logout
    useAuthStore.getState().logout();

    const { user, isAuthenticated, accessToken } = useAuthStore.getState();
    expect(user).toBeNull();
    expect(isAuthenticated).toBe(false);
    expect(accessToken).toBeNull();
  });
});
```

---

## 4. 테스트 설정

### 4.1 pytest 설정

```toml
# pyproject.toml

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests (requires DB)",
    "e2e: End-to-end tests (requires full stack)",
    "slow: Slow tests (> 5 seconds)",
]
addopts = "-v --tb=short --strict-markers"
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src/code_tutor"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
```

### 4.2 공통 Fixtures

```python
# tests/conftest.py

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from code_tutor.main import app
from code_tutor.shared.infrastructure.database import Base

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5433/test_db"

@pytest.fixture(scope="session")
def event_loop():
    """이벤트 루프 생성"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def db_engine():
    """테스트 DB 엔진"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine):
    """테스트 DB 세션 (각 테스트마다 롤백)"""
    async_session = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def client():
    """HTTP 클라이언트"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def auth_headers(client):
    """인증된 헤더"""
    response = await client.post("/api/v1/auth/register", json={
        "email": "fixture@example.com",
        "password": "SecurePass123",
        "nickname": "Fixture"
    })
    token = response.json()["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

### 4.3 Jest 설정

```javascript
// jest.config.js

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/test/setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/main.tsx',
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 80,
      statements: 80,
    },
  },
};
```

---

## 5. 테스트 실행

### 5.1 로컬 실행

```bash
# Backend
uv run pytest                           # 전체 테스트
uv run pytest tests/unit                # Unit만
uv run pytest tests/integration         # Integration만
uv run pytest -m "not slow"             # 느린 테스트 제외
uv run pytest --cov --cov-report=html   # 커버리지 리포트

# Frontend
npm test                                # 전체 테스트
npm test -- --coverage                  # 커버리지
npm test -- --watch                     # 워치 모드
```

### 5.2 CI 환경

```yaml
# .github/workflows/test.yml 참조 (DEPLOYMENT.md)
```

---

## 6. 품질 기준

### 6.1 커버리지 목표

| 영역 | 최소 커버리지 |
|------|---------------|
| Domain Models | 90% |
| Application Services | 85% |
| Adapters | 70% |
| Frontend Components | 80% |

### 6.2 테스트 품질 체크리스트

- [ ] 각 테스트는 하나의 동작만 검증
- [ ] 테스트 이름이 동작을 설명
- [ ] Given-When-Then 패턴 사용
- [ ] 외부 의존성은 Mock 처리
- [ ] 테스트 간 독립성 보장
- [ ] 엣지 케이스 포함
- [ ] 에러 케이스 포함
