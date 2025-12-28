# Code Tutor AI - 보안 설계

## Security Architecture & OWASP Guidelines

**버전**: 1.0
**최종 수정**: 2025-12-26

---

## 1. 보안 개요

### 1.1 보안 원칙

| 원칙 | 설명 |
|------|------|
| **Defense in Depth** | 다층 방어 (네트워크, 애플리케이션, 데이터) |
| **Least Privilege** | 최소 권한 원칙 |
| **Secure by Default** | 기본적으로 안전한 설정 |
| **Fail Securely** | 실패 시에도 보안 유지 |

### 1.2 위협 모델

```
┌─────────────────────────────────────────────────────────────┐
│                    Threat Landscape                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   External Attackers                                        │
│   ├── Credential Theft (Brute Force, Phishing)              │
│   ├── Injection Attacks (SQL, XSS, Command)                 │
│   ├── Code Execution (Sandbox Escape)                       │
│   └── Data Exfiltration                                     │
│                                                             │
│   Malicious Users                                           │
│   ├── Sandbox Abuse (Crypto Mining, DDoS)                   │
│   ├── Resource Exhaustion                                   │
│   └── Privilege Escalation                                  │
│                                                             │
│   Insider Threats                                           │
│   ├── Data Leakage                                          │
│   └── Unauthorized Access                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. OWASP Top 10 대응

### 2.1 A01: Broken Access Control

#### 위험
- 다른 사용자의 데이터 접근
- 권한 없는 기능 실행

#### 대책

```python
# 1. Route-level Authorization
@router.get("/users/{user_id}/submissions")
async def get_user_submissions(
    user_id: UUID,
    current_user: User = Depends(get_current_user)
):
    # 본인 확인
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return await submission_service.get_by_user(user_id)

# 2. Object-level Authorization
class SubmissionService:
    async def get_submission(
        self,
        submission_id: UUID,
        requesting_user_id: UUID
    ) -> Submission:
        submission = await self.repo.find_by_id(submission_id)
        if submission.user_id != requesting_user_id:
            raise ForbiddenError("Not your submission")
        return submission

# 3. Role-based Access Control (RBAC)
class Permission(Enum):
    READ_PROBLEMS = "read:problems"
    SUBMIT_CODE = "submit:code"
    VIEW_SOLUTIONS = "view:solutions"
    ADMIN_USERS = "admin:users"

@require_permission(Permission.ADMIN_USERS)
async def delete_user(user_id: UUID):
    ...
```

### 2.2 A02: Cryptographic Failures

#### 위험
- 비밀번호 평문 저장
- 민감 데이터 노출

#### 대책

```python
# 1. Password Hashing (bcrypt)
from passlib.hash import bcrypt

class Password:
    @classmethod
    def create(cls, plain: str) -> 'Password':
        # bcrypt with cost factor 12
        return cls(bcrypt.using(rounds=12).hash(plain))

    def verify(self, plain: str) -> bool:
        return bcrypt.verify(plain, self.hashed_value)

# 2. JWT with Strong Secret
import secrets
from jose import jwt

JWT_SECRET = secrets.token_urlsafe(64)  # 512-bit
JWT_ALGORITHM = "HS256"

def create_token(data: dict, expires_delta: timedelta) -> str:
    expire = datetime.utcnow() + expires_delta
    to_encode = data.copy()
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

# 3. Sensitive Data Encryption at Rest
from cryptography.fernet import Fernet

class EncryptedField:
    def __init__(self, key: bytes):
        self.fernet = Fernet(key)

    def encrypt(self, value: str) -> str:
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt(self, encrypted: str) -> str:
        return self.fernet.decrypt(encrypted.encode()).decode()
```

### 2.3 A03: Injection

#### 위험
- SQL Injection
- Command Injection
- XSS

#### 대책

```python
# 1. SQL Injection Prevention (SQLAlchemy ORM)
# ❌ Bad
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ Good - Parameterized Query
stmt = select(User).where(User.email == email)
result = await session.execute(stmt)

# 2. Command Injection Prevention
# ❌ Bad
import os
os.system(f"python {user_code}")

# ✅ Good - Sandboxed Execution
class DockerSandbox:
    async def execute(self, code: str) -> ExecutionOutput:
        # Code is written to file, never executed directly
        container = await self.docker.containers.create(
            image="python-sandbox:latest",
            command=["python", "/app/solution.py"],
            stdin_open=False,
            network_disabled=True,  # No network
            read_only=True,         # Read-only filesystem
            mem_limit="256m",       # Memory limit
            cpu_quota=50000,        # CPU limit
        )
        ...

# 3. XSS Prevention (Frontend)
// ❌ Bad
element.innerHTML = userInput;

// ✅ Good - React auto-escapes
return <div>{userInput}</div>;

// ✅ Good - Sanitize if HTML needed
import DOMPurify from 'dompurify';
return <div dangerouslySetInnerHTML={{
    __html: DOMPurify.sanitize(userMarkdown)
}} />;
```

### 2.4 A04: Insecure Design

#### 대책

```python
# 1. Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")  # 로그인 시도 제한
async def login(request: Request):
    ...

@app.post("/api/v1/code/execute")
@limiter.limit("20/minute")  # 코드 실행 제한
async def execute_code(request: Request):
    ...

# 2. Account Lockout
class AuthService:
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=15)

    async def login(self, email: str, password: str) -> TokenDTO:
        user = await self.user_repo.find_by_email(email)

        if user.is_locked():
            raise AccountLockedError(user.lockout_until)

        if not user.authenticate(password):
            user.record_failed_attempt()
            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.lock(self.LOCKOUT_DURATION)
            await self.user_repo.save(user)
            raise InvalidCredentialsError()

        user.reset_failed_attempts()
        return self.generate_tokens(user)

# 3. Input Length Limits
class SubmitCommand(BaseModel):
    code: str = Field(..., max_length=50000)  # 50KB max

    @validator('code')
    def validate_code(cls, v):
        if len(v.encode('utf-8')) > 50000:
            raise ValueError('Code too large')
        return v
```

### 2.5 A05: Security Misconfiguration

#### 대책

```python
# 1. Secure Headers
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://codetutor.ai"],  # Specific origin
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline';"
    )
    return response

# 2. Error Handling (No Stack Traces)
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
                # No stack trace in production!
            }
        }
    )

# 3. Production Config
class Settings(BaseSettings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
if settings.DEBUG:
    raise RuntimeError("DEBUG mode enabled in production!")
```

### 2.6 A06: Vulnerable Components

#### 대책

```toml
# pyproject.toml - Pin versions

[project]
dependencies = [
    "fastapi>=0.109.0,<0.110.0",
    "sqlalchemy>=2.0.25,<2.1.0",
    "pydantic>=2.5.0,<2.6.0",
]

[tool.pip-audit]
# CI에서 취약점 스캔
```

```yaml
# .github/workflows/security.yml
- name: Python dependency audit
  run: uv pip audit

- name: npm audit
  run: npm audit --audit-level=high
```

### 2.7 A07: Identification and Authentication Failures

#### 대책

```python
# 1. Password Policy
class Password:
    MIN_LENGTH = 8
    REQUIRE_DIGIT = True
    REQUIRE_UPPER = True

    @classmethod
    def validate(cls, password: str) -> None:
        if len(password) < cls.MIN_LENGTH:
            raise ValueError(f"Password must be at least {cls.MIN_LENGTH} characters")
        if cls.REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain a digit")
        if cls.REQUIRE_UPPER and not any(c.isupper() for c in password):
            raise ValueError("Password must contain an uppercase letter")

# 2. Session Management
class TokenService:
    ACCESS_TOKEN_EXPIRE = timedelta(minutes=15)
    REFRESH_TOKEN_EXPIRE = timedelta(days=7)

    async def refresh_token(self, refresh_token: str) -> TokenDTO:
        # Validate refresh token
        payload = self.decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise InvalidTokenError()

        # Check if token is blacklisted
        if await self.is_blacklisted(refresh_token):
            raise TokenRevokedError()

        # Generate new tokens
        return self.generate_tokens(payload["sub"])

    async def logout(self, refresh_token: str) -> None:
        # Blacklist the refresh token
        await self.redis.setex(
            f"blacklist:{refresh_token}",
            self.REFRESH_TOKEN_EXPIRE,
            "1"
        )

# 3. Multi-factor Authentication (Future)
# 이메일/SMS 인증 코드 전송
```

### 2.8 A08: Software and Data Integrity Failures

#### 대책

```python
# 1. Subresource Integrity (Frontend)
<script
    src="https://cdn.example.com/lib.js"
    integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxIk..."
    crossorigin="anonymous"
></script>

# 2. Signed Artifacts (CI/CD)
# Docker image signing with Cosign
```

### 2.9 A09: Security Logging and Monitoring

#### 대책

```python
# 1. Security Event Logging
import structlog

logger = structlog.get_logger()

class AuthService:
    async def login(self, email: str, password: str) -> TokenDTO:
        try:
            user = await self.authenticate(email, password)
            logger.info(
                "login_success",
                user_id=str(user.id),
                email=email,
                ip=request.client.host
            )
            return self.generate_tokens(user)
        except InvalidCredentialsError:
            logger.warning(
                "login_failed",
                email=email,
                ip=request.client.host,
                reason="invalid_credentials"
            )
            raise

# 2. Security Alerts
class SecurityMonitor:
    async def check_brute_force(self, ip: str) -> None:
        attempts = await self.redis.incr(f"login_attempts:{ip}")
        await self.redis.expire(f"login_attempts:{ip}", 3600)

        if attempts > 10:
            logger.critical(
                "brute_force_detected",
                ip=ip,
                attempts=attempts
            )
            await self.alert_service.send(
                "Brute force attack detected",
                {"ip": ip, "attempts": attempts}
            )
```

### 2.10 A10: Server-Side Request Forgery (SSRF)

#### 대책

```python
# 1. URL Validation
import ipaddress
from urllib.parse import urlparse

BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]

def validate_url(url: str) -> None:
    parsed = urlparse(url)

    # Only allow http/https
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid URL scheme")

    # Resolve hostname
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
    except socket.gaierror:
        raise ValueError("Cannot resolve hostname")

    # Block internal networks
    for network in BLOCKED_NETWORKS:
        if ip in network:
            raise ValueError("Internal network access denied")
```

---

## 3. 코드 실행 샌드박스 보안

### 3.1 Docker 샌드박스 설정

```python
class DockerSandbox:
    """안전한 코드 실행 환경"""

    SECURITY_CONFIG = {
        "network_disabled": True,        # 네트워크 차단
        "read_only": True,               # 읽기 전용 파일시스템
        "mem_limit": "256m",             # 메모리 256MB
        "memswap_limit": "256m",         # 스왑 비활성화
        "cpu_quota": 50000,              # CPU 50%
        "cpu_period": 100000,
        "pids_limit": 50,                # 프로세스 수 제한
        "user": "nobody",                # 권한 없는 사용자
        "cap_drop": ["ALL"],             # 모든 capability 제거
        "security_opt": [
            "no-new-privileges:true",    # 권한 상승 방지
            "seccomp=sandbox-profile.json"  # seccomp 프로필
        ],
    }

    async def execute(self, code: str, stdin: str = "") -> ExecutionOutput:
        container = await self.docker.containers.create(
            image="python-sandbox:latest",
            command=["python", "/app/solution.py"],
            **self.SECURITY_CONFIG
        )

        try:
            # 타임아웃 적용
            result = await asyncio.wait_for(
                container.wait(),
                timeout=5.0
            )
            ...
        finally:
            await container.remove(force=True)
```

### 3.2 Python 샌드박스 제한

```python
# sandbox-entrypoint.py

import sys
import resource

# 1. 위험한 모듈 차단
BLOCKED_MODULES = {
    'os', 'subprocess', 'shutil', 'socket', 'requests',
    'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib',
    'pickle', 'marshal', 'ctypes', 'multiprocessing'
}

class SecureImporter:
    def find_module(self, name, path=None):
        if name.split('.')[0] in BLOCKED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed")
        return None

sys.meta_path.insert(0, SecureImporter())

# 2. 리소스 제한
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))      # CPU 5초
resource.setrlimit(resource.RLIMIT_NOFILE, (10, 10)) # 파일 10개
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))    # 프로세스 1개

# 3. 빌트인 함수 제한
BLOCKED_BUILTINS = ['exec', 'eval', 'compile', 'open', '__import__']
for name in BLOCKED_BUILTINS:
    if name in __builtins__:
        del __builtins__[name]
```

---

## 4. 인증 및 세션 관리

### 4.1 JWT 토큰 구조

```python
# Access Token (15분)
{
    "sub": "user_id",
    "email": "user@example.com",
    "type": "access",
    "iat": 1703577600,
    "exp": 1703578500
}

# Refresh Token (7일)
{
    "sub": "user_id",
    "type": "refresh",
    "jti": "unique_token_id",  # 토큰 식별자 (블랙리스트용)
    "iat": 1703577600,
    "exp": 1704182400
}
```

### 4.2 토큰 저장

```typescript
// Frontend - Secure Token Storage

// ❌ Bad: localStorage (XSS 취약)
localStorage.setItem('token', accessToken);

// ✅ Good: HttpOnly Cookie (서버 설정)
// 또는 메모리에만 저장 (새로고침 시 재인증)
class TokenStore {
    private accessToken: string | null = null;

    setToken(token: string) {
        this.accessToken = token;
    }

    getToken(): string | null {
        return this.accessToken;
    }

    clearToken() {
        this.accessToken = null;
    }
}
```

---

## 5. 데이터 보호

### 5.1 민감 데이터 분류

| 분류 | 데이터 | 보호 수준 |
|------|--------|-----------|
| **극비** | 비밀번호 | bcrypt 해시 |
| **기밀** | 이메일 | 암호화 저장 (선택) |
| **내부** | 제출 코드 | 접근 제어 |
| **공개** | 문제 설명 | 없음 |

### 5.2 GDPR 준수

```python
# 1. 데이터 삭제 요청
class UserService:
    async def delete_account(self, user_id: UserId) -> None:
        """GDPR Article 17 - Right to erasure"""
        # 관련 데이터 삭제
        await self.submission_repo.delete_by_user(user_id)
        await self.conversation_repo.delete_by_user(user_id)
        await self.progress_repo.delete_by_user(user_id)

        # 사용자 삭제
        await self.user_repo.delete(user_id)

        # 감사 로그
        logger.info("user_data_deleted", user_id=str(user_id))

# 2. 데이터 내보내기
class UserService:
    async def export_data(self, user_id: UserId) -> dict:
        """GDPR Article 20 - Right to data portability"""
        return {
            "user": await self.user_repo.find_by_id(user_id),
            "submissions": await self.submission_repo.find_by_user(user_id),
            "conversations": await self.conversation_repo.find_by_user(user_id),
            "progress": await self.progress_repo.find_by_user(user_id),
        }
```

---

## 6. 보안 체크리스트

### 6.1 개발 단계

- [ ] 입력값 검증 (길이, 형식, 범위)
- [ ] 출력값 이스케이프 (HTML, SQL)
- [ ] 인증/인가 검사
- [ ] 에러 메시지에 민감 정보 없음
- [ ] 로깅에 민감 정보 없음
- [ ] 의존성 취약점 스캔

### 6.2 배포 단계

- [ ] HTTPS 강제
- [ ] 보안 헤더 설정
- [ ] CORS 제한
- [ ] Rate Limiting 설정
- [ ] 환경 변수 보안
- [ ] 디버그 모드 비활성화

### 6.3 운영 단계

- [ ] 정기 보안 감사
- [ ] 침투 테스트
- [ ] 로그 모니터링
- [ ] 취약점 패치
- [ ] 백업 암호화
