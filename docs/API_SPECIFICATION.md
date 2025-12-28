# Code Tutor AI - API 명세서

## RESTful API Specification (OpenAPI 3.0)

**버전**: 1.0
**Base URL**: `https://api.codetutor.ai/v1`

---

## 목차

1. [인증 (Authentication)](#1-인증-authentication)
2. [사용자 (Users)](#2-사용자-users)
3. [문제 (Problems)](#3-문제-problems)
4. [제출 (Submissions)](#4-제출-submissions)
5. [AI 튜터 (AI Tutor)](#5-ai-튜터-ai-tutor)
6. [코드 실행 (Code Execution)](#6-코드-실행-code-execution)
7. [대시보드 (Dashboard)](#7-대시보드-dashboard)
8. [공통 응답](#8-공통-응답)

---

# 1. 인증 (Authentication)

## 1.1 회원가입

사용자 계정을 생성합니다.

```
POST /auth/register
```

### Request Body

```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "nickname": "코딩초보"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| email | string | ✅ | 이메일 (유효한 형식) |
| password | string | ✅ | 비밀번호 (8자 이상, 영문+숫자) |
| nickname | string | ✅ | 닉네임 (2-20자) |

### Response

**201 Created**
```json
{
  "success": true,
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "nickname": "코딩초보",
    "created_at": "2025-12-26T10:30:00Z"
  },
  "tokens": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 900
  }
}
```

**400 Bad Request**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "비밀번호는 8자 이상이어야 합니다.",
    "details": {
      "field": "password",
      "constraint": "min_length",
      "value": 8
    }
  }
}
```

**409 Conflict**
```json
{
  "success": false,
  "error": {
    "code": "EMAIL_ALREADY_EXISTS",
    "message": "이미 등록된 이메일입니다."
  }
}
```

---

## 1.2 로그인

JWT 토큰을 발급받습니다.

```
POST /auth/login
```

### Request Body

```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "user": {
      "user_id": "550e8400-e29b-41d4-a716-446655440000",
      "email": "user@example.com",
      "nickname": "코딩초보"
    },
    "tokens": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "Bearer",
      "expires_in": 900
    }
  }
}
```

**401 Unauthorized**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "이메일 또는 비밀번호가 올바르지 않습니다."
  }
}
```

---

## 1.3 토큰 갱신

Access Token을 갱신합니다.

```
POST /auth/refresh
```

### Request Headers

```
Authorization: Bearer {refresh_token}
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 900
  }
}
```

---

## 1.4 로그아웃

현재 세션을 종료합니다.

```
POST /auth/logout
```

### Request Headers

```
Authorization: Bearer {access_token}
```

### Response

**200 OK**
```json
{
  "success": true,
  "message": "로그아웃되었습니다."
}
```

---

# 2. 사용자 (Users)

## 2.1 내 정보 조회

로그인한 사용자의 정보를 조회합니다.

```
GET /users/me
```

### Request Headers

```
Authorization: Bearer {access_token}
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "nickname": "코딩초보",
    "avatar_url": null,
    "bio": null,
    "created_at": "2025-12-26T10:30:00Z",
    "stats": {
      "problems_solved": 15,
      "total_submissions": 42,
      "success_rate": 0.357,
      "streak_days": 7,
      "total_score": 350
    }
  }
}
```

---

## 2.2 프로필 수정

사용자 프로필을 수정합니다.

```
PUT /users/me
```

### Request Body

```json
{
  "nickname": "알고리즘마스터",
  "bio": "Python으로 코딩테스트 준비 중!"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| nickname | string | ❌ | 새 닉네임 (2-20자) |
| bio | string | ❌ | 자기소개 (최대 200자) |
| avatar_url | string | ❌ | 프로필 이미지 URL |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "nickname": "알고리즘마스터",
    "bio": "Python으로 코딩테스트 준비 중!",
    "updated_at": "2025-12-26T11:00:00Z"
  }
}
```

---

## 2.3 비밀번호 변경

사용자 비밀번호를 변경합니다.

```
PUT /users/me/password
```

### Request Body

```json
{
  "current_password": "oldPassword123",
  "new_password": "newSecurePassword456"
}
```

### Response

**200 OK**
```json
{
  "success": true,
  "message": "비밀번호가 변경되었습니다."
}
```

**400 Bad Request**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PASSWORD",
    "message": "현재 비밀번호가 올바르지 않습니다."
  }
}
```

---

# 3. 문제 (Problems)

## 3.1 문제 목록 조회

알고리즘 문제 목록을 조회합니다.

```
GET /problems
```

### Query Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| page | integer | 1 | 페이지 번호 |
| limit | integer | 20 | 페이지당 개수 (최대 50) |
| difficulty | string | - | 난이도 필터 (easy, medium, hard) |
| category | string | - | 카테고리 필터 |
| status | string | - | 풀이 상태 (solved, unsolved, attempted) |
| search | string | - | 제목 검색 |
| sort | string | created_at | 정렬 기준 (created_at, difficulty, title) |
| order | string | desc | 정렬 순서 (asc, desc) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "problems": [
      {
        "problem_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "title": "두 수의 합",
        "difficulty": "easy",
        "category": "array",
        "solved_count": 1234,
        "acceptance_rate": 0.72,
        "user_status": "solved",
        "score": 10
      },
      {
        "problem_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
        "title": "가장 긴 부분 문자열",
        "difficulty": "medium",
        "category": "string",
        "solved_count": 567,
        "acceptance_rate": 0.45,
        "user_status": "attempted",
        "score": 25
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 5,
      "total_count": 100,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

---

## 3.2 문제 상세 조회

특정 문제의 상세 정보를 조회합니다.

```
GET /problems/{problem_id}
```

### Path Parameters

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| problem_id | UUID | 문제 ID |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "problem_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "title": "두 수의 합",
    "description": "정수 배열 `nums`와 정수 `target`이 주어집니다. 두 수의 합이 `target`이 되는 인덱스를 반환하세요.\n\n각 입력에는 정확히 하나의 해가 존재하며, 같은 요소를 두 번 사용할 수 없습니다.",
    "difficulty": "easy",
    "category": "array",
    "constraints": [
      "2 <= nums.length <= 10^4",
      "-10^9 <= nums[i] <= 10^9",
      "-10^9 <= target <= 10^9",
      "정확히 하나의 해만 존재합니다."
    ],
    "examples": [
      {
        "input": "nums = [2, 7, 11, 15], target = 9",
        "output": "[0, 1]",
        "explanation": "nums[0] + nums[1] = 2 + 7 = 9 이므로 [0, 1]을 반환합니다."
      },
      {
        "input": "nums = [3, 2, 4], target = 6",
        "output": "[1, 2]",
        "explanation": "nums[1] + nums[2] = 2 + 4 = 6"
      }
    ],
    "test_cases": [
      {
        "input": "[2, 7, 11, 15]\n9",
        "expected_output": "[0, 1]",
        "is_visible": true
      },
      {
        "input": "[3, 2, 4]\n6",
        "expected_output": "[1, 2]",
        "is_visible": true
      }
    ],
    "hints_available": 3,
    "time_limit_ms": 5000,
    "memory_limit_mb": 256,
    "score": 10,
    "solved_count": 1234,
    "submission_count": 1712,
    "acceptance_rate": 0.72,
    "user_status": "solved",
    "tags": ["hash-table", "array"]
  }
}
```

**404 Not Found**
```json
{
  "success": false,
  "error": {
    "code": "PROBLEM_NOT_FOUND",
    "message": "문제를 찾을 수 없습니다."
  }
}
```

---

## 3.3 힌트 조회

문제의 힌트를 조회합니다. (점수 감점 적용)

```
GET /problems/{problem_id}/hints/{hint_order}
```

### Path Parameters

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| problem_id | UUID | 문제 ID |
| hint_order | integer | 힌트 순서 (1, 2, 3...) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "hint_order": 1,
    "content": "이 문제는 해시맵을 사용하면 O(n) 시간복잡도로 해결할 수 있습니다.",
    "penalty": 5,
    "remaining_score": 5
  },
  "meta": {
    "hints_used": 1,
    "hints_total": 3
  }
}
```

---

## 3.4 추천 문제 조회

AI가 추천하는 맞춤형 문제를 조회합니다.

```
GET /problems/recommended
```

### Query Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| count | integer | 5 | 추천 문제 수 (최대 10) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "problem_id": "c3d4e5f6-a7b8-9012-cdef-345678901234",
        "title": "연결 리스트 뒤집기",
        "difficulty": "easy",
        "category": "linked_list",
        "reason": "배열 문제를 많이 풀었으니 연결 리스트에 도전해보세요!",
        "confidence": 0.85,
        "predicted_success_rate": 0.75
      },
      {
        "problem_id": "d4e5f6a7-b8c9-0123-def0-456789012345",
        "title": "이진 탐색",
        "difficulty": "easy",
        "category": "binary_search",
        "reason": "정렬 문제의 다음 단계로 이진 탐색을 추천합니다.",
        "confidence": 0.82,
        "predicted_success_rate": 0.70
      }
    ],
    "model_version": "ncf-v1.0"
  }
}
```

---

# 4. 제출 (Submissions)

## 4.1 코드 제출

문제에 대한 코드를 제출합니다.

```
POST /problems/{problem_id}/submit
```

### Path Parameters

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| problem_id | UUID | 문제 ID |

### Request Body

```json
{
  "code": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
  "language": "python"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| code | string | ✅ | 제출할 코드 (최대 50000자) |
| language | string | ❌ | 언어 (기본값: python) |

### Response

**202 Accepted** (비동기 처리)
```json
{
  "success": true,
  "data": {
    "submission_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
    "status": "pending",
    "message": "코드가 제출되었습니다. 채점 중입니다..."
  }
}
```

---

## 4.2 제출 결과 조회

제출 결과를 조회합니다.

```
GET /submissions/{submission_id}
```

### Response

**200 OK** (채점 완료)
```json
{
  "success": true,
  "data": {
    "submission_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
    "problem_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "completed",
    "verdict": "accepted",
    "runtime_ms": 32,
    "memory_kb": 14320,
    "passed_tests": 10,
    "total_tests": 10,
    "score": 10,
    "submitted_at": "2025-12-26T12:00:00Z",
    "test_results": [
      {
        "test_number": 1,
        "status": "passed",
        "runtime_ms": 28,
        "memory_kb": 14200,
        "input": "[2, 7, 11, 15]\n9",
        "expected_output": "[0, 1]",
        "actual_output": "[0, 1]"
      },
      {
        "test_number": 2,
        "status": "passed",
        "runtime_ms": 32,
        "memory_kb": 14320,
        "input": "[3, 2, 4]\n6",
        "expected_output": "[1, 2]",
        "actual_output": "[1, 2]"
      }
    ]
  }
}
```

**200 OK** (채점 중)
```json
{
  "success": true,
  "data": {
    "submission_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
    "status": "running",
    "progress": 4,
    "total_tests": 10,
    "message": "테스트 케이스 4/10 실행 중..."
  }
}
```

**200 OK** (실패)
```json
{
  "success": true,
  "data": {
    "submission_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
    "status": "completed",
    "verdict": "wrong_answer",
    "runtime_ms": 45,
    "memory_kb": 15000,
    "passed_tests": 7,
    "total_tests": 10,
    "score": 0,
    "failed_test": {
      "test_number": 8,
      "input": "[3, 3]\n6",
      "expected_output": "[0, 1]",
      "actual_output": "[]"
    }
  }
}
```

---

## 4.3 내 제출 기록 조회

사용자의 제출 기록을 조회합니다.

```
GET /submissions
```

### Query Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| page | integer | 1 | 페이지 번호 |
| limit | integer | 20 | 페이지당 개수 |
| problem_id | UUID | - | 특정 문제의 제출만 조회 |
| verdict | string | - | 결과 필터 (accepted, wrong_answer, etc.) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "submissions": [
      {
        "submission_id": "e5f6a7b8-c9d0-1234-ef01-567890123456",
        "problem_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "problem_title": "두 수의 합",
        "verdict": "accepted",
        "runtime_ms": 32,
        "memory_kb": 14320,
        "score": 10,
        "submitted_at": "2025-12-26T12:00:00Z"
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 3,
      "total_count": 42
    }
  }
}
```

---

# 5. AI 튜터 (AI Tutor)

## 5.1 대화 시작/계속

AI 튜터와 대화를 시작하거나 계속합니다.

```
POST /ai/chat
```

### Request Body

```json
{
  "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
  "message": "퀵 정렬 알고리즘이 뭐야?"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| conversation_id | UUID | ❌ | 대화 ID (새 대화시 생략) |
| message | string | ✅ | 사용자 메시지 |
| context_problem_id | UUID | ❌ | 관련 문제 ID (문제 맥락 제공) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
    "message": {
      "role": "assistant",
      "content": "퀵 정렬(Quick Sort)은 **분할 정복(Divide and Conquer)** 기법을 사용하는 정렬 알고리즘입니다.\n\n## 핵심 아이디어\n1. **피벗(Pivot)** 선택: 배열에서 하나의 요소를 선택합니다.\n2. **분할(Partition)**: 피벗보다 작은 요소는 왼쪽, 큰 요소는 오른쪽으로 나눕니다.\n3. **재귀(Recursion)**: 분할된 두 부분에 대해 재귀적으로 퀵 정렬을 수행합니다.\n\n## Python 구현\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quick_sort(left) + middle + quick_sort(right)\n```\n\n## 시간 복잡도\n- 평균: **O(n log n)**\n- 최악: **O(n²)** (이미 정렬된 배열에서 피벗이 항상 최솟값/최댓값일 때)\n\n더 궁금한 점이 있으신가요?",
      "created_at": "2025-12-26T12:30:00Z"
    },
    "tokens_used": 412
  }
}
```

---

## 5.2 대화 기록 조회

이전 대화 기록을 조회합니다.

```
GET /ai/conversations
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "conversations": [
      {
        "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
        "topic": "퀵 정렬 알고리즘이 뭐야?",
        "message_count": 8,
        "created_at": "2025-12-26T12:30:00Z",
        "updated_at": "2025-12-26T13:00:00Z"
      }
    ]
  }
}
```

---

## 5.3 특정 대화 조회

```
GET /ai/conversations/{conversation_id}
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
    "messages": [
      {
        "role": "user",
        "content": "퀵 정렬 알고리즘이 뭐야?",
        "created_at": "2025-12-26T12:30:00Z"
      },
      {
        "role": "assistant",
        "content": "퀵 정렬(Quick Sort)은...",
        "created_at": "2025-12-26T12:30:05Z"
      }
    ],
    "created_at": "2025-12-26T12:30:00Z"
  }
}
```

---

## 5.4 코드 리뷰 요청

AI에게 코드 리뷰를 요청합니다.

```
POST /ai/review
```

### Request Body

```json
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "context": "피보나치 수열을 구현했는데, 더 효율적인 방법이 있을까요?"
}
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "review_id": "a7b8c9d0-e1f2-3456-0123-789012345678",
    "complexity": {
      "time_complexity": "O(2^n)",
      "space_complexity": "O(n)",
      "explanation": "재귀 호출이 중복되어 지수적으로 증가합니다. 예를 들어 fibonacci(5)를 계산하면 fibonacci(3)이 2번, fibonacci(2)가 3번 호출됩니다."
    },
    "overall_score": 45,
    "suggestions": [
      {
        "severity": "error",
        "line_number": null,
        "message": "시간 복잡도가 O(2^n)으로 매우 비효율적입니다. n이 40만 되어도 수십 초가 걸립니다.",
        "suggested_code": null
      },
      {
        "severity": "warning",
        "line_number": null,
        "message": "메모이제이션을 사용하면 O(n)으로 개선할 수 있습니다.",
        "suggested_code": "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]"
      },
      {
        "severity": "info",
        "line_number": null,
        "message": "반복문을 사용하면 공간 복잡도도 O(1)로 개선됩니다.",
        "suggested_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
      }
    ],
    "quality_scores": {
      "complexity_score": 0.85,
      "readability_score": 0.90,
      "bug_risk_score": 0.10,
      "optimization_potential": 0.95
    }
  }
}
```

---

# 6. 코드 실행 (Code Execution)

## 6.1 코드 실행

테스트 실행 (제출 없이 결과만 확인)

```
POST /code/execute
```

### Request Body

```json
{
  "code": "def solution(nums, target):\n    return [0, 1]\n\nprint(solution([2, 7, 11, 15], 9))",
  "language": "python",
  "stdin": ""
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| code | string | ✅ | 실행할 코드 |
| language | string | ❌ | 언어 (기본값: python) |
| stdin | string | ❌ | 표준 입력 |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "execution_id": "b8c9d0e1-f2a3-4567-1234-890123456789",
    "status": "success",
    "stdout": "[0, 1]\n",
    "stderr": "",
    "exit_code": 0,
    "runtime_ms": 28,
    "memory_kb": 13456
  }
}
```

**200 OK** (런타임 에러)
```json
{
  "success": true,
  "data": {
    "execution_id": "b8c9d0e1-f2a3-4567-1234-890123456789",
    "status": "runtime_error",
    "stdout": "",
    "stderr": "Traceback (most recent call last):\n  File \"solution.py\", line 3, in <module>\n    print(nums[10])\nIndexError: list index out of range",
    "exit_code": 1,
    "runtime_ms": 15,
    "memory_kb": 12800
  }
}
```

**200 OK** (시간 초과)
```json
{
  "success": true,
  "data": {
    "execution_id": "b8c9d0e1-f2a3-4567-1234-890123456789",
    "status": "timeout",
    "stdout": "",
    "stderr": "Time limit exceeded",
    "exit_code": -1,
    "runtime_ms": 5000,
    "memory_kb": 0
  }
}
```

---

# 7. 대시보드 (Dashboard)

## 7.1 학습 통계 조회

```
GET /dashboard/stats
```

### Query Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| period | string | week | 기간 (day, week, month, all) |

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "overview": {
      "problems_solved": 15,
      "total_submissions": 42,
      "success_rate": 0.357,
      "streak_days": 7,
      "total_score": 350,
      "rank_percentile": 0.25
    },
    "by_category": [
      {
        "category": "array",
        "solved": 5,
        "total": 10,
        "progress": 0.5
      },
      {
        "category": "linked_list",
        "solved": 2,
        "total": 8,
        "progress": 0.25
      },
      {
        "category": "tree",
        "solved": 3,
        "total": 12,
        "progress": 0.25
      }
    ],
    "by_difficulty": {
      "easy": {"solved": 10, "total": 15},
      "medium": {"solved": 4, "total": 25},
      "hard": {"solved": 1, "total": 10}
    },
    "activity_heatmap": [
      {"date": "2025-12-20", "count": 3},
      {"date": "2025-12-21", "count": 5},
      {"date": "2025-12-22", "count": 2},
      {"date": "2025-12-23", "count": 4},
      {"date": "2025-12-24", "count": 0},
      {"date": "2025-12-25", "count": 1},
      {"date": "2025-12-26", "count": 3}
    ],
    "weak_categories": [
      {
        "category": "linked_list",
        "progress": 0.25,
        "recommendation": "연결 리스트 기초부터 다시 복습해보세요!"
      }
    ]
  }
}
```

---

## 7.2 학습 예측 조회

AI의 학습 성과 예측을 조회합니다.

```
GET /dashboard/prediction
```

### Response

**200 OK**
```json
{
  "success": true,
  "data": {
    "current_success_rate": 0.357,
    "predicted_success_rate": 0.42,
    "prediction_period": "next_week",
    "confidence": 0.78,
    "insights": [
      "현재 추세라면 다음 주에 성공률이 6% 상승할 것으로 예상됩니다.",
      "연결 리스트 문제를 3개 더 풀면 카테고리 완료율이 50%에 도달합니다.",
      "현재 7일 연속 학습 중입니다. 좋은 습관이에요!"
    ],
    "recommendations": [
      {
        "type": "practice",
        "message": "화요일에 스택/큐 문제 복습을 추천드립니다.",
        "reason": "지난주 화요일 학습 효율이 가장 높았습니다."
      },
      {
        "type": "review",
        "message": "3일 전에 틀린 '가장 긴 부분 문자열' 문제를 다시 풀어보세요.",
        "problem_id": "c9d0e1f2-a3b4-5678-2345-901234567890"
      }
    ],
    "model_version": "lstm-v1.0"
  }
}
```

---

# 8. 공통 응답

## 8.1 성공 응답 형식

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req-123456",
    "timestamp": "2025-12-26T12:00:00Z"
  }
}
```

## 8.2 에러 응답 형식

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "사용자 친화적인 에러 메시지",
    "details": { ... }
  },
  "meta": {
    "request_id": "req-123456",
    "timestamp": "2025-12-26T12:00:00Z"
  }
}
```

## 8.3 에러 코드 목록

### 4xx Client Errors

| 코드 | HTTP Status | 설명 |
|------|-------------|------|
| VALIDATION_ERROR | 400 | 입력값 검증 실패 |
| INVALID_CREDENTIALS | 401 | 인증 실패 |
| TOKEN_EXPIRED | 401 | 토큰 만료 |
| FORBIDDEN | 403 | 권한 없음 |
| PROBLEM_NOT_FOUND | 404 | 문제 없음 |
| USER_NOT_FOUND | 404 | 사용자 없음 |
| SUBMISSION_NOT_FOUND | 404 | 제출 없음 |
| EMAIL_ALREADY_EXISTS | 409 | 이메일 중복 |
| RATE_LIMIT_EXCEEDED | 429 | 요청 제한 초과 |

### 5xx Server Errors

| 코드 | HTTP Status | 설명 |
|------|-------------|------|
| INTERNAL_ERROR | 500 | 내부 서버 오류 |
| LLM_ERROR | 500 | AI 모델 오류 |
| SANDBOX_ERROR | 500 | 코드 실행 오류 |
| DATABASE_ERROR | 500 | 데이터베이스 오류 |
| SERVICE_UNAVAILABLE | 503 | 서비스 일시 중단 |

## 8.4 Rate Limiting

| 엔드포인트 | 제한 |
|------------|------|
| POST /ai/chat | 10회/분 |
| POST /ai/review | 5회/분 |
| POST /code/execute | 20회/분 |
| POST /problems/*/submit | 10회/분 |
| 기타 | 60회/분 |

### Rate Limit Headers

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1703592000
```

## 8.5 인증 헤더

모든 인증이 필요한 요청에는 다음 헤더가 필요합니다:

```
Authorization: Bearer {access_token}
```

---

# 부록: WebSocket API

## 실시간 채팅

```
WebSocket /ws/chat
```

### 연결

```javascript
const ws = new WebSocket('wss://api.codetutor.ai/ws/chat?token={access_token}');
```

### 메시지 전송

```json
{
  "type": "message",
  "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567",
  "content": "퀵 정렬에서 피벗 선택이 왜 중요해?"
}
```

### 응답 수신 (스트리밍)

```json
{
  "type": "stream_start",
  "conversation_id": "f6a7b8c9-d0e1-2345-f012-678901234567"
}
```

```json
{
  "type": "stream_chunk",
  "content": "피벗 선택은 "
}
```

```json
{
  "type": "stream_chunk",
  "content": "퀵 정렬의 성능에 "
}
```

```json
{
  "type": "stream_end",
  "tokens_used": 156
}
```

---

# 부록: OpenAPI Schema

전체 OpenAPI 3.0 스키마는 다음 위치에서 확인할 수 있습니다:

```
GET /openapi.json
GET /docs        # Swagger UI
GET /redoc       # ReDoc
```
