"""문자열 메서드 기초 문제 시딩 스크립트

10개의 문자열 기초 문제 추가:
- Easy 7개: len, upper/lower, split, join, strip, find, replace
- Medium 3개: slicing, formatting, regex
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"

STRING_BASICS_PROBLEMS = [
    # ============== EASY 문제들 ==============
    {
        "title": "문자열 길이 구하기",
        "description": """문자열의 길이를 구하는 함수를 작성하세요.

### 입력
- 문자열 `s`

### 출력
- 문자열의 길이 (정수)

### 예제
```
입력: "Hello, World!"
출력: 13
```

### 힌트
- Python의 `len()` 함수를 사용하세요.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["len() 함수 사용", "공백도 문자로 계산"],
        "solution_template": "def solution(s: str) -> int:\n    pass",
        "reference_solution": """def solution(s: str) -> int:
    return len(s)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics"],
        "pattern_explanation": "len() 함수는 문자열의 길이를 반환하는 가장 기본적인 함수입니다.",
        "approach_hint": "len() 함수 사용",
        "time_complexity_hint": "O(1)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "Hello, World!", "output": "13", "is_sample": True},
            {"input": "", "output": "0", "is_sample": False},
            {"input": "Python", "output": "6", "is_sample": False},
        ]
    },
    {
        "title": "대소문자 변환",
        "description": """주어진 문자열을 요청에 따라 대문자 또는 소문자로 변환하세요.

### 입력
- 문자열 `s`
- 변환 타입 `convert_type`: "upper", "lower", "capitalize", "title"

### 출력
- 변환된 문자열

### 예제
```
입력: s = "hello world", convert_type = "upper"
출력: "HELLO WORLD"

입력: s = "HELLO WORLD", convert_type = "capitalize"
출력: "Hello world"
```

### 메서드 설명
- `upper()`: 모두 대문자
- `lower()`: 모두 소문자
- `capitalize()`: 첫 글자만 대문자
- `title()`: 각 단어 첫 글자 대문자""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["upper(), lower(), capitalize(), title() 메서드 활용", "조건문으로 타입 분기"],
        "solution_template": "def solution(s: str, convert_type: str) -> str:\n    pass",
        "reference_solution": """def solution(s: str, convert_type: str) -> str:
    if convert_type == "upper":
        return s.upper()
    elif convert_type == "lower":
        return s.lower()
    elif convert_type == "capitalize":
        return s.capitalize()
    elif convert_type == "title":
        return s.title()
    return s""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-case"],
        "pattern_explanation": "문자열 대소문자 변환은 텍스트 정규화의 기본입니다.",
        "approach_hint": "문자열 메서드 사용",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "hello world\nupper", "output": "HELLO WORLD", "is_sample": True},
            {"input": "HELLO WORLD\ncapitalize", "output": "Hello world", "is_sample": False},
            {"input": "hello python\ntitle", "output": "Hello Python", "is_sample": False},
        ]
    },
    {
        "title": "문자열 분할",
        "description": """주어진 구분자로 문자열을 분할하세요.

### 입력
- 문자열 `s`
- 구분자 `delimiter`

### 출력
- 분할된 문자열 리스트

### 예제
```
입력: s = "apple,banana,cherry", delimiter = ","
출력: ["apple", "banana", "cherry"]

입력: s = "hello world python", delimiter = " "
출력: ["hello", "world", "python"]
```

### 힌트
- `split()` 메서드를 사용하세요.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["split(delimiter) 사용", "delimiter가 없으면 공백으로 분할"],
        "solution_template": "def solution(s: str, delimiter: str) -> list:\n    pass",
        "reference_solution": """def solution(s: str, delimiter: str) -> list:
    return s.split(delimiter)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-split"],
        "pattern_explanation": "split()은 문자열을 리스트로 변환하는 핵심 메서드입니다.",
        "approach_hint": "split() 메서드",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "apple,banana,cherry\n,", "output": '["apple", "banana", "cherry"]', "is_sample": True},
            {"input": "hello world python\n ", "output": '["hello", "world", "python"]', "is_sample": False},
        ]
    },
    {
        "title": "문자열 결합",
        "description": """리스트의 문자열들을 구분자로 연결하세요.

### 입력
- 문자열 리스트 `words`
- 구분자 `delimiter`

### 출력
- 결합된 문자열

### 예제
```
입력: words = ["apple", "banana", "cherry"], delimiter = ", "
출력: "apple, banana, cherry"

입력: words = ["Hello", "World"], delimiter = " "
출력: "Hello World"
```

### 힌트
- `join()` 메서드를 사용하세요.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(words) <= 10^4",
        "hints": ["delimiter.join(words) 형태", "join은 구분자에서 호출"],
        "solution_template": "def solution(words: list, delimiter: str) -> str:\n    pass",
        "reference_solution": """def solution(words: list, delimiter: str) -> str:
    return delimiter.join(words)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-join"],
        "pattern_explanation": "join()은 리스트를 문자열로 합치는 효율적인 방법입니다. + 연산보다 빠릅니다.",
        "approach_hint": "join() 메서드",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": '["apple", "banana", "cherry"]\n, ', "output": "apple, banana, cherry", "is_sample": True},
            {"input": '["Hello", "World"]\n ', "output": "Hello World", "is_sample": False},
        ]
    },
    {
        "title": "문자열 공백 제거",
        "description": """문자열의 앞뒤 공백을 제거하세요.

### 입력
- 문자열 `s`
- 제거 타입 `strip_type`: "both", "left", "right"

### 출력
- 공백이 제거된 문자열

### 예제
```
입력: s = "  Hello World  ", strip_type = "both"
출력: "Hello World"

입력: s = "  Python  ", strip_type = "left"
출력: "Python  "
```

### 메서드 설명
- `strip()`: 양쪽 공백 제거
- `lstrip()`: 왼쪽 공백 제거
- `rstrip()`: 오른쪽 공백 제거""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["strip(), lstrip(), rstrip() 사용", "조건문으로 타입 분기"],
        "solution_template": "def solution(s: str, strip_type: str) -> str:\n    pass",
        "reference_solution": """def solution(s: str, strip_type: str) -> str:
    if strip_type == "both":
        return s.strip()
    elif strip_type == "left":
        return s.lstrip()
    elif strip_type == "right":
        return s.rstrip()
    return s""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-strip"],
        "pattern_explanation": "strip 계열 메서드는 사용자 입력 정제에 필수적입니다.",
        "approach_hint": "strip(), lstrip(), rstrip()",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "  Hello World  \nboth", "output": "Hello World", "is_sample": True},
            {"input": "  Python  \nleft", "output": "Python  ", "is_sample": False},
            {"input": "  Python  \nright", "output": "  Python", "is_sample": False},
        ]
    },
    {
        "title": "문자열 찾기",
        "description": """문자열 내에서 특정 부분 문자열의 위치를 찾으세요.

### 입력
- 문자열 `s`
- 찾을 문자열 `target`

### 출력
- 처음 발견된 위치 (인덱스), 없으면 -1

### 예제
```
입력: s = "Hello, World!", target = "World"
출력: 7

입력: s = "Python Programming", target = "Java"
출력: -1
```

### 힌트
- `find()` 메서드를 사용하세요.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["find()는 없으면 -1 반환", "index()는 없으면 예외 발생"],
        "solution_template": "def solution(s: str, target: str) -> int:\n    pass",
        "reference_solution": """def solution(s: str, target: str) -> int:
    return s.find(target)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-find"],
        "pattern_explanation": "find()는 부분 문자열 검색의 기본입니다. index()와 달리 예외를 발생시키지 않습니다.",
        "approach_hint": "find() 메서드",
        "time_complexity_hint": "O(n*m)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "Hello, World!\nWorld", "output": "7", "is_sample": True},
            {"input": "Python Programming\nJava", "output": "-1", "is_sample": False},
        ]
    },
    {
        "title": "문자열 치환",
        "description": """문자열 내의 특정 부분을 다른 문자열로 치환하세요.

### 입력
- 문자열 `s`
- 찾을 문자열 `old`
- 바꿀 문자열 `new`

### 출력
- 치환된 문자열

### 예제
```
입력: s = "Hello, World!", old = "World", new = "Python"
출력: "Hello, Python!"

입력: s = "apple apple apple", old = "apple", new = "orange"
출력: "orange orange orange"
```

### 힌트
- `replace()` 메서드를 사용하세요.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "0 <= len(s) <= 10^4",
        "hints": ["replace(old, new) 사용", "모든 occurrence를 치환"],
        "solution_template": "def solution(s: str, old: str, new: str) -> str:\n    pass",
        "reference_solution": """def solution(s: str, old: str, new: str) -> str:
    return s.replace(old, new)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-basics", "string-replace"],
        "pattern_explanation": "replace()는 문자열 치환의 기본 메서드입니다. 정규식이 필요 없는 단순 치환에 적합합니다.",
        "approach_hint": "replace() 메서드",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "Hello, World!\nWorld\nPython", "output": "Hello, Python!", "is_sample": True},
            {"input": "apple apple apple\napple\norange", "output": "orange orange orange", "is_sample": False},
        ]
    },
    # ============== MEDIUM 문제들 ==============
    {
        "title": "문자열 슬라이싱",
        "description": """문자열을 슬라이싱하여 부분 문자열을 추출하세요.

### 입력
- 문자열 `s`
- 시작 인덱스 `start` (포함)
- 끝 인덱스 `end` (미포함)
- 스텝 `step`

### 출력
- 슬라이싱된 문자열

### 예제
```
입력: s = "Hello, World!", start = 0, end = 5, step = 1
출력: "Hello"

입력: s = "Python", start = 0, end = 6, step = 2
출력: "Pto"

입력: s = "Reverse", start = -1, end = None, step = -1
출력: "esreveR"
```

### 힌트
- `s[start:end:step]` 형태로 슬라이싱
- step이 음수면 역순""",
        "difficulty": "medium",
        "category": "string",
        "constraints": "-10^4 <= start, end <= 10^4",
        "hints": ["s[start:end:step] 문법", "음수 인덱스는 뒤에서부터", "None은 끝까지"],
        "solution_template": "def solution(s: str, start: int, end, step: int) -> str:\n    pass",
        "reference_solution": """def solution(s: str, start: int, end, step: int) -> str:
    if end is None:
        return s[start::step]
    return s[start:end:step]""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-slicing"],
        "pattern_explanation": "슬라이싱은 Python의 강력한 기능입니다. 문자열 뒤집기, 부분 추출 등에 활용됩니다.",
        "approach_hint": "[start:end:step] 슬라이싱",
        "time_complexity_hint": "O(k) - k는 결과 길이",
        "space_complexity_hint": "O(k)",
        "test_cases": [
            {"input": "Hello, World!\n0\n5\n1", "output": "Hello", "is_sample": True},
            {"input": "Python\n0\n6\n2", "output": "Pto", "is_sample": False},
        ]
    },
    {
        "title": "문자열 포맷팅",
        "description": """다양한 값들을 문자열로 포맷팅하세요.

### 입력
- 이름 `name`
- 나이 `age`
- 점수 `score` (소수점)

### 출력
- 포맷팅된 문자열: "이름: {name}, 나이: {age}세, 점수: {score:.2f}점"

### 예제
```
입력: name = "홍길동", age = 25, score = 95.5
출력: "이름: 홍길동, 나이: 25세, 점수: 95.50점"

입력: name = "김철수", age = 30, score = 88.333
출력: "이름: 김철수, 나이: 30세, 점수: 88.33점"
```

### 힌트
- f-string 또는 format() 메서드 사용
- 소수점 2자리: `{value:.2f}`""",
        "difficulty": "medium",
        "category": "string",
        "constraints": "0 <= age <= 150",
        "hints": ["f-string: f'{변수}'", "소수점 포맷: .2f", ".format() 메서드도 가능"],
        "solution_template": "def solution(name: str, age: int, score: float) -> str:\n    pass",
        "reference_solution": """def solution(name: str, age: int, score: float) -> str:
    return f"이름: {name}, 나이: {age}세, 점수: {score:.2f}점\"""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["string-formatting"],
        "pattern_explanation": "f-string은 Python 3.6+에서 가장 권장되는 문자열 포맷팅 방법입니다.",
        "approach_hint": "f-string 포맷팅",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "홍길동\n25\n95.5", "output": "이름: 홍길동, 나이: 25세, 점수: 95.50점", "is_sample": True},
            {"input": "김철수\n30\n88.333", "output": "이름: 김철수, 나이: 30세, 점수: 88.33점", "is_sample": False},
        ]
    },
    {
        "title": "정규식 기초",
        "description": """정규식을 사용하여 문자열에서 패턴을 찾으세요.

### 입력
- 문자열 `text`
- 찾을 패턴 `pattern` ("email", "phone", "number")

### 출력
- 패턴에 맞는 모든 문자열 리스트

### 패턴 설명
- `email`: 이메일 형식 (예: test@example.com)
- `phone`: 전화번호 (예: 010-1234-5678)
- `number`: 숫자 (연속된 숫자)

### 예제
```
입력: text = "연락처: test@email.com, 010-1234-5678", pattern = "email"
출력: ["test@email.com"]

입력: text = "가격: 10000원, 수량: 5개", pattern = "number"
출력: ["10000", "5"]
```

### 힌트
- `re` 모듈의 `findall()` 함수 사용
- 이메일: `r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'`""",
        "difficulty": "medium",
        "category": "string",
        "constraints": "0 <= len(text) <= 10^4",
        "hints": ["import re", "re.findall(pattern, text)", "이메일/전화번호 정규식 패턴 사용"],
        "solution_template": "def solution(text: str, pattern: str) -> list:\n    pass",
        "reference_solution": """def solution(text: str, pattern: str) -> list:
    import re

    patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}',
        "phone": r'\\d{2,3}-\\d{3,4}-\\d{4}',
        "number": r'\\d+'
    }

    regex = patterns.get(pattern, pattern)
    return re.findall(regex, text)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 128,
        "pattern_ids": ["regex-basics"],
        "pattern_explanation": "정규식은 복잡한 문자열 패턴 매칭에 필수적입니다. findall()은 모든 매칭을 리스트로 반환합니다.",
        "approach_hint": "re.findall() 사용",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(m) - m은 매칭 수",
        "test_cases": [
            {"input": "연락처: test@email.com, 010-1234-5678\nemail", "output": '["test@email.com"]', "is_sample": True},
            {"input": "가격: 10000원, 수량: 5개\nnumber", "output": '["10000", "5"]', "is_sample": False},
        ]
    },
]


def seed_problems():
    """문자열 기초 문제들을 데이터베이스에 삽입"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 이미 존재하는 문제 확인
    cursor.execute("SELECT title FROM problems WHERE category = 'string'")
    existing = {row[0] for row in cursor.fetchall()}

    added = 0
    skipped = 0

    for problem in STRING_BASICS_PROBLEMS:
        if problem["title"] in existing:
            print(f"  [SKIP] {problem['title']} - 이미 존재")
            skipped += 1
            continue

        problem_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # 문제 삽입
        cursor.execute("""
            INSERT INTO problems (
                id, title, description, difficulty, category,
                constraints, hints, solution_template, reference_solution,
                time_limit_ms, memory_limit_mb, pattern_ids, pattern_explanation,
                approach_hint, time_complexity_hint, space_complexity_hint,
                is_published, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_id,
            problem["title"],
            problem["description"],
            problem["difficulty"],
            problem["category"],
            problem["constraints"],
            json.dumps(problem["hints"], ensure_ascii=False),
            problem["solution_template"],
            problem["reference_solution"],
            problem["time_limit_ms"],
            problem["memory_limit_mb"],
            json.dumps(problem["pattern_ids"], ensure_ascii=False),
            problem["pattern_explanation"],
            problem["approach_hint"],
            problem["time_complexity_hint"],
            problem["space_complexity_hint"],
            True,  # is_published
            now,
            now
        ))

        # 테스트 케이스 삽입
        for i, tc in enumerate(problem["test_cases"]):
            tc_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO test_cases (
                    id, problem_id, input_data, expected_output,
                    is_sample, "order", created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tc_id,
                problem_id,
                tc["input"],
                tc["output"],
                tc["is_sample"],
                i,
                now
            ))

        print(f"  [ADD] {problem['title']}")
        added += 1

    conn.commit()
    conn.close()

    print(f"\n완료: {added}개 추가, {skipped}개 건너뜀")
    return added, skipped


if __name__ == "__main__":
    print("=" * 50)
    print("문자열 기초 문제 시딩")
    print("=" * 50)
    seed_problems()
