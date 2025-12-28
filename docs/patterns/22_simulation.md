# Pattern 22: Simulation (구현 / 시뮬레이션)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (삼성 SW 역량 필수) |
| **시간복잡도** | 문제마다 다름 |
| **공간복잡도** | O(N²) 보통 |
| **선행 지식** | 구현력, 조건문, 반복문 |

## 정의

**시뮬레이션**은 문제에서 주어진 **규칙/조건을 그대로 코드로 구현**하는 유형입니다. 알고리즘적 기법보다 **정확한 구현력**이 핵심입니다.

## 삼성 SW 역량 테스트 특징

```
✅ 2D 그리드 조작
✅ 복잡한 조건 처리
✅ 방향 전환 (상하좌우, 대각선)
✅ 여러 객체 동시 이동
✅ 시간 순서대로 진행
✅ 특정 조건에서 상태 변화
```

## 핵심 기법

| 기법 | 용도 |
|------|------|
| 방향 배열 | 상하좌우 이동 |
| 좌표 변환 | 회전, 대칭 |
| 상태 관리 | 객체별 정보 저장 |
| 시간 순서 | 라운드별 처리 |

---

## 템플릿 코드

### 템플릿 1: 방향 이동 (4방향 / 8방향)

```python
# 4방향: 상, 우, 하, 좌
DR4 = [-1, 0, 1, 0]
DC4 = [0, 1, 0, -1]

# 8방향: 상, 상우, 우, 하우, 하, 하좌, 좌, 상좌
DR8 = [-1, -1, 0, 1, 1, 1, 0, -1]
DC8 = [0, 1, 1, 1, 0, -1, -1, -1]

def move(r: int, c: int, direction: int) -> tuple:
    """direction 방향으로 한 칸 이동"""
    return r + DR4[direction], c + DC4[direction]

def is_valid(r: int, c: int, rows: int, cols: int) -> bool:
    """범위 체크"""
    return 0 <= r < rows and 0 <= c < cols

def turn_right(direction: int) -> int:
    """90도 오른쪽 회전"""
    return (direction + 1) % 4

def turn_left(direction: int) -> int:
    """90도 왼쪽 회전"""
    return (direction + 3) % 4

def turn_back(direction: int) -> int:
    """180도 회전"""
    return (direction + 2) % 4
```

### 템플릿 2: 2D 배열 회전

```python
def rotate_90_clockwise(matrix: list) -> list:
    """2D 배열 90도 시계방향 회전"""
    n = len(matrix)
    m = len(matrix[0])

    # n×m → m×n
    rotated = [[0] * n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            rotated[j][n - 1 - i] = matrix[i][j]

    return rotated

def rotate_90_counter_clockwise(matrix: list) -> list:
    """90도 반시계방향 회전"""
    n = len(matrix)
    m = len(matrix[0])

    rotated = [[0] * n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            rotated[m - 1 - j][i] = matrix[i][j]

    return rotated

def rotate_180(matrix: list) -> list:
    """180도 회전"""
    return [row[::-1] for row in matrix[::-1]]

def transpose(matrix: list) -> list:
    """전치 (대각선 기준 뒤집기)"""
    n = len(matrix)
    m = len(matrix[0])
    return [[matrix[j][i] for j in range(n)] for i in range(m)]

def flip_horizontal(matrix: list) -> list:
    """좌우 반전"""
    return [row[::-1] for row in matrix]

def flip_vertical(matrix: list) -> list:
    """상하 반전"""
    return matrix[::-1]
```

### 템플릿 3: 뱀 게임 (BOJ 3190)

```python
from collections import deque

def snake_game(n: int, apples: list, commands: list) -> int:
    """
    뱀 게임 시뮬레이션

    삼성 SW 역량 테스트 기출
    """
    # 사과 위치
    apple_set = set((r, c) for r, c in apples)

    # 뱀: deque로 관리 (앞 = 머리, 뒤 = 꼬리)
    snake = deque([(0, 0)])
    snake_set = {(0, 0)}

    # 방향: 0=우, 1=하, 2=좌, 3=상
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    direction = 0

    # 명령 처리
    command_idx = 0
    command_time = commands[0][0] if commands else float('inf')

    time = 0

    while True:
        time += 1

        # 머리 이동
        head_r, head_c = snake[0]
        new_r = head_r + dr[direction]
        new_c = head_c + dc[direction]

        # 충돌 체크
        if not (0 <= new_r < n and 0 <= new_c < n):
            break  # 벽
        if (new_r, new_c) in snake_set:
            break  # 자기 몸

        # 머리 추가
        snake.appendleft((new_r, new_c))
        snake_set.add((new_r, new_c))

        # 사과 체크
        if (new_r, new_c) in apple_set:
            apple_set.remove((new_r, new_c))
        else:
            # 꼬리 제거
            tail = snake.pop()
            snake_set.remove(tail)

        # 방향 전환 명령
        if time == command_time:
            turn = commands[command_idx][1]
            if turn == 'D':  # 오른쪽
                direction = (direction + 1) % 4
            else:  # 왼쪽
                direction = (direction + 3) % 4

            command_idx += 1
            if command_idx < len(commands):
                command_time = commands[command_idx][0]
            else:
                command_time = float('inf')

    return time
```

### 템플릿 4: 로봇 청소기 (BOJ 14503)

```python
def robot_cleaner(room: list, start_r: int, start_c: int, start_d: int) -> int:
    """
    로봇 청소기 시뮬레이션

    삼성 SW 역량 테스트 기출
    """
    n, m = len(room), len(room[0])

    # 방향: 0=북, 1=동, 2=남, 3=서
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    r, c, d = start_r, start_c, start_d
    cleaned = 0

    while True:
        # 1. 현재 위치 청소
        if room[r][c] == 0:
            room[r][c] = 2  # 청소됨
            cleaned += 1

        # 2. 주변 4칸 확인
        has_dirty = False
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < n and 0 <= nc < m and room[nr][nc] == 0:
                has_dirty = True
                break

        if has_dirty:
            # 3. 왼쪽으로 회전
            d = (d + 3) % 4

            # 4. 앞이 청소되지 않은 빈 칸이면 전진
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < n and 0 <= nc < m and room[nr][nc] == 0:
                r, c = nr, nc
        else:
            # 5. 후진
            back_d = (d + 2) % 4
            nr, nc = r + dr[back_d], c + dc[back_d]

            if 0 <= nr < n and 0 <= nc < m and room[nr][nc] != 1:
                r, c = nr, nc
            else:
                break  # 후진 불가, 종료

    return cleaned
```

### 템플릿 5: 미세먼지 안녕! (BOJ 17144)

```python
def fine_dust(room: list, t: int) -> int:
    """
    공기청정기 + 미세먼지 확산 시뮬레이션

    삼성 SW 역량 테스트 기출
    """
    n, m = len(room), len(room[0])

    # 공기청정기 위치 찾기
    cleaner = []
    for r in range(n):
        if room[r][0] == -1:
            cleaner.append(r)

    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    for _ in range(t):
        # 1. 미세먼지 확산
        spread = [[0] * m for _ in range(n)]

        for r in range(n):
            for c in range(m):
                if room[r][c] > 0:
                    amount = room[r][c] // 5
                    count = 0

                    for d in range(4):
                        nr, nc = r + dr[d], c + dc[d]
                        if 0 <= nr < n and 0 <= nc < m and room[nr][nc] != -1:
                            spread[nr][nc] += amount
                            count += 1

                    room[r][c] -= amount * count

        for r in range(n):
            for c in range(m):
                room[r][c] += spread[r][c]

        # 2. 공기청정기 작동 (위쪽 반시계, 아래쪽 시계)
        top, bottom = cleaner

        # 위쪽 반시계
        for r in range(top - 1, 0, -1):
            room[r][0] = room[r - 1][0]
        for c in range(m - 1):
            room[0][c] = room[0][c + 1]
        for r in range(top):
            room[r][m - 1] = room[r + 1][m - 1]
        for c in range(m - 1, 1, -1):
            room[top][c] = room[top][c - 1]
        room[top][1] = 0

        # 아래쪽 시계
        for r in range(bottom + 1, n - 1):
            room[r][0] = room[r + 1][0]
        for c in range(m - 1):
            room[n - 1][c] = room[n - 1][c + 1]
        for r in range(n - 1, bottom, -1):
            room[r][m - 1] = room[r - 1][m - 1]
        for c in range(m - 1, 1, -1):
            room[bottom][c] = room[bottom][c - 1]
        room[bottom][1] = 0

    # 미세먼지 총량
    return sum(room[r][c] for r in range(n) for c in range(m) if room[r][c] > 0)
```

### 템플릿 6: 상어 시리즈 패턴

```python
class Shark:
    """상어 객체 (삼성 SW 역량 단골)"""
    def __init__(self, r: int, c: int, d: int, speed: int, size: int):
        self.r = r
        self.c = c
        self.d = d  # 방향
        self.speed = speed
        self.size = size
        self.alive = True

    def move(self, rows: int, cols: int):
        """이동 (벽에서 반사)"""
        # 방향별 이동
        dr = [-1, 1, 0, 0]
        dc = [0, 0, 1, -1]

        for _ in range(self.speed):
            nr = self.r + dr[self.d]
            nc = self.c + dc[self.d]

            # 벽 반사
            if not (0 <= nr < rows):
                self.d = 1 - self.d  # 상↔하
                nr = self.r + dr[self.d]
            if not (0 <= nc < cols):
                self.d = 5 - self.d  # 좌↔우
                nc = self.c + dc[self.d]

            self.r, self.c = nr, nc
```

---

## 예제 문제

### 문제 1: 뱀 (BOJ 3190) - Gold 4

**문제 설명**
뱀이 사과를 먹으며 성장하고, 자기 몸이나 벽에 부딪히면 게임 종료.

**입력/출력 예시**
```
입력: n = 6, apples = [(3,4), (2,5), (5,3)]
      commands = [(3,'D'), (15,'L'), (17,'D')]
출력: 9
```

**풀이**: 템플릿 3 사용

---

### 문제 2: 로봇 청소기 (BOJ 14503) - Gold 5

**문제 설명**
로봇 청소기가 규칙에 따라 청소하는 칸 수.

**풀이**: 템플릿 4 사용

---

### 문제 3: 치킨 배달 (BOJ 15686) - Gold 5

**문제 설명**
M개의 치킨집을 선택하여 도시의 치킨 거리 최소화.

**풀이 (조합 + 시뮬레이션)**
```python
from itertools import combinations

def chicken_delivery(n: int, m: int, city: list) -> int:
    houses = []
    chickens = []

    for r in range(n):
        for c in range(n):
            if city[r][c] == 1:
                houses.append((r, c))
            elif city[r][c] == 2:
                chickens.append((r, c))

    min_dist = float('inf')

    for selected in combinations(chickens, m):
        total = 0

        for hr, hc in houses:
            min_chicken = min(abs(hr - cr) + abs(hc - cc)
                              for cr, cc in selected)
            total += min_chicken

        min_dist = min(min_dist, total)

    return min_dist
```

---

### 문제 4: 주사위 굴리기 (BOJ 14499) - Gold 4

**문제 설명**
주사위를 굴려 이동하며 값을 복사하거나 기록.

**풀이**
```python
def dice_roll(n: int, m: int, board: list, r: int, c: int, commands: list) -> list:
    # 주사위 상태: [위, 앞, 오른쪽, 왼쪽, 뒤, 아래]
    dice = [0, 0, 0, 0, 0, 0]

    dr = [0, 0, -1, 1]  # 동, 서, 북, 남
    dc = [1, -1, 0, 0]

    result = []

    for cmd in commands:
        d = cmd - 1
        nr, nc = r + dr[d], c + dc[d]

        if not (0 <= nr < n and 0 <= nc < m):
            continue

        r, c = nr, nc

        # 주사위 굴리기
        if cmd == 1:  # 동
            dice = [dice[3], dice[1], dice[0], dice[5], dice[4], dice[2]]
        elif cmd == 2:  # 서
            dice = [dice[2], dice[1], dice[5], dice[0], dice[4], dice[3]]
        elif cmd == 3:  # 북
            dice = [dice[1], dice[5], dice[2], dice[3], dice[0], dice[4]]
        elif cmd == 4:  # 남
            dice = [dice[4], dice[0], dice[2], dice[3], dice[5], dice[1]]

        # 바닥 처리
        if board[r][c] == 0:
            board[r][c] = dice[5]
        else:
            dice[5] = board[r][c]
            board[r][c] = 0

        result.append(dice[0])

    return result
```

---

## Editorial (풀이 전략)

### Step 1: 문제 분석

```
1. 그리드 크기 확인 (N ≤ 50 등)
2. 객체 식별 (뱀, 상어, 물고기 등)
3. 이동/회전 규칙 파악
4. 충돌/종료 조건 파악
5. 시간 순서 확인
```

### Step 2: 자료구조 선택

| 상황 | 자료구조 |
|------|---------|
| 객체 위치 관리 | 2D 배열 or dict |
| 순서 있는 이동 | deque |
| 여러 객체 | 객체 리스트/dict |
| 빠른 위치 검색 | set |

### Step 3: 디버깅 팁

```python
def print_grid(grid):
    """그리드 상태 출력"""
    for row in grid:
        print(' '.join(map(str, row)))
    print()

# 각 단계마다 상태 출력
print(f"시간: {time}, 위치: ({r}, {c}), 방향: {d}")
print_grid(grid)
```

---

## 자주 하는 실수

### 1. 인덱스 범위 오류
```python
# ❌ 0-indexed vs 1-indexed 혼동
room[r][c]  # 문제는 1-indexed인데 0-indexed로 접근

# ✅ 입력 시 변환
r, c = map(int, input().split())
r -= 1
c -= 1
```

### 2. 방향 순서 혼동
```python
# ❌ 문제의 방향 정의와 다름
dr = [0, 1, 0, -1]  # 동남서북
dc = [1, 0, -1, 0]

# ✅ 문제에서 정의한 순서 확인
# 문제: 0=북, 1=동, 2=남, 3=서
dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]
```

### 3. 동시 처리 vs 순차 처리
```python
# ❌ 순차 처리 (먼저 이동한 객체가 영향)
for shark in sharks:
    shark.move()
    grid[shark.r][shark.c] = shark

# ✅ 동시 처리 (별도 배열 사용)
for shark in sharks:
    shark.move()
# 이동 후 일괄 배치
for shark in sharks:
    grid[shark.r][shark.c] = shark
```

---

## BOJ 추천 문제 (삼성 SW 역량 기출)

| # | 문제명 | 난이도 | 핵심 |
|---|-------|-------|------|
| 3190 | 뱀 | Gold 4 | deque, 방향 |
| 14503 | 로봇 청소기 | Gold 5 | 방향, 조건 |
| 14499 | 주사위 굴리기 | Gold 4 | 상태 관리 |
| 15686 | 치킨 배달 | Gold 5 | 조합 |
| 17144 | 미세먼지 안녕! | Gold 4 | 확산, 순환 |
| 15683 | 감시 | Gold 4 | 백트래킹, 방향 |
| 14891 | 톱니바퀴 | Gold 5 | 회전, 연쇄 |
| 14500 | 테트로미노 | Gold 4 | 브루트포스 |
| 16236 | 아기 상어 | Gold 3 | BFS |
| 17143 | 낚시왕 | Gold 2 | 상어 이동 |
| 20055 | 컨베이어 벨트 | Gold 5 | 순환, 상태 |
| 21608 | 상어 초등학교 | Gold 5 | 조건 |

---

## 임베딩용 키워드

```
simulation, 시뮬레이션, 구현, implementation, 삼성 SW 역량,
2D grid, 2차원 배열, direction, 방향, rotation, 회전,
snake game, 뱀, robot cleaner, 로봇 청소기, fine dust, 미세먼지,
dice, 주사위, shark, 상어, BFS, 조건 처리
```
