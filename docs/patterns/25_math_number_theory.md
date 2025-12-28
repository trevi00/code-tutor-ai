# Pattern 25: Math & Number Theory (수학 및 정수론)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(1) ~ O(√n) |
| **공간복잡도** | O(1) ~ O(n) |
| **선행 지식** | 기초 수학, 나머지 연산 |

## 정의

**수학/정수론**은 숫자의 성질과 관계를 활용하여 문제를 해결하는 알고리즘입니다. 소수, 약수, 최대공약수, 조합론 등이 포함됩니다.

## 핵심 개념

| 주제 | 핵심 공식/개념 |
|------|---------------|
| **소수** | n이 소수 ⟺ 2~√n 중 약수 없음 |
| **GCD/LCM** | lcm(a,b) = a*b / gcd(a,b) |
| **모듈러 연산** | (a+b)%m = ((a%m)+(b%m))%m |
| **거듭제곱** | a^n mod m (분할 정복) |
| **조합** | nCr = n! / (r! × (n-r)!) |
| **페르마 소정리** | a^(p-1) ≡ 1 (mod p) |

---

## 템플릿 코드

### 템플릿 1: 소수 판별

```python
def is_prime(n: int) -> bool:
    """
    소수 판별

    Time: O(√n)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False

    return True


def is_prime_optimized(n: int) -> bool:
    """6k ± 1 최적화"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True
```

### 템플릿 2: 에라토스테네스의 체

```python
def sieve_of_eratosthenes(n: int) -> list:
    """
    n 이하의 모든 소수

    Time: O(n log log n)
    Space: O(n)
    """
    if n < 2:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]


def sieve_smallest_factor(n: int) -> list:
    """
    각 수의 최소 소인수

    소인수분해에 활용
    """
    spf = list(range(n + 1))  # smallest prime factor

    for i in range(2, int(n ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i

    return spf


def factorize_with_spf(n: int, spf: list) -> list:
    """SPF를 이용한 빠른 소인수분해"""
    factors = []
    while n > 1:
        factors.append(spf[n])
        n //= spf[n]
    return factors
```

### 템플릿 3: GCD / LCM

```python
import math

def gcd(a: int, b: int) -> int:
    """유클리드 호제법"""
    while b:
        a, b = b, a % b
    return a

# Python 3.9+ 에서는 math.gcd 사용
# Python 3.9+ 에서는 math.lcm 사용

def lcm(a: int, b: int) -> int:
    """최소공배수"""
    return a * b // gcd(a, b)


def gcd_list(nums: list) -> int:
    """여러 수의 GCD"""
    result = nums[0]
    for num in nums[1:]:
        result = gcd(result, num)
        if result == 1:
            break
    return result


def lcm_list(nums: list) -> int:
    """여러 수의 LCM"""
    result = nums[0]
    for num in nums[1:]:
        result = lcm(result, num)
    return result


def extended_gcd(a: int, b: int) -> tuple:
    """
    확장 유클리드 호제법
    ax + by = gcd(a, b)를 만족하는 x, y 반환
    """
    if b == 0:
        return a, 1, 0

    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y
```

### 템플릿 4: 모듈러 연산

```python
MOD = 10 ** 9 + 7


def mod_pow(base: int, exp: int, mod: int = MOD) -> int:
    """
    빠른 거듭제곱 (분할 정복)

    Time: O(log exp)
    """
    result = 1
    base %= mod

    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod

    return result


def mod_inverse(a: int, mod: int = MOD) -> int:
    """
    모듈러 역원 (페르마 소정리)

    mod가 소수일 때만 사용
    a^(-1) ≡ a^(p-2) (mod p)
    """
    return mod_pow(a, mod - 2, mod)


def mod_inverse_extended(a: int, mod: int) -> int:
    """확장 유클리드를 이용한 모듈러 역원"""
    g, x, _ = extended_gcd(a, mod)
    if g != 1:
        return -1  # 역원 없음
    return (x % mod + mod) % mod


def mod_divide(a: int, b: int, mod: int = MOD) -> int:
    """모듈러 나눗셈: a / b mod p"""
    return (a * mod_inverse(b, mod)) % mod
```

### 템플릿 5: 조합론 (nCr)

```python
MOD = 10 ** 9 + 7


class Combination:
    """
    조합 계산 클래스

    전처리: O(n)
    쿼리: O(1)
    """
    def __init__(self, n: int, mod: int = MOD):
        self.mod = mod
        self.fact = [1] * (n + 1)
        self.inv_fact = [1] * (n + 1)

        # 팩토리얼 계산
        for i in range(1, n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        # 역 팩토리얼 계산
        self.inv_fact[n] = mod_pow(self.fact[n], mod - 2, mod)
        for i in range(n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def nCr(self, n: int, r: int) -> int:
        """n개 중 r개 선택"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def nPr(self, n: int, r: int) -> int:
        """n개 중 r개 순열"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod

    def nHr(self, n: int, r: int) -> int:
        """중복 조합: n+r-1Cr"""
        return self.nCr(n + r - 1, r)


def nCr_small(n: int, r: int) -> int:
    """작은 수의 조합 (파스칼 삼각형)"""
    if r > n - r:
        r = n - r

    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)

    return result
```

### 템플릿 6: 소인수분해

```python
def prime_factorization(n: int) -> dict:
    """
    소인수분해

    Time: O(√n)
    Returns: {소인수: 지수}
    """
    factors = {}
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1

    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    return factors


def divisors(n: int) -> list:
    """
    모든 약수

    Time: O(√n)
    """
    result = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)

    return sorted(result)


def count_divisors(n: int) -> int:
    """약수의 개수"""
    factors = prime_factorization(n)
    count = 1
    for exp in factors.values():
        count *= (exp + 1)
    return count


def sum_divisors(n: int) -> int:
    """약수의 합"""
    factors = prime_factorization(n)
    total = 1
    for p, e in factors.items():
        total *= (pow(p, e + 1) - 1) // (p - 1)
    return total
```

### 템플릿 7: 피보나치 (행렬 거듭제곱)

```python
def fib_matrix(n: int, mod: int = 10 ** 9 + 7) -> int:
    """
    피보나치 O(log n)

    행렬 거듭제곱 활용
    """
    if n <= 1:
        return n

    def matrix_mult(A, B, mod):
        return [
            [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod,
             (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod],
            [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod,
             (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod]
        ]

    def matrix_pow(M, n, mod):
        result = [[1, 0], [0, 1]]  # 단위 행렬

        while n > 0:
            if n & 1:
                result = matrix_mult(result, M, mod)
            n >>= 1
            M = matrix_mult(M, M, mod)

        return result

    F = [[1, 1], [1, 0]]
    result = matrix_pow(F, n - 1, mod)

    return result[0][0]
```

### 템플릿 8: 오일러 피 함수

```python
def euler_phi(n: int) -> int:
    """
    오일러 피 함수
    1~n 중 n과 서로소인 수의 개수

    Time: O(√n)
    """
    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result


def euler_phi_sieve(n: int) -> list:
    """오일러 피 함수 체"""
    phi = list(range(n + 1))

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    return phi
```

### 템플릿 9: 카탈란 수

```python
def catalan(n: int, mod: int = 10 ** 9 + 7) -> int:
    """
    n번째 카탈란 수

    C_n = (2n)! / ((n+1)! * n!)

    용도:
    - 올바른 괄호 조합
    - 이진 트리 개수
    - 볼록 다각형 삼각분할
    """
    comb = Combination(2 * n, mod)
    return comb.nCr(2 * n, n) * mod_inverse(n + 1, mod) % mod


def catalan_dp(n: int) -> list:
    """DP로 카탈란 수열 계산"""
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1

    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]

    return dp
```

### 템플릿 10: 중국인의 나머지 정리

```python
def chinese_remainder_theorem(remainders: list, moduli: list) -> int:
    """
    중국인의 나머지 정리 (CRT)

    x ≡ r_i (mod m_i) 를 만족하는 x
    """
    from functools import reduce

    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y

    M = reduce(lambda a, b: a * b, moduli)
    x = 0

    for r, m in zip(remainders, moduli):
        Mi = M // m
        _, inv, _ = extended_gcd(Mi, m)
        x += r * Mi * inv

    return x % M
```

---

## 예제 문제

### 문제 1: 소수 찾기 (프로그래머스 Level 1)

**문제 설명**
1부터 n 사이의 소수 개수.

**풀이**
```python
def solution(n: int) -> int:
    primes = sieve_of_eratosthenes(n)
    return len(primes)
```

---

### 문제 2: 큰 수 만들기 - GCD (BOJ 2609)

**문제 설명**
두 자연수의 최대공약수와 최소공배수.

**풀이**
```python
def solution(a: int, b: int) -> tuple:
    g = gcd(a, b)
    l = a * b // g
    return g, l
```

---

### 문제 3: 이항 계수 (BOJ 11050, 11051) - Silver

**문제 설명**
nCk mod p 계산.

**풀이**
```python
def solution(n: int, k: int) -> int:
    MOD = 10007
    comb = Combination(n, MOD)
    return comb.nCr(n, k)
```

---

### 문제 4: 팩토리얼 0의 개수 (BOJ 1676) - Silver 5

**문제 설명**
N! 끝자리 0의 개수.

**풀이**
```python
def trailing_zeros(n: int) -> int:
    """5의 개수 = 0의 개수"""
    count = 0
    power = 5

    while power <= n:
        count += n // power
        power *= 5

    return count
```

---

### 문제 5: 분수 합 (BOJ 1735) - Silver 3

**문제 설명**
두 분수의 합을 기약분수로.

**풀이**
```python
def solution(a1: int, b1: int, a2: int, b2: int) -> tuple:
    # a1/b1 + a2/b2
    numerator = a1 * b2 + a2 * b1
    denominator = b1 * b2

    g = gcd(numerator, denominator)
    return numerator // g, denominator // g
```

---

### 문제 6: 거듭제곱 (BOJ 1629) - Silver 1

**문제 설명**
A^B mod C.

**풀이**
```python
def solution(a: int, b: int, c: int) -> int:
    return mod_pow(a, b, c)
```

---

### 문제 7: 피보나치 (BOJ 11444) - Gold 3

**문제 설명**
F(n) mod (10^9 + 7).

**풀이**
```python
def solution(n: int) -> int:
    return fib_matrix(n)
```

---

### 문제 8: 조합 0의 개수 (BOJ 2004) - Silver 2

**문제 설명**
nCm의 끝자리 0의 개수.

**풀이**
```python
def count_factor(n: int, p: int) -> int:
    """n!에서 p의 개수"""
    count = 0
    power = p
    while power <= n:
        count += n // power
        power *= p
    return count


def trailing_zeros_comb(n: int, m: int) -> int:
    # nCm = n! / (m! * (n-m)!)
    twos = count_factor(n, 2) - count_factor(m, 2) - count_factor(n - m, 2)
    fives = count_factor(n, 5) - count_factor(m, 5) - count_factor(n - m, 5)

    return min(twos, fives)
```

---

## Editorial (풀이 전략)

### Step 1: 문제 유형 파악

| 키워드 | 알고리즘 |
|--------|---------|
| 소수, 약수 | 에라토스테네스, √n 탐색 |
| 최대공약수, 최소공배수 | 유클리드 호제법 |
| 나머지, mod | 모듈러 연산 |
| 조합, 순열 | nCr, 팩토리얼 |
| 거듭제곱 | 분할 정복 |
| 피보나치 (n이 매우 큼) | 행렬 거듭제곱 |

### Step 2: 수의 범위 확인

```
n ≤ 10^6: 에라토스테네스 체
n ≤ 10^12: √n 순회
n ≤ 10^18: log n 알고리즘 (거듭제곱 등)
```

### Step 3: 모듈러 연산 주의

```python
# 덧셈/뺄셈
(a + b) % mod
(a - b + mod) % mod  # 음수 방지

# 곱셈
(a * b) % mod

# 나눗셈 (mod가 소수일 때)
a * mod_inverse(b) % mod
```

---

## 자주 하는 실수

### 1. 정수 오버플로우
```python
# ❌ 곱셈 후 모듈러
result = (a * b) % mod  # a * b가 이미 오버플로우

# ✅ Python은 큰 정수 지원하지만, 명시적 처리 권장
result = ((a % mod) * (b % mod)) % mod
```

### 2. 나눗셈에서 모듈러
```python
# ❌ 직접 나눗셈
result = (a / b) % mod

# ✅ 역원 사용
result = (a * mod_inverse(b, mod)) % mod
```

### 3. 0! = 1 처리
```python
# ❌ 0 팩토리얼 누락
fact = [1] * (n + 1)
for i in range(1, n + 1):  # i=0 누락해도 OK (초기값 1)

# 하지만 명시적으로:
fact[0] = 1  # 0! = 1
```

### 4. 소수 판별 경계
```python
# ❌ i < √n
for i in range(2, int(n ** 0.5)):  # √n 자체 누락

# ✅ i ≤ √n
for i in range(2, int(n ** 0.5) + 1):
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 | 유형 |
|--------|---|-------|-------|------|
| LeetCode | 204 | Count Primes | Medium | 에라토스테네스 |
| LeetCode | 50 | Pow(x, n) | Medium | 거듭제곱 |
| LeetCode | 172 | Factorial Trailing Zeroes | Medium | 수학 |
| LeetCode | 509 | Fibonacci Number | Easy | 피보나치 |
| LeetCode | 62 | Unique Paths | Medium | 조합 |
| BOJ | 1929 | 소수 구하기 | Silver 3 | 에라토스테네스 |
| BOJ | 2609 | 최대공약수와 최소공배수 | Bronze 1 | GCD/LCM |
| BOJ | 1676 | 팩토리얼 0의 개수 | Silver 5 | 수학 |
| BOJ | 1629 | 곱셈 | Silver 1 | 거듭제곱 |
| BOJ | 11444 | 피보나치 수 6 | Gold 3 | 행렬 거듭제곱 |
| BOJ | 11050 | 이항 계수 1 | Bronze 1 | 조합 |
| BOJ | 11051 | 이항 계수 2 | Silver 1 | 조합 + 모듈러 |
| BOJ | 11401 | 이항 계수 3 | Gold 1 | 페르마 소정리 |
| BOJ | 1735 | 분수 합 | Silver 3 | GCD |
| 프로그래머스 | - | 소수 찾기 | Level 1 | 에라토스테네스 |
| 프로그래머스 | - | 멀쩡한 사각형 | Level 2 | GCD |

---

## 공식 정리

### 기본 공식

| 공식 | 설명 |
|------|------|
| `gcd(a, b) = gcd(b, a % b)` | 유클리드 호제법 |
| `lcm(a, b) = a * b / gcd(a, b)` | LCM |
| `a^(p-1) ≡ 1 (mod p)` | 페르마 소정리 (p는 소수) |
| `a^(-1) ≡ a^(p-2) (mod p)` | 모듈러 역원 |
| `nCr = n! / (r! × (n-r)!)` | 조합 |
| `C_n = C_{2n}^n / (n+1)` | 카탈란 수 |

### 수열 공식

| 수열 | 점화식 |
|------|--------|
| 피보나치 | F(n) = F(n-1) + F(n-2) |
| 카탈란 | C(n) = Σ C(i) × C(n-1-i) |
| 삼각수 | T(n) = n(n+1)/2 |
| 제곱수 합 | Σi² = n(n+1)(2n+1)/6 |

---

## 임베딩용 키워드

```
math, number theory, 수학, 정수론, prime, 소수, GCD, LCM,
최대공약수, 최소공배수, euclidean, 유클리드, modular, 모듈러,
combination, 조합, permutation, 순열, factorial, 팩토리얼,
sieve, 체, eratosthenes, 에라토스테네스, fermat, 페르마,
fibonacci, 피보나치, catalan, 카탈란, euler phi, 오일러 피
```
