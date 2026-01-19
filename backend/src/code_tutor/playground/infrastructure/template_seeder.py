"""Seed code templates for playground."""

import json
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.playground.domain import (
    PlaygroundLanguage,
    TemplateCategory,
)
from code_tutor.playground.infrastructure.models import CodeTemplateModel

TEMPLATES = [
    # Python Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic Python hello world program",
        "code": '''# Hello World in Python
print("Hello, World!")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Two Sum",
        "description": "Find two numbers that add up to target",
        "code": '''def two_sum(nums: list[int], target: int) -> list[int]:
    """Find indices of two numbers that add up to target."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# Test
nums = [2, 7, 11, 15]
target = 9
print(f"Input: nums={nums}, target={target}")
print(f"Output: {two_sum(nums, target)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["array", "hash-table", "two-sum"],
    },
    {
        "id": str(uuid4()),
        "title": "Binary Search",
        "description": "Binary search algorithm implementation",
        "code": '''def binary_search(arr: list[int], target: int) -> int:
    """Find target index using binary search. Returns -1 if not found."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# Test
arr = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
print(f"Array: {arr}")
print(f"Searching for: {target}")
print(f"Found at index: {binary_search(arr, target)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["binary-search", "array", "divide-and-conquer"],
    },
    {
        "id": str(uuid4()),
        "title": "QuickSort",
        "description": "QuickSort algorithm implementation",
        "code": '''def quicksort(arr: list[int]) -> list[int]:
    """Sort array using QuickSort algorithm."""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


# Test
arr = [64, 34, 25, 12, 22, 11, 90]
print(f"Original: {arr}")
print(f"Sorted: {quicksort(arr)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["sorting", "quicksort", "divide-and-conquer"],
    },
    {
        "id": str(uuid4()),
        "title": "Linked List",
        "description": "Basic linked list implementation",
        "code": '''class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements


# Test
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(f"Linked List: {ll.display()}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.DATA_STRUCTURE.value,
        "tags": ["linked-list", "data-structure"],
    },

    # JavaScript Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic JavaScript hello world program",
        "code": '''// Hello World in JavaScript
console.log("Hello, World!");
''',
        "language": PlaygroundLanguage.JAVASCRIPT.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Array Methods",
        "description": "Common JavaScript array methods",
        "code": '''// JavaScript Array Methods Demo

const numbers = [1, 2, 3, 4, 5];

// map - transform each element
const doubled = numbers.map(n => n * 2);
console.log("Doubled:", doubled);

// filter - keep elements matching condition
const evens = numbers.filter(n => n % 2 === 0);
console.log("Evens:", evens);

// reduce - accumulate to single value
const sum = numbers.reduce((acc, n) => acc + n, 0);
console.log("Sum:", sum);

// find - get first matching element
const found = numbers.find(n => n > 3);
console.log("Found > 3:", found);

// some/every - check conditions
console.log("Some > 3:", numbers.some(n => n > 3));
console.log("Every > 0:", numbers.every(n => n > 0));
''',
        "language": PlaygroundLanguage.JAVASCRIPT.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["array", "functional", "beginner"],
    },
    {
        "id": str(uuid4()),
        "title": "Promises and Async/Await",
        "description": "JavaScript async programming patterns",
        "code": '''// Promises and Async/Await Demo

// Simulated async operation
function fetchData(id) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (id > 0) {
                resolve({ id, name: `User ${id}` });
            } else {
                reject(new Error("Invalid ID"));
            }
        }, 100);
    });
}

// Using Promise.then()
fetchData(1)
    .then(user => console.log("Promise:", user))
    .catch(err => console.error(err));

// Using async/await
async function getUser() {
    try {
        const user = await fetchData(2);
        console.log("Async/Await:", user);
    } catch (err) {
        console.error(err);
    }
}

getUser();

// Multiple concurrent requests
Promise.all([fetchData(1), fetchData(2), fetchData(3)])
    .then(users => console.log("All users:", users));
''',
        "language": PlaygroundLanguage.JAVASCRIPT.value,
        "category": TemplateCategory.UTILITY.value,
        "tags": ["async", "promises", "intermediate"],
    },

    # Java Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic Java hello world program",
        "code": '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
''',
        "language": PlaygroundLanguage.JAVA.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "ArrayList Example",
        "description": "Working with Java ArrayList",
        "code": '''import java.util.ArrayList;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {
        ArrayList<Integer> numbers = new ArrayList<>();

        // Add elements
        numbers.add(5);
        numbers.add(2);
        numbers.add(8);
        numbers.add(1);

        System.out.println("Original: " + numbers);

        // Sort
        Collections.sort(numbers);
        System.out.println("Sorted: " + numbers);

        // Size and access
        System.out.println("Size: " + numbers.size());
        System.out.println("First element: " + numbers.get(0));

        // Remove
        numbers.remove(0);
        System.out.println("After remove: " + numbers);
    }
}
''',
        "language": PlaygroundLanguage.JAVA.value,
        "category": TemplateCategory.DATA_STRUCTURE.value,
        "tags": ["arraylist", "collections", "beginner"],
    },

    # Go Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic Go hello world program",
        "code": '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
''',
        "language": PlaygroundLanguage.GO.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Goroutines and Channels",
        "description": "Go concurrency with goroutines and channels",
        "code": '''package main

import (
    "fmt"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\\n", id, j)
        time.Sleep(100 * time.Millisecond)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 5)
    results := make(chan int, 5)

    // Start 3 workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }

    // Send 5 jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)

    // Collect results
    for r := 1; r <= 5; r++ {
        result := <-results
        fmt.Printf("Result: %d\\n", result)
    }
}
''',
        "language": PlaygroundLanguage.GO.value,
        "category": TemplateCategory.UTILITY.value,
        "tags": ["concurrency", "goroutines", "channels"],
    },

    # Rust Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic Rust hello world program",
        "code": '''fn main() {
    println!("Hello, World!");
}
''',
        "language": PlaygroundLanguage.RUST.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Ownership and Borrowing",
        "description": "Rust ownership and borrowing concepts",
        "code": '''fn main() {
    // Ownership
    let s1 = String::from("hello");
    let s2 = s1.clone(); // Clone to keep s1 valid
    println!("s1 = {}, s2 = {}", s1, s2);

    // Borrowing (immutable reference)
    let len = calculate_length(&s1);
    println!("Length of '{}' is {}", s1, len);

    // Mutable borrowing
    let mut s3 = String::from("hello");
    change(&mut s3);
    println!("Changed: {}", s3);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(s: &mut String) {
    s.push_str(", world!");
}
''',
        "language": PlaygroundLanguage.RUST.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["ownership", "borrowing", "memory"],
    },

    # C++ Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic C++ hello world program",
        "code": '''#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
''',
        "language": PlaygroundLanguage.CPP.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "STL Vector",
        "description": "C++ STL vector operations",
        "code": '''#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {5, 2, 8, 1, 9};

    std::cout << "Original: ";
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;

    // Sort
    std::sort(vec.begin(), vec.end());
    std::cout << "Sorted: ";
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;

    // Add element
    vec.push_back(10);
    std::cout << "After push_back(10): ";
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;

    // Find
    auto it = std::find(vec.begin(), vec.end(), 8);
    if (it != vec.end()) {
        std::cout << "Found 8 at index: " << (it - vec.begin()) << std::endl;
    }

    return 0;
}
''',
        "language": PlaygroundLanguage.CPP.value,
        "category": TemplateCategory.DATA_STRUCTURE.value,
        "tags": ["stl", "vector", "algorithms"],
    },

    # C Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic C hello world program",
        "code": '''#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
''',
        "language": PlaygroundLanguage.C.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Pointers",
        "description": "C pointers and memory",
        "code": '''#include <stdio.h>
#include <stdlib.h>

int main() {
    // Basic pointer
    int x = 10;
    int *ptr = &x;
    printf("x = %d, *ptr = %d\\n", x, *ptr);

    // Pointer arithmetic
    int arr[] = {1, 2, 3, 4, 5};
    int *p = arr;
    printf("Array elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", *(p + i));
    }
    printf("\\n");

    // Dynamic memory
    int *dynamic = (int *)malloc(5 * sizeof(int));
    if (dynamic != NULL) {
        for (int i = 0; i < 5; i++) {
            dynamic[i] = i * 10;
        }
        printf("Dynamic array: ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", dynamic[i]);
        }
        printf("\\n");
        free(dynamic);
    }

    return 0;
}
''',
        "language": PlaygroundLanguage.C.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["pointers", "memory", "malloc"],
    },

    # TypeScript Templates
    {
        "id": str(uuid4()),
        "title": "Hello World",
        "description": "Basic TypeScript hello world program",
        "code": '''// Hello World in TypeScript
const message: string = "Hello, World!";
console.log(message);
''',
        "language": PlaygroundLanguage.TYPESCRIPT.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["beginner", "hello-world"],
    },
    {
        "id": str(uuid4()),
        "title": "Interfaces and Types",
        "description": "TypeScript type system examples",
        "code": '''// Interface definition
interface User {
    id: number;
    name: string;
    email: string;
    age?: number; // Optional property
}

// Type alias
type UserRole = "admin" | "user" | "guest";

// Generic function
function getFirst<T>(arr: T[]): T | undefined {
    return arr.length > 0 ? arr[0] : undefined;
}

// Example usage
const user: User = {
    id: 1,
    name: "John Doe",
    email: "john@example.com"
};

const role: UserRole = "admin";

console.log("User:", user);
console.log("Role:", role);
console.log("First number:", getFirst([1, 2, 3]));
console.log("First string:", getFirst(["a", "b", "c"]));
''',
        "language": PlaygroundLanguage.TYPESCRIPT.value,
        "category": TemplateCategory.BASIC.value,
        "tags": ["types", "interfaces", "generics"],
    },
    # ============== Phase 1: 기초 자료구조 & 테크닉 템플릿 ==============
    {
        "id": str(uuid4()),
        "title": "Prefix Sum (누적합)",
        "description": "구간 합을 O(1)에 계산하는 Prefix Sum 템플릿",
        "code": '''def build_prefix_sum(nums: list[int]) -> list[int]:
    """Prefix Sum 배열을 생성합니다.

    prefix[i]는 nums[0]부터 nums[i-1]까지의 합입니다.
    구간 [l, r]의 합 = prefix[r+1] - prefix[l]
    """
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    return prefix


def range_sum(prefix: list[int], left: int, right: int) -> int:
    """구간 [left, right]의 합을 O(1)에 계산합니다."""
    return prefix[right + 1] - prefix[left]


# 사용 예시
nums = [1, 2, 3, 4, 5]
prefix = build_prefix_sum(nums)

print(f"배열: {nums}")
print(f"Prefix Sum: {prefix}")
print(f"구간 [1, 3]의 합: {range_sum(prefix, 1, 3)}")  # 2+3+4 = 9
print(f"구간 [0, 4]의 합: {range_sum(prefix, 0, 4)}")  # 전체 합 = 15
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["prefix-sum", "range-query", "array"],
    },
    {
        "id": str(uuid4()),
        "title": "2D Prefix Sum (2차원 누적합)",
        "description": "2차원 배열에서 영역 합을 O(1)에 계산하는 템플릿",
        "code": '''def build_2d_prefix_sum(matrix: list[list[int]]) -> list[list[int]]:
    """2D Prefix Sum 배열을 생성합니다.

    prefix[i][j]는 (0,0)부터 (i-1,j-1)까지의 합입니다.
    """
    if not matrix or not matrix[0]:
        return [[]]

    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            prefix[i+1][j+1] = (prefix[i][j+1] + prefix[i+1][j]
                               - prefix[i][j] + matrix[i][j])
    return prefix


def range_sum_2d(prefix: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> int:
    """영역 (r1,c1) ~ (r2,c2)의 합을 O(1)에 계산합니다.

    포함-배제 원리 적용:
    전체 - 위쪽 - 왼쪽 + 중복 빼진 부분
    """
    return (prefix[r2+1][c2+1] - prefix[r1][c2+1]
            - prefix[r2+1][c1] + prefix[r1][c1])


# 사용 예시
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
prefix = build_2d_prefix_sum(matrix)

print("Matrix:")
for row in matrix:
    print(row)

print(f"\\n영역 (0,0)~(1,1)의 합: {range_sum_2d(prefix, 0, 0, 1, 1)}")  # 1+2+4+5 = 12
print(f"영역 (1,1)~(2,2)의 합: {range_sum_2d(prefix, 1, 1, 2, 2)}")  # 5+6+8+9 = 28
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["prefix-sum-2d", "range-query", "matrix"],
    },
    {
        "id": str(uuid4()),
        "title": "Hash Frequency (빈도수 계산)",
        "description": "해시맵을 활용한 빈도수 계산 및 그룹핑 템플릿",
        "code": '''from collections import Counter, defaultdict


def count_frequency(items: list) -> dict:
    """각 요소의 빈도수를 계산합니다."""
    return dict(Counter(items))


def group_by_key(items: list, key_func) -> dict:
    """key_func 결과를 기준으로 그룹핑합니다."""
    groups = defaultdict(list)
    for item in items:
        groups[key_func(item)].append(item)
    return dict(groups)


def find_most_common(items: list, n: int = 1) -> list:
    """가장 빈번한 n개 요소를 반환합니다."""
    return Counter(items).most_common(n)


# 사용 예시
# 1. 빈도수 계산
text = "banana"
freq = count_frequency(text)
print(f"빈도수: {freq}")  # {'b': 1, 'a': 3, 'n': 2}

# 2. 애너그램 그룹핑
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
anagram_groups = group_by_key(words, lambda w: ''.join(sorted(w)))
print(f"\\n애너그램 그룹: {anagram_groups}")

# 3. 가장 빈번한 요소
nums = [1, 1, 1, 2, 2, 3]
most_common = find_most_common(nums, 2)
print(f"\\n가장 빈번한 2개: {most_common}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["hash-table", "counter", "frequency", "grouping"],
    },
    {
        "id": str(uuid4()),
        "title": "Prime Check (소수 판별)",
        "description": "효율적인 소수 판별 템플릿",
        "code": '''def is_prime(n: int) -> bool:
    """n이 소수인지 판별합니다. O(sqrt(n))

    최적화:
    - 2 미만은 소수가 아님
    - 2는 유일한 짝수 소수
    - 3 이상 홀수만 검사
    - sqrt(n)까지만 검사
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
    """6k±1 최적화를 적용한 소수 판별.

    2, 3을 제외한 모든 소수는 6k±1 형태입니다.
    """
    if n < 2:
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


# 사용 예시
test_numbers = [1, 2, 3, 4, 17, 18, 97, 100]
print("소수 판별 결과:")
for n in test_numbers:
    print(f"  {n}: {'소수' if is_prime(n) else '합성수'}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["math", "prime", "number-theory"],
    },
    {
        "id": str(uuid4()),
        "title": "Sieve of Eratosthenes (에라토스테네스의 체)",
        "description": "범위 내 모든 소수를 효율적으로 찾는 체 알고리즘",
        "code": '''def sieve_of_eratosthenes(limit: int) -> list[int]:
    """1부터 limit까지의 모든 소수를 반환합니다.

    시간복잡도: O(n log log n)
    공간복잡도: O(n)
    """
    if limit < 2:
        return []

    # True = 소수 후보
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    # 2부터 sqrt(limit)까지 검사
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            # i의 배수들을 모두 제거 (i*i부터 시작)
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(limit + 1) if is_prime[i]]


def nth_prime(n: int) -> int:
    """n번째 소수를 반환합니다.

    n번째 소수의 상한 추정: n * (ln(n) + ln(ln(n)))
    """
    import math
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 100

    primes = sieve_of_eratosthenes(limit)
    return primes[n - 1] if n <= len(primes) else -1


# 사용 예시
limit = 50
primes = sieve_of_eratosthenes(limit)
print(f"1~{limit} 사이의 소수: {primes}")
print(f"개수: {len(primes)}개")

print(f"\\n10번째 소수: {nth_prime(10)}")
print(f"100번째 소수: {nth_prime(100)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["math", "prime", "sieve", "number-theory"],
    },
    {
        "id": str(uuid4()),
        "title": "KMP String Matching (KMP 문자열 매칭)",
        "description": "O(n+m) 시간에 패턴 매칭을 수행하는 KMP 알고리즘",
        "code": '''def compute_lps(pattern: str) -> list[int]:
    """LPS (Longest Proper Prefix which is also Suffix) 배열 계산.

    lps[i] = pattern[0:i+1]에서 접두사=접미사인 최대 길이
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # 이전 longest prefix suffix의 길이
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # 이전 lps 값으로 점프
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text: str, pattern: str) -> list[int]:
    """KMP 알고리즘으로 패턴이 등장하는 모든 시작 인덱스를 반환.

    시간복잡도: O(n + m)
    공간복잡도: O(m)
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    lps = compute_lps(pattern)
    result = []
    i = j = 0  # text, pattern의 인덱스

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            # 패턴 발견!
            result.append(i - j)
            j = lps[j - 1]  # 다음 매칭을 위해 점프
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]  # 패턴 내에서 점프
            else:
                i += 1

    return result


# 사용 예시
text = "ABABDABACDABABCABAB"
pattern = "ABABC"

print(f"텍스트: {text}")
print(f"패턴: {pattern}")
print(f"LPS 배열: {compute_lps(pattern)}")
print(f"패턴 위치: {kmp_search(text, pattern)}")

# 중첩 패턴 예시
text2 = "AAAAAA"
pattern2 = "AA"
print(f"\\n텍스트: {text2}")
print(f"패턴: {pattern2}")
print(f"패턴 위치: {kmp_search(text2, pattern2)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["string", "pattern-matching", "kmp"],
    },
    # ============== Phase 2: 정렬 알고리즘 템플릿 ==============
    {
        "id": str(uuid4()),
        "title": "Counting Sort (계수 정렬)",
        "description": "O(n+k) 시간에 정렬하는 비교 기반이 아닌 정렬",
        "code": '''def counting_sort(nums: list[int], min_val: int = 0, max_val: int = 100) -> list[int]:
    """계수 정렬: 값의 범위가 제한될 때 효율적

    시간복잡도: O(n + k), k는 값의 범위
    공간복잡도: O(k)
    """
    if not nums:
        return []

    # 오프셋 적용 (음수 처리)
    offset = -min_val if min_val < 0 else 0
    range_size = max_val - min_val + 1

    count = [0] * range_size

    # 카운트
    for num in nums:
        count[num + offset] += 1

    # 결과 생성
    result = []
    for i in range(range_size):
        result.extend([i - offset] * count[i])

    return result


def counting_sort_stable(nums: list[int], max_val: int) -> list[int]:
    """안정 정렬 버전의 계수 정렬

    동일 값의 상대적 순서가 유지됩니다.
    기수 정렬의 기반으로 사용됩니다.
    """
    if not nums:
        return []

    count = [0] * (max_val + 1)
    output = [0] * len(nums)

    # 카운트
    for num in nums:
        count[num] += 1

    # 누적합 (각 값이 끝나는 위치)
    for i in range(1, max_val + 1):
        count[i] += count[i - 1]

    # 뒤에서부터 순회 (안정성 보장)
    for i in range(len(nums) - 1, -1, -1):
        output[count[nums[i]] - 1] = nums[i]
        count[nums[i]] -= 1

    return output


# 사용 예시
nums = [4, 2, 2, 8, 3, 3, 1, 0]
print(f"원본: {nums}")
print(f"계수 정렬: {counting_sort(nums)}")

nums_neg = [4, -2, 2, -8, 3, -3, 1]
print(f"\\n음수 포함: {nums_neg}")
print(f"계수 정렬: {counting_sort(nums_neg, -8, 4)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["sorting", "counting-sort", "non-comparison"],
    },
    {
        "id": str(uuid4()),
        "title": "Radix Sort (기수 정렬)",
        "description": "자릿수별로 정렬하는 O(d*(n+k)) 정렬 알고리즘",
        "code": '''def radix_sort_lsd(nums: list[int]) -> list[int]:
    """LSD(Least Significant Digit) 기수 정렬

    가장 낮은 자릿수부터 정렬합니다.
    시간복잡도: O(d * (n + k)), d=자릿수, k=기수(10)
    """
    if not nums:
        return []

    nums = nums.copy()
    max_val = max(nums)
    exp = 1

    while max_val // exp > 0:
        # 현재 자릿수 기준 계수 정렬
        count = [0] * 10
        output = [0] * len(nums)

        for num in nums:
            digit = (num // exp) % 10
            count[digit] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(len(nums) - 1, -1, -1):
            digit = (nums[i] // exp) % 10
            output[count[digit] - 1] = nums[i]
            count[digit] -= 1

        nums = output
        exp *= 10

    return nums


def radix_sort_msd(strs: list[str], d: int = 0) -> list[str]:
    """MSD(Most Significant Digit) 기수 정렬 - 문자열용

    가장 높은 자릿수(첫 문자)부터 정렬합니다.
    재귀적으로 각 버킷을 정렬합니다.
    """
    if len(strs) <= 1:
        return strs

    if d >= len(strs[0]):
        return strs

    # 26개 버킷 (a-z)
    buckets = [[] for _ in range(26)]
    for s in strs:
        idx = ord(s[d]) - ord('a')
        buckets[idx].append(s)

    result = []
    for bucket in buckets:
        if bucket:
            result.extend(radix_sort_msd(bucket, d + 1))

    return result


# 사용 예시
nums = [170, 45, 75, 90, 802, 24, 2, 66]
print(f"정수 배열: {nums}")
print(f"LSD 기수 정렬: {radix_sort_lsd(nums)}")

strs = ["cat", "bat", "cab", "abc", "bac"]
print(f"\\n문자열 배열: {strs}")
print(f"MSD 기수 정렬: {radix_sort_msd(strs)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["sorting", "radix-sort", "non-comparison"],
    },
    {
        "id": str(uuid4()),
        "title": "Shell Sort (셸 정렬)",
        "description": "삽입 정렬을 개선한 gap 기반 정렬 알고리즘",
        "code": '''def shell_sort(nums: list[int], gap_sequence: str = "shell") -> list[int]:
    """셸 정렬: 삽입 정렬의 개선 버전

    gap_sequence 옵션:
    - "shell": n/2, n/4, ..., 1 (기본)
    - "hibbard": 2^k - 1 시퀀스
    - "knuth": (3^k - 1) / 2 시퀀스
    """
    if not nums:
        return []

    nums = nums.copy()
    n = len(nums)

    # Gap 시퀀스 생성
    if gap_sequence == "hibbard":
        gaps = []
        k = 1
        while 2**k - 1 < n:
            gaps.append(2**k - 1)
            k += 1
        gaps.reverse()
    elif gap_sequence == "knuth":
        gaps = []
        h = 1
        while h < n // 3:
            gaps.append(h)
            h = 3 * h + 1
        gaps.reverse()
    else:  # shell
        gaps = []
        gap = n // 2
        while gap > 0:
            gaps.append(gap)
            gap //= 2

    # 각 gap에 대해 삽입 정렬 수행
    for gap in gaps:
        for i in range(gap, n):
            temp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > temp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = temp

    return nums


# 사용 예시
nums = [12, 34, 54, 2, 3, 9, 7, 1, 15, 6]
print(f"원본: {nums}")
print(f"Shell 간격: {shell_sort(nums, 'shell')}")
print(f"Hibbard 간격: {shell_sort(nums, 'hibbard')}")
print(f"Knuth 간격: {shell_sort(nums, 'knuth')}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["sorting", "shell-sort", "gap-sequence"],
    },
    {
        "id": str(uuid4()),
        "title": "Tree Sort & External Sort",
        "description": "BST 기반 트리 정렬과 K-way Merge 외부 정렬",
        "code": '''import heapq


class TreeNode:
    """BST 노드"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def tree_sort(nums: list[int]) -> list[int]:
    """트리 정렬: BST에 삽입 후 중위 순회

    시간복잡도: O(n log n) 평균, O(n^2) 최악
    """
    def insert(root, val):
        if root is None:
            return TreeNode(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)
        return root

    def inorder(root, result):
        if root:
            inorder(root.left, result)
            result.append(root.val)
            inorder(root.right, result)

    if not nums:
        return []

    root = None
    for num in nums:
        root = insert(root, num)

    result = []
    inorder(root, result)
    return result


def external_sort(nums: list[int], chunk_size: int) -> list[int]:
    """외부 정렬 시뮬레이션

    1. 배열을 chunk_size씩 나눠 정렬 (Run 생성)
    2. 힙 기반 K-way Merge

    실제 외부 정렬은 디스크 I/O를 고려합니다.
    """
    if not nums:
        return []

    # 1. Run 생성
    runs = []
    for i in range(0, len(nums), chunk_size):
        run = sorted(nums[i:i + chunk_size])
        runs.append(run)

    # 2. K-way Merge using min-heap
    # (값, run 인덱스, run 내 인덱스)
    heap = []
    for run_idx, run in enumerate(runs):
        if run:
            heapq.heappush(heap, (run[0], run_idx, 0))

    result = []
    while heap:
        val, run_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 같은 run에서 다음 원소 추가
        if elem_idx + 1 < len(runs[run_idx]):
            next_val = runs[run_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, run_idx, elem_idx + 1))

    return result


# 사용 예시
nums = [5, 3, 7, 2, 4, 6, 8, 1]
print(f"원본: {nums}")
print(f"트리 정렬: {tree_sort(nums)}")

nums2 = [5, 3, 8, 1, 2, 7, 4, 6]
print(f"\\n외부 정렬 (chunk=3): {external_sort(nums2, 3)}")
''',
        "language": PlaygroundLanguage.PYTHON.value,
        "category": TemplateCategory.ALGORITHM.value,
        "tags": ["sorting", "tree-sort", "external-sort", "k-way-merge"],
    },
]


async def seed_templates(session: AsyncSession) -> int:
    """Seed code templates if they don't exist."""
    # Check if templates already exist
    result = await session.execute(
        select(CodeTemplateModel).limit(1)
    )
    if result.scalar_one_or_none():
        return 0  # Templates already exist

    # Insert templates
    count = 0
    for template_data in TEMPLATES:
        # Convert string id to UUID if needed
        template_id = template_data["id"]
        if isinstance(template_id, str):
            template_id = UUID(template_id)

        # Convert tags list to JSON string if needed
        tags = template_data["tags"]
        if isinstance(tags, list):
            tags = json.dumps(tags)

        template = CodeTemplateModel(
            id=template_id,
            title=template_data["title"],
            description=template_data["description"],
            code=template_data["code"],
            language=template_data["language"],
            category=template_data["category"],
            tags=tags,
            usage_count=0,
        )
        session.add(template)
        count += 1

    await session.commit()
    return count
