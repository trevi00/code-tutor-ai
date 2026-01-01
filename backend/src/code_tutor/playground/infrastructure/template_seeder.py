"""Seed code templates for playground."""

from uuid import uuid4

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
        template = CodeTemplateModel(
            id=template_data["id"],
            title=template_data["title"],
            description=template_data["description"],
            code=template_data["code"],
            language=template_data["language"],
            category=template_data["category"],
            tags=template_data["tags"],
            usage_count=0,
        )
        session.add(template)
        count += 1

    await session.commit()
    return count
