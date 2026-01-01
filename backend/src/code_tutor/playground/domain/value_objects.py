"""Playground domain value objects."""

from enum import Enum


class PlaygroundLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"


class PlaygroundVisibility(str, Enum):
    """Playground visibility settings."""

    PRIVATE = "private"  # Only owner can see
    UNLISTED = "unlisted"  # Anyone with link can see
    PUBLIC = "public"  # Listed in public gallery


class TemplateCategory(str, Enum):
    """Template categories."""

    BASIC = "basic"
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    PATTERN = "pattern"
    SNIPPET = "snippet"
    STARTER = "starter"
    UTILITY = "utility"


# Language configurations
LANGUAGE_CONFIG = {
    PlaygroundLanguage.PYTHON: {
        "extension": ".py",
        "docker_image": "python:3.11-slim",
        "run_command": "python -u {file}",
        "display_name": "Python 3.11",
    },
    PlaygroundLanguage.JAVASCRIPT: {
        "extension": ".js",
        "docker_image": "node:20-slim",
        "run_command": "node {file}",
        "display_name": "JavaScript (Node.js 20)",
    },
    PlaygroundLanguage.TYPESCRIPT: {
        "extension": ".ts",
        "docker_image": "node:20-slim",
        "run_command": "npx ts-node {file}",
        "display_name": "TypeScript",
    },
    PlaygroundLanguage.JAVA: {
        "extension": ".java",
        "docker_image": "openjdk:17-slim",
        "run_command": "java {file}",
        "display_name": "Java 17",
    },
    PlaygroundLanguage.CPP: {
        "extension": ".cpp",
        "docker_image": "gcc:12",
        "run_command": "g++ -o /tmp/a.out {file} && /tmp/a.out",
        "display_name": "C++ (GCC 12)",
    },
    PlaygroundLanguage.C: {
        "extension": ".c",
        "docker_image": "gcc:12",
        "run_command": "gcc -o /tmp/a.out {file} && /tmp/a.out",
        "display_name": "C (GCC 12)",
    },
    PlaygroundLanguage.GO: {
        "extension": ".go",
        "docker_image": "golang:1.21-alpine",
        "run_command": "go run {file}",
        "display_name": "Go 1.21",
    },
    PlaygroundLanguage.RUST: {
        "extension": ".rs",
        "docker_image": "rust:1.74-slim",
        "run_command": "rustc -o /tmp/a.out {file} && /tmp/a.out",
        "display_name": "Rust 1.74",
    },
}


# Default code templates per language
DEFAULT_CODE = {
    PlaygroundLanguage.PYTHON: '''# Python Playground
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
''',
    PlaygroundLanguage.JAVASCRIPT: '''// JavaScript Playground
function main() {
    console.log("Hello, World!");
}

main();
''',
    PlaygroundLanguage.TYPESCRIPT: '''// TypeScript Playground
function main(): void {
    console.log("Hello, World!");
}

main();
''',
    PlaygroundLanguage.JAVA: '''// Java Playground
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
''',
    PlaygroundLanguage.CPP: '''// C++ Playground
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
''',
    PlaygroundLanguage.C: '''// C Playground
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
''',
    PlaygroundLanguage.GO: '''// Go Playground
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
''',
    PlaygroundLanguage.RUST: '''// Rust Playground
fn main() {
    println!("Hello, World!");
}
''',
}
