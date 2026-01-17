#!/usr/bin/env python
"""Main seed script to populate database with initial data.

This script runs all individual seed scripts in the correct order.

Usage:
    # From backend directory
    uv run python scripts/seed_all.py

    # Or with options
    uv run python scripts/seed_all.py --skip-problems  # Skip problem seeding
    uv run python scripts/seed_all.py --only users,badges  # Only seed specific data
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root and scripts to path
project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(scripts_dir))

from code_tutor.shared.infrastructure.database import init_db


async def run_seed_users():
    """Run user seeding."""
    from seed_users import seed_users
    return await seed_users()


async def run_seed_badges():
    """Run badge and challenge seeding."""
    from seed_badges import seed_gamification
    return await seed_gamification()


async def run_seed_roadmap():
    """Run roadmap seeding."""
    from seed_roadmap import seed_roadmap
    return await seed_roadmap()


async def run_seed_typing():
    """Run typing exercises seeding."""
    from seed_typing_exercises import seed_typing_exercises
    return seed_typing_exercises()


async def run_seed_problems():
    """Run problem seeding using SQLAlchemy."""
    from seed_sample_problems import seed_sample_problems
    return await seed_sample_problems()


# Seed functions registry
SEEDERS = {
    "users": {
        "func": run_seed_users,
        "description": "Admin and test users",
        "order": 1,
    },
    "badges": {
        "func": run_seed_badges,
        "description": "Gamification badges and challenges",
        "order": 2,
    },
    "roadmap": {
        "func": run_seed_roadmap,
        "description": "Learning paths, modules, and lessons",
        "order": 3,
    },
    "typing": {
        "func": run_seed_typing,
        "description": "Typing practice exercises",
        "order": 4,
    },
    "problems": {
        "func": run_seed_problems,
        "description": "Sample coding problems (6 problems)",
        "order": 5,
    },
}


async def seed_all(skip: list[str] = None, only: list[str] = None):
    """Run all seed scripts.

    Args:
        skip: List of seeders to skip
        only: List of seeders to run (exclusive)
    """
    skip = skip or []
    only = only or []

    print("=" * 60)
    print("  Code Tutor AI - Database Seeding")
    print("=" * 60)

    # Initialize database
    print("\nInitializing database connection...")
    await init_db()

    # Determine which seeders to run
    if only:
        seeders_to_run = {k: v for k, v in SEEDERS.items() if k in only}
    else:
        seeders_to_run = {k: v for k, v in SEEDERS.items() if k not in skip}

    # Sort by order
    sorted_seeders = sorted(seeders_to_run.items(), key=lambda x: x[1]["order"])

    total_start = time.time()
    results = {}

    for name, config in sorted_seeders:
        print(f"\n{'=' * 60}")
        print(f"  Seeding: {name.upper()}")
        print(f"  Description: {config['description']}")
        print("=" * 60)

        start = time.time()
        try:
            result = await config["func"]()
            elapsed = time.time() - start
            results[name] = {"status": "success", "result": result, "time": elapsed}
            print(f"\n[OK] {name} completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[name] = {"status": "error", "error": str(e), "time": elapsed}
            print(f"\n[ERROR] {name} failed: {e}")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  SEEDING SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        status = "OK" if result["status"] == "success" else "FAILED"
        print(f"  {name}: {status} ({result['time']:.2f}s)")

    print("-" * 60)
    print(f"  Total time: {total_elapsed:.2f}s")
    print("=" * 60)

    # Return success if all seeders succeeded
    return all(r["status"] == "success" for r in results.values())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed database with initial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_all.py                    # Run all seeders
  python scripts/seed_all.py --skip problems    # Skip problem seeding
  python scripts/seed_all.py --only users,badges  # Only seed users and badges
  python scripts/seed_all.py --list             # List available seeders
        """
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of seeders to skip",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated list of seeders to run exclusively",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available seeders",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable seeders:")
        print("-" * 50)
        for name, config in sorted(SEEDERS.items(), key=lambda x: x[1]["order"]):
            print(f"  {name:12} - {config['description']}")
        print()
        return

    skip = args.skip.split(",") if args.skip else []
    only = args.only.split(",") if args.only else []

    # Validate seeder names
    all_names = set(SEEDERS.keys())
    invalid_skip = set(skip) - all_names
    invalid_only = set(only) - all_names

    if invalid_skip:
        print(f"Error: Unknown seeders to skip: {invalid_skip}")
        sys.exit(1)

    if invalid_only:
        print(f"Error: Unknown seeders to run: {invalid_only}")
        sys.exit(1)

    success = asyncio.run(seed_all(skip=skip, only=only))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
