"""Visualization application services."""

import random
from typing import Any

from code_tutor.visualization.application.generators import (
    generate_bfs,
    generate_binary_search,
    generate_bubble_sort,
    generate_dfs,
    generate_insertion_sort,
    generate_linear_search,
    generate_merge_sort,
    generate_quick_sort,
    generate_selection_sort,
)
from code_tutor.visualization.domain import (
    ALGORITHM_INFO,
    AlgorithmCategory,
    AlgorithmType,
    GraphVisualization,
    Visualization,
)


class VisualizationService:
    """Service for generating algorithm visualizations."""

    # Map algorithm types to their generator functions
    SORTING_GENERATORS = {
        AlgorithmType.BUBBLE_SORT: generate_bubble_sort,
        AlgorithmType.SELECTION_SORT: generate_selection_sort,
        AlgorithmType.INSERTION_SORT: generate_insertion_sort,
        AlgorithmType.QUICK_SORT: generate_quick_sort,
        AlgorithmType.MERGE_SORT: generate_merge_sort,
    }

    SEARCH_GENERATORS = {
        AlgorithmType.LINEAR_SEARCH: generate_linear_search,
        AlgorithmType.BINARY_SEARCH: generate_binary_search,
    }

    GRAPH_GENERATORS = {
        AlgorithmType.BFS: generate_bfs,
        AlgorithmType.DFS: generate_dfs,
    }

    def get_available_algorithms(self) -> list[dict]:
        """Get list of all available algorithms with metadata."""
        algorithms = []

        for algo_type, info in ALGORITHM_INFO.items():
            algorithms.append({
                "id": algo_type.value,
                "name": info["name"],
                "name_en": info["name_en"],
                "category": info["category"].value,
                "time_complexity": info["time_complexity"],
                "space_complexity": info["space_complexity"],
                "description": info["description"],
            })

        return algorithms

    def get_algorithms_by_category(self, category: AlgorithmCategory) -> list[dict]:
        """Get algorithms filtered by category."""
        return [
            algo for algo in self.get_available_algorithms()
            if algo["category"] == category.value
        ]

    def generate_random_array(
        self,
        size: int = 10,
        min_val: int = 1,
        max_val: int = 100,
    ) -> list[int]:
        """Generate a random array for visualization."""
        return [random.randint(min_val, max_val) for _ in range(size)]

    def generate_sorting_visualization(
        self,
        algorithm: AlgorithmType,
        data: list[int] | None = None,
        size: int = 10,
    ) -> Visualization:
        """Generate visualization for a sorting algorithm."""
        if algorithm not in self.SORTING_GENERATORS:
            raise ValueError(f"Unsupported sorting algorithm: {algorithm}")

        if data is None:
            data = self.generate_random_array(size)

        generator = self.SORTING_GENERATORS[algorithm]
        return generator(data)

    def generate_search_visualization(
        self,
        algorithm: AlgorithmType,
        data: list[int] | None = None,
        target: int | None = None,
        size: int = 10,
    ) -> Visualization:
        """Generate visualization for a search algorithm."""
        if algorithm not in self.SEARCH_GENERATORS:
            raise ValueError(f"Unsupported search algorithm: {algorithm}")

        if data is None:
            data = self.generate_random_array(size, 1, 50)
            # For binary search, sort the array
            if algorithm == AlgorithmType.BINARY_SEARCH:
                data = sorted(data)

        if target is None:
            # Pick a random target (either from array or not)
            if random.random() < 0.8:  # 80% chance to pick from array
                target = random.choice(data)
            else:
                target = random.randint(1, 100)

        generator = self.SEARCH_GENERATORS[algorithm]
        return generator(data, target)

    def generate_graph_visualization(
        self,
        algorithm: AlgorithmType,
        start_node: str = "A",
    ) -> GraphVisualization:
        """Generate visualization for a graph algorithm."""
        if algorithm not in self.GRAPH_GENERATORS:
            raise ValueError(f"Unsupported graph algorithm: {algorithm}")

        generator = self.GRAPH_GENERATORS[algorithm]
        return generator(start_node)

    def generate_visualization(
        self,
        algorithm: AlgorithmType,
        **kwargs: Any,
    ) -> Visualization | GraphVisualization:
        """Generate visualization for any supported algorithm."""
        if algorithm in self.SORTING_GENERATORS:
            return self.generate_sorting_visualization(algorithm, **kwargs)
        elif algorithm in self.SEARCH_GENERATORS:
            return self.generate_search_visualization(algorithm, **kwargs)
        elif algorithm in self.GRAPH_GENERATORS:
            return self.generate_graph_visualization(algorithm, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
