"""Visualization generators module."""

from code_tutor.visualization.application.generators.graph import (
    generate_bfs,
    generate_dfs,
)
from code_tutor.visualization.application.generators.searching import (
    generate_binary_search,
    generate_linear_search,
)
from code_tutor.visualization.application.generators.sorting import (
    generate_bubble_sort,
    generate_insertion_sort,
    generate_merge_sort,
    generate_quick_sort,
    generate_selection_sort,
)

__all__ = [
    # Sorting
    "generate_bubble_sort",
    "generate_selection_sort",
    "generate_insertion_sort",
    "generate_quick_sort",
    "generate_merge_sort",
    # Searching
    "generate_linear_search",
    "generate_binary_search",
    # Graph
    "generate_bfs",
    "generate_dfs",
]
