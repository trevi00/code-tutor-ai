"""Tests for Visualization Module."""

import pytest

from code_tutor.visualization.domain.value_objects import (
    AlgorithmCategory,
    AlgorithmType,
    AnimationType,
    ElementState,
    ALGORITHM_INFO,
)
from code_tutor.visualization.domain.entities import (
    VisualizationStep,
    TreeNode,
    GraphNode,
    GraphEdge,
    Visualization,
    TreeVisualization,
    GraphVisualization,
)
from code_tutor.visualization.application.generators.sorting import (
    generate_bubble_sort,
    generate_selection_sort,
    generate_insertion_sort,
    generate_quick_sort,
    generate_merge_sort,
)
from code_tutor.visualization.application.generators.searching import (
    generate_linear_search,
    generate_binary_search,
)
from code_tutor.visualization.application.generators.graph import (
    create_sample_graph,
    generate_bfs,
    generate_dfs,
)
from code_tutor.visualization.application.services import VisualizationService


class TestAlgorithmCategory:
    """Tests for AlgorithmCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert AlgorithmCategory.SORTING.value == "sorting"
        assert AlgorithmCategory.SEARCHING.value == "searching"
        assert AlgorithmCategory.DATA_STRUCTURE.value == "data_structure"
        assert AlgorithmCategory.GRAPH.value == "graph"
        assert AlgorithmCategory.DYNAMIC_PROGRAMMING.value == "dynamic_programming"

    def test_category_from_string(self):
        """Test creating category from string."""
        assert AlgorithmCategory("sorting") == AlgorithmCategory.SORTING
        assert AlgorithmCategory("graph") == AlgorithmCategory.GRAPH


class TestAlgorithmType:
    """Tests for AlgorithmType enum."""

    def test_sorting_types(self):
        """Test sorting algorithm types."""
        assert AlgorithmType.BUBBLE_SORT.value == "bubble_sort"
        assert AlgorithmType.SELECTION_SORT.value == "selection_sort"
        assert AlgorithmType.INSERTION_SORT.value == "insertion_sort"
        assert AlgorithmType.MERGE_SORT.value == "merge_sort"
        assert AlgorithmType.QUICK_SORT.value == "quick_sort"

    def test_search_types(self):
        """Test search algorithm types."""
        assert AlgorithmType.LINEAR_SEARCH.value == "linear_search"
        assert AlgorithmType.BINARY_SEARCH.value == "binary_search"

    def test_graph_types(self):
        """Test graph algorithm types."""
        assert AlgorithmType.BFS.value == "bfs"
        assert AlgorithmType.DFS.value == "dfs"


class TestAnimationType:
    """Tests for AnimationType enum."""

    def test_animation_types(self):
        """Test animation type values."""
        assert AnimationType.COMPARE.value == "compare"
        assert AnimationType.SWAP.value == "swap"
        assert AnimationType.SET.value == "set"
        assert AnimationType.HIGHLIGHT.value == "highlight"
        assert AnimationType.SORTED.value == "sorted"
        assert AnimationType.FOUND.value == "found"
        assert AnimationType.NOT_FOUND.value == "not_found"


class TestElementState:
    """Tests for ElementState enum."""

    def test_element_states(self):
        """Test element state values."""
        assert ElementState.DEFAULT.value == "default"
        assert ElementState.COMPARING.value == "comparing"
        assert ElementState.SWAPPING.value == "swapping"
        assert ElementState.SORTED.value == "sorted"
        assert ElementState.PIVOT.value == "pivot"
        assert ElementState.FOUND.value == "found"
        assert ElementState.VISITED.value == "visited"


class TestAlgorithmInfo:
    """Tests for ALGORITHM_INFO dictionary."""

    def test_all_algorithms_have_info(self):
        """Test that all algorithm types in info have required fields."""
        required_fields = ["name", "name_en", "category", "time_complexity", "space_complexity", "description"]

        for algo_type, info in ALGORITHM_INFO.items():
            for field in required_fields:
                assert field in info, f"{algo_type} missing {field}"

    def test_bubble_sort_info(self):
        """Test bubble sort metadata."""
        info = ALGORITHM_INFO[AlgorithmType.BUBBLE_SORT]
        assert info["name"] == "버블 정렬"
        assert info["name_en"] == "Bubble Sort"
        assert info["category"] == AlgorithmCategory.SORTING
        assert info["time_complexity"] == "O(n²)"
        assert info["space_complexity"] == "O(1)"

    def test_binary_search_info(self):
        """Test binary search metadata."""
        info = ALGORITHM_INFO[AlgorithmType.BINARY_SEARCH]
        assert info["name"] == "이진 탐색"
        assert info["category"] == AlgorithmCategory.SEARCHING
        assert info["time_complexity"] == "O(log n)"


class TestVisualizationStep:
    """Tests for VisualizationStep entity."""

    def test_step_creation(self):
        """Test creating a visualization step."""
        step = VisualizationStep(
            step_number=0,
            action=AnimationType.COMPARE,
            indices=[0, 1],
            values=[5, 3],
            array_state=[5, 3, 8, 1],
            element_states=[ElementState.COMPARING, ElementState.COMPARING, ElementState.DEFAULT, ElementState.DEFAULT],
            description="비교 중",
            code_line=4,
        )

        assert step.step_number == 0
        assert step.action == AnimationType.COMPARE
        assert step.indices == [0, 1]
        assert step.values == [5, 3]
        assert step.description == "비교 중"
        assert step.code_line == 4

    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = VisualizationStep(
            step_number=1,
            action=AnimationType.SWAP,
            indices=[0, 1],
            values=[3, 5],
            array_state=[3, 5, 8, 1],
            element_states=[ElementState.SWAPPING, ElementState.SWAPPING, ElementState.DEFAULT, ElementState.DEFAULT],
            description="교환",
            code_line=5,
            auxiliary_data={"i": 0, "j": 1},
        )

        data = step.to_dict()

        assert data["step_number"] == 1
        assert data["action"] == "swap"
        assert data["indices"] == [0, 1]
        assert data["element_states"] == ["swapping", "swapping", "default", "default"]
        assert data["auxiliary_data"] == {"i": 0, "j": 1}


class TestTreeNode:
    """Tests for TreeNode entity."""

    def test_tree_node_creation(self):
        """Test creating a tree node."""
        node = TreeNode(
            id="1",
            value=10,
            x=100,
            y=50,
        )

        assert node.id == "1"
        assert node.value == 10
        assert node.left is None
        assert node.right is None
        assert node.state == ElementState.DEFAULT

    def test_tree_node_with_children(self):
        """Test tree node with children."""
        left = TreeNode(id="2", value=5, x=50, y=100)
        right = TreeNode(id="3", value=15, x=150, y=100)
        root = TreeNode(id="1", value=10, x=100, y=50, left=left, right=right)

        assert root.left == left
        assert root.right == right

    def test_tree_node_to_dict(self):
        """Test converting tree node to dictionary."""
        left = TreeNode(id="2", value=5, x=50, y=100)
        root = TreeNode(id="1", value=10, x=100, y=50, left=left, state=ElementState.CURRENT)

        data = root.to_dict()

        assert data["id"] == "1"
        assert data["value"] == 10
        assert data["state"] == "current"
        assert data["left"]["id"] == "2"
        assert data["right"] is None


class TestGraphNode:
    """Tests for GraphNode entity."""

    def test_graph_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(id="A", value="A", x=100, y=100)

        assert node.id == "A"
        assert node.value == "A"
        assert node.x == 100
        assert node.y == 100
        assert node.state == ElementState.DEFAULT

    def test_graph_node_to_dict(self):
        """Test converting graph node to dictionary."""
        node = GraphNode(id="A", value="A", x=100, y=100, state=ElementState.VISITED)
        data = node.to_dict()

        assert data["id"] == "A"
        assert data["x"] == 100
        assert data["state"] == "visited"


class TestGraphEdge:
    """Tests for GraphEdge entity."""

    def test_graph_edge_creation(self):
        """Test creating a graph edge."""
        edge = GraphEdge(source="A", target="B")

        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.weight == 1.0
        assert edge.state == ElementState.DEFAULT

    def test_graph_edge_with_weight(self):
        """Test graph edge with custom weight."""
        edge = GraphEdge(source="A", target="B", weight=5.0)
        assert edge.weight == 5.0

    def test_graph_edge_to_dict(self):
        """Test converting graph edge to dictionary."""
        edge = GraphEdge(source="A", target="B", weight=3.0, state=ElementState.ACTIVE)
        data = edge.to_dict()

        assert data["source"] == "A"
        assert data["target"] == "B"
        assert data["weight"] == 3.0
        assert data["state"] == "active"


class TestVisualization:
    """Tests for Visualization entity."""

    def test_visualization_creation(self):
        """Test creating a visualization."""
        viz = Visualization(
            algorithm_type=AlgorithmType.BUBBLE_SORT,
            category=AlgorithmCategory.SORTING,
            initial_data=[5, 3, 8, 1],
            code="def bubble_sort(arr): pass",
        )

        assert viz.algorithm_type == AlgorithmType.BUBBLE_SORT
        assert viz.category == AlgorithmCategory.SORTING
        assert viz.initial_data == [5, 3, 8, 1]
        assert viz.steps == []
        assert viz.total_comparisons == 0

    def test_visualization_add_step(self):
        """Test adding steps to visualization."""
        viz = Visualization(
            algorithm_type=AlgorithmType.BUBBLE_SORT,
            category=AlgorithmCategory.SORTING,
            initial_data=[5, 3],
        )

        step = VisualizationStep(
            step_number=0,
            action=AnimationType.COMPARE,
            indices=[0, 1],
            values=[5, 3],
            array_state=[5, 3],
            element_states=[ElementState.COMPARING, ElementState.COMPARING],
            description="비교",
        )

        viz.add_step(step)

        assert len(viz.steps) == 1
        assert viz.steps[0] == step

    def test_visualization_to_dict(self):
        """Test converting visualization to dictionary."""
        viz = Visualization(
            algorithm_type=AlgorithmType.BUBBLE_SORT,
            category=AlgorithmCategory.SORTING,
            initial_data=[5, 3],
            final_data=[3, 5],
            total_comparisons=1,
            total_swaps=1,
        )

        data = viz.to_dict()

        assert data["algorithm_type"] == "bubble_sort"
        assert data["category"] == "sorting"
        assert data["initial_data"] == [5, 3]
        assert data["final_data"] == [3, 5]
        assert data["total_comparisons"] == 1
        assert data["total_swaps"] == 1


class TestTreeVisualization:
    """Tests for TreeVisualization entity."""

    def test_tree_visualization_creation(self):
        """Test creating tree visualization."""
        root = TreeNode(id="1", value=10, x=100, y=50)
        viz = TreeVisualization(
            algorithm_type=AlgorithmType.BST,
            root=root,
            code="def insert(root, val): pass",
        )

        assert viz.algorithm_type == AlgorithmType.BST
        assert viz.root == root
        assert viz.steps == []

    def test_tree_visualization_to_dict(self):
        """Test converting tree visualization to dictionary."""
        root = TreeNode(id="1", value=10, x=100, y=50)
        viz = TreeVisualization(
            algorithm_type=AlgorithmType.BST,
            root=root,
        )

        data = viz.to_dict()

        assert data["algorithm_type"] == "bst"
        assert data["root"]["id"] == "1"
        assert data["total_steps"] == 0


class TestGraphVisualization:
    """Tests for GraphVisualization entity."""

    def test_graph_visualization_creation(self):
        """Test creating graph visualization."""
        nodes = [GraphNode(id="A", value="A", x=100, y=100)]
        edges = [GraphEdge(source="A", target="B")]
        viz = GraphVisualization(
            algorithm_type=AlgorithmType.BFS,
            nodes=nodes,
            edges=edges,
        )

        assert viz.algorithm_type == AlgorithmType.BFS
        assert len(viz.nodes) == 1
        assert len(viz.edges) == 1

    def test_graph_visualization_to_dict(self):
        """Test converting graph visualization to dictionary."""
        nodes = [GraphNode(id="A", value="A", x=100, y=100)]
        edges = [GraphEdge(source="A", target="B")]
        viz = GraphVisualization(
            algorithm_type=AlgorithmType.BFS,
            nodes=nodes,
            edges=edges,
        )

        data = viz.to_dict()

        assert data["algorithm_type"] == "bfs"
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1


# =====================
# Sorting Generator Tests
# =====================

class TestBubbleSortGenerator:
    """Tests for bubble sort visualization generator."""

    def test_bubble_sort_basic(self):
        """Test bubble sort with basic input."""
        data = [5, 3, 8, 1]
        viz = generate_bubble_sort(data)

        assert viz.algorithm_type == AlgorithmType.BUBBLE_SORT
        assert viz.category == AlgorithmCategory.SORTING
        assert viz.initial_data == [5, 3, 8, 1]
        assert viz.final_data == [1, 3, 5, 8]
        assert len(viz.steps) > 0

    def test_bubble_sort_already_sorted(self):
        """Test bubble sort with already sorted array."""
        data = [1, 2, 3, 4]
        viz = generate_bubble_sort(data)

        assert viz.final_data == [1, 2, 3, 4]
        assert viz.total_swaps == 0

    def test_bubble_sort_reverse_sorted(self):
        """Test bubble sort with reverse sorted array."""
        data = [4, 3, 2, 1]
        viz = generate_bubble_sort(data)

        assert viz.final_data == [1, 2, 3, 4]
        assert viz.total_swaps > 0

    def test_bubble_sort_single_element(self):
        """Test bubble sort with single element."""
        data = [42]
        viz = generate_bubble_sort(data)

        assert viz.final_data == [42]

    def test_bubble_sort_step_structure(self):
        """Test bubble sort step structure."""
        data = [3, 1]
        viz = generate_bubble_sort(data)

        # Check first step is initial state
        first_step = viz.steps[0]
        assert first_step.action == AnimationType.HIGHLIGHT
        assert first_step.description == "정렬을 시작합니다."


class TestSelectionSortGenerator:
    """Tests for selection sort visualization generator."""

    def test_selection_sort_basic(self):
        """Test selection sort with basic input."""
        data = [5, 3, 8, 1]
        viz = generate_selection_sort(data)

        assert viz.algorithm_type == AlgorithmType.SELECTION_SORT
        assert viz.final_data == [1, 3, 5, 8]

    def test_selection_sort_already_sorted(self):
        """Test selection sort with already sorted array."""
        data = [1, 2, 3, 4]
        viz = generate_selection_sort(data)

        assert viz.final_data == [1, 2, 3, 4]

    def test_selection_sort_with_duplicates(self):
        """Test selection sort with duplicate values."""
        data = [3, 1, 3, 2]
        viz = generate_selection_sort(data)

        assert viz.final_data == [1, 2, 3, 3]


class TestInsertionSortGenerator:
    """Tests for insertion sort visualization generator."""

    def test_insertion_sort_basic(self):
        """Test insertion sort with basic input."""
        data = [5, 3, 8, 1]
        viz = generate_insertion_sort(data)

        assert viz.algorithm_type == AlgorithmType.INSERTION_SORT
        assert viz.final_data == [1, 3, 5, 8]

    def test_insertion_sort_already_sorted(self):
        """Test insertion sort with already sorted array."""
        data = [1, 2, 3, 4]
        viz = generate_insertion_sort(data)

        assert viz.final_data == [1, 2, 3, 4]
        # Should have minimal swaps
        assert viz.total_swaps == 0

    def test_insertion_sort_code_present(self):
        """Test that insertion sort includes code."""
        data = [3, 1]
        viz = generate_insertion_sort(data)

        assert "insertion_sort" in viz.code
        assert "key = arr[i]" in viz.code


class TestQuickSortGenerator:
    """Tests for quick sort visualization generator."""

    def test_quick_sort_basic(self):
        """Test quick sort with basic input."""
        data = [5, 3, 8, 1, 9, 2]
        viz = generate_quick_sort(data)

        assert viz.algorithm_type == AlgorithmType.QUICK_SORT
        assert viz.final_data == [1, 2, 3, 5, 8, 9]

    def test_quick_sort_already_sorted(self):
        """Test quick sort with already sorted array."""
        data = [1, 2, 3, 4]
        viz = generate_quick_sort(data)

        assert viz.final_data == [1, 2, 3, 4]

    def test_quick_sort_pivot_step(self):
        """Test quick sort includes pivot selection steps."""
        data = [5, 3, 8, 1]
        viz = generate_quick_sort(data)

        pivot_steps = [s for s in viz.steps if s.action == AnimationType.PIVOT]
        assert len(pivot_steps) > 0


class TestMergeSortGenerator:
    """Tests for merge sort visualization generator."""

    def test_merge_sort_basic(self):
        """Test merge sort with basic input."""
        data = [5, 3, 8, 1]
        viz = generate_merge_sort(data)

        assert viz.algorithm_type == AlgorithmType.MERGE_SORT
        assert viz.final_data == [1, 3, 5, 8]

    def test_merge_sort_divide_steps(self):
        """Test merge sort includes divide steps."""
        data = [5, 3, 8, 1]
        viz = generate_merge_sort(data)

        divide_steps = [s for s in viz.steps if s.action == AnimationType.DIVIDE]
        assert len(divide_steps) > 0

    def test_merge_sort_merge_steps(self):
        """Test merge sort includes merge steps."""
        data = [5, 3, 8, 1]
        viz = generate_merge_sort(data)

        merge_steps = [s for s in viz.steps if s.action == AnimationType.MERGE]
        assert len(merge_steps) > 0


# =====================
# Search Generator Tests
# =====================

class TestLinearSearchGenerator:
    """Tests for linear search visualization generator."""

    def test_linear_search_found(self):
        """Test linear search when target is found."""
        data = [5, 3, 8, 1, 9]
        target = 8
        viz = generate_linear_search(data, target)

        assert viz.algorithm_type == AlgorithmType.LINEAR_SEARCH
        assert viz.category == AlgorithmCategory.SEARCHING

        found_steps = [s for s in viz.steps if s.action == AnimationType.FOUND]
        assert len(found_steps) == 1

    def test_linear_search_not_found(self):
        """Test linear search when target is not found."""
        data = [5, 3, 8, 1, 9]
        target = 100
        viz = generate_linear_search(data, target)

        not_found_steps = [s for s in viz.steps if s.action == AnimationType.NOT_FOUND]
        assert len(not_found_steps) == 1

    def test_linear_search_first_element(self):
        """Test linear search finding first element."""
        data = [5, 3, 8, 1]
        target = 5
        viz = generate_linear_search(data, target)

        assert viz.total_comparisons == 1

    def test_linear_search_last_element(self):
        """Test linear search finding last element."""
        data = [5, 3, 8, 1]
        target = 1
        viz = generate_linear_search(data, target)

        assert viz.total_comparisons == 4


class TestBinarySearchGenerator:
    """Tests for binary search visualization generator."""

    def test_binary_search_found(self):
        """Test binary search when target is found."""
        data = [1, 3, 5, 7, 9, 11, 13]
        target = 7
        viz = generate_binary_search(data, target)

        assert viz.algorithm_type == AlgorithmType.BINARY_SEARCH

        found_steps = [s for s in viz.steps if s.action == AnimationType.FOUND]
        assert len(found_steps) == 1

    def test_binary_search_not_found(self):
        """Test binary search when target is not found."""
        data = [1, 3, 5, 7, 9]
        target = 4
        viz = generate_binary_search(data, target)

        not_found_steps = [s for s in viz.steps if s.action == AnimationType.NOT_FOUND]
        assert len(not_found_steps) == 1

    def test_binary_search_sorts_input(self):
        """Test that binary search sorts the input array."""
        data = [9, 3, 7, 1, 5]
        target = 5
        viz = generate_binary_search(data, target)

        # Initial data should be sorted
        assert viz.initial_data == sorted(data)

    def test_binary_search_efficiency(self):
        """Test binary search is efficient (log n comparisons)."""
        data = list(range(1, 17))  # 16 elements
        target = 8
        viz = generate_binary_search(data, target)

        # Should find in log2(16) = 4 or fewer comparisons
        assert viz.total_comparisons <= 4


# =====================
# Graph Generator Tests
# =====================

class TestCreateSampleGraph:
    """Tests for sample graph creation."""

    def test_create_sample_graph_structure(self):
        """Test sample graph has correct structure."""
        nodes, edges, adjacency = create_sample_graph()

        assert len(nodes) == 7
        assert len(adjacency) == 7

        # Check node labels
        node_ids = {n.id for n in nodes}
        assert node_ids == {"A", "B", "C", "D", "E", "F", "G"}

    def test_create_sample_graph_edges(self):
        """Test sample graph edges are bidirectional."""
        nodes, edges, adjacency = create_sample_graph()

        # Check A is connected to B
        assert "B" in adjacency["A"]
        assert "A" in adjacency["B"]

    def test_create_sample_graph_positions(self):
        """Test nodes have valid positions."""
        nodes, edges, adjacency = create_sample_graph()

        for node in nodes:
            assert node.x > 0
            assert node.y > 0


class TestBFSGenerator:
    """Tests for BFS visualization generator."""

    def test_bfs_basic(self):
        """Test BFS with default start node."""
        viz = generate_bfs("A")

        assert viz.algorithm_type == AlgorithmType.BFS
        assert len(viz.steps) > 0
        assert len(viz.nodes) == 7

    def test_bfs_visits_all_nodes(self):
        """Test BFS visits all connected nodes."""
        viz = generate_bfs("A")

        # Find complete step
        complete_step = None
        for step in viz.steps:
            if step.get("action") == "complete":
                complete_step = step
                break

        assert complete_step is not None
        assert len(complete_step["visited"]) == 7

    def test_bfs_init_step(self):
        """Test BFS has initialization step."""
        viz = generate_bfs("A")

        first_step = viz.steps[0]
        assert first_step["action"] == "init"
        assert "BFS를" in first_step["description"]

    def test_bfs_includes_code(self):
        """Test BFS includes code."""
        viz = generate_bfs("A")

        assert "bfs" in viz.code
        assert "queue" in viz.code


class TestDFSGenerator:
    """Tests for DFS visualization generator."""

    def test_dfs_basic(self):
        """Test DFS with default start node."""
        viz = generate_dfs("A")

        assert viz.algorithm_type == AlgorithmType.DFS
        assert len(viz.steps) > 0

    def test_dfs_visits_all_nodes(self):
        """Test DFS visits all connected nodes."""
        viz = generate_dfs("A")

        # Find complete step
        complete_step = None
        for step in viz.steps:
            if step.get("action") == "complete":
                complete_step = step
                break

        assert complete_step is not None
        assert len(complete_step["visited"]) == 7

    def test_dfs_has_backtrack_steps(self):
        """Test DFS includes backtrack steps."""
        viz = generate_dfs("A")

        backtrack_steps = [s for s in viz.steps if s.get("action") == "backtrack"]
        assert len(backtrack_steps) > 0

    def test_dfs_includes_code(self):
        """Test DFS includes code."""
        viz = generate_dfs("A")

        assert "dfs" in viz.code
        assert "visited" in viz.code


# =====================
# Visualization Service Tests
# =====================

class TestVisualizationService:
    """Tests for VisualizationService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VisualizationService()

    def test_get_available_algorithms(self, service):
        """Test getting all available algorithms."""
        algorithms = service.get_available_algorithms()

        assert len(algorithms) > 0

        for algo in algorithms:
            assert "id" in algo
            assert "name" in algo
            assert "name_en" in algo
            assert "category" in algo
            assert "time_complexity" in algo
            assert "space_complexity" in algo

    def test_get_algorithms_by_category_sorting(self, service):
        """Test filtering algorithms by sorting category."""
        sorting_algos = service.get_algorithms_by_category(AlgorithmCategory.SORTING)

        assert len(sorting_algos) > 0
        for algo in sorting_algos:
            assert algo["category"] == "sorting"

    def test_get_algorithms_by_category_searching(self, service):
        """Test filtering algorithms by searching category."""
        search_algos = service.get_algorithms_by_category(AlgorithmCategory.SEARCHING)

        assert len(search_algos) > 0
        for algo in search_algos:
            assert algo["category"] == "searching"

    def test_generate_random_array(self, service):
        """Test random array generation."""
        arr = service.generate_random_array(size=10, min_val=1, max_val=100)

        assert len(arr) == 10
        assert all(1 <= x <= 100 for x in arr)

    def test_generate_random_array_custom_size(self, service):
        """Test random array with custom size."""
        arr = service.generate_random_array(size=5)
        assert len(arr) == 5

    def test_generate_sorting_visualization_bubble(self, service):
        """Test generating bubble sort visualization."""
        data = [5, 3, 8, 1]
        viz = service.generate_sorting_visualization(AlgorithmType.BUBBLE_SORT, data=data)

        assert viz.algorithm_type == AlgorithmType.BUBBLE_SORT
        assert viz.final_data == [1, 3, 5, 8]

    def test_generate_sorting_visualization_without_data(self, service):
        """Test generating sorting visualization with random data."""
        viz = service.generate_sorting_visualization(AlgorithmType.BUBBLE_SORT, size=8)

        assert viz.algorithm_type == AlgorithmType.BUBBLE_SORT
        assert len(viz.initial_data) == 8

    def test_generate_sorting_visualization_invalid_algorithm(self, service):
        """Test generating sorting visualization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported sorting algorithm"):
            service.generate_sorting_visualization(AlgorithmType.BFS)

    def test_generate_search_visualization_linear(self, service):
        """Test generating linear search visualization."""
        data = [5, 3, 8, 1]
        viz = service.generate_search_visualization(AlgorithmType.LINEAR_SEARCH, data=data, target=8)

        assert viz.algorithm_type == AlgorithmType.LINEAR_SEARCH

    def test_generate_search_visualization_binary(self, service):
        """Test generating binary search visualization."""
        data = [1, 3, 5, 7, 9]
        viz = service.generate_search_visualization(AlgorithmType.BINARY_SEARCH, data=data, target=5)

        assert viz.algorithm_type == AlgorithmType.BINARY_SEARCH

    def test_generate_search_visualization_without_target(self, service):
        """Test generating search visualization with random target."""
        viz = service.generate_search_visualization(AlgorithmType.LINEAR_SEARCH, size=10)

        assert viz.algorithm_type == AlgorithmType.LINEAR_SEARCH

    def test_generate_search_visualization_invalid_algorithm(self, service):
        """Test generating search visualization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported search algorithm"):
            service.generate_search_visualization(AlgorithmType.BUBBLE_SORT)

    def test_generate_graph_visualization_bfs(self, service):
        """Test generating BFS visualization."""
        viz = service.generate_graph_visualization(AlgorithmType.BFS, start_node="A")

        assert viz.algorithm_type == AlgorithmType.BFS
        assert len(viz.nodes) == 7

    def test_generate_graph_visualization_dfs(self, service):
        """Test generating DFS visualization."""
        viz = service.generate_graph_visualization(AlgorithmType.DFS, start_node="A")

        assert viz.algorithm_type == AlgorithmType.DFS

    def test_generate_graph_visualization_invalid_algorithm(self, service):
        """Test generating graph visualization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported graph algorithm"):
            service.generate_graph_visualization(AlgorithmType.BUBBLE_SORT)

    def test_generate_visualization_sorting(self, service):
        """Test generic visualization for sorting."""
        viz = service.generate_visualization(AlgorithmType.BUBBLE_SORT, data=[3, 1, 2])

        assert viz.algorithm_type == AlgorithmType.BUBBLE_SORT

    def test_generate_visualization_search(self, service):
        """Test generic visualization for search."""
        viz = service.generate_visualization(AlgorithmType.LINEAR_SEARCH, data=[3, 1, 2], target=1)

        assert viz.algorithm_type == AlgorithmType.LINEAR_SEARCH

    def test_generate_visualization_graph(self, service):
        """Test generic visualization for graph."""
        viz = service.generate_visualization(AlgorithmType.BFS, start_node="A")

        assert viz.algorithm_type == AlgorithmType.BFS

    def test_generate_visualization_invalid_algorithm(self, service):
        """Test generic visualization with unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            service.generate_visualization(AlgorithmType.DIJKSTRA)


# =====================
# API Route Tests
# =====================

@pytest.fixture
def test_client():
    """Create test client for API testing."""
    from fastapi.testclient import TestClient
    from code_tutor.visualization.interface.routes import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestVisualizationRoutes:
    """Tests for Visualization API routes."""

    def test_list_algorithms(self, test_client):
        """Test GET /visualization/algorithms."""
        response = test_client.get("/visualization/algorithms")
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert "algorithms" in data["data"]
        assert "total" in data["data"]
        assert data["data"]["total"] > 0

    def test_list_algorithms_by_category_sorting(self, test_client):
        """Test filtering algorithms by sorting category."""
        response = test_client.get("/visualization/algorithms?category=sorting")
        assert response.status_code == 200

        data = response.json()
        algorithms = data["data"]["algorithms"]
        assert all(a["category"] == "sorting" for a in algorithms)

    def test_list_algorithms_by_category_searching(self, test_client):
        """Test filtering algorithms by searching category."""
        response = test_client.get("/visualization/algorithms?category=searching")
        assert response.status_code == 200

        data = response.json()
        algorithms = data["data"]["algorithms"]
        assert all(a["category"] == "searching" for a in algorithms)

    def test_list_algorithms_by_category_graph(self, test_client):
        """Test filtering algorithms by graph category."""
        response = test_client.get("/visualization/algorithms?category=graph")
        assert response.status_code == 200

        data = response.json()
        algorithms = data["data"]["algorithms"]
        assert all(a["category"] == "graph" for a in algorithms)

    def test_get_algorithm_info_bubble_sort(self, test_client):
        """Test GET /visualization/algorithms/bubble_sort."""
        response = test_client.get("/visualization/algorithms/bubble_sort")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["id"] == "bubble_sort"
        assert data["data"]["name"] == "버블 정렬"
        assert data["data"]["time_complexity"] == "O(n²)"

    def test_get_algorithm_info_not_found(self, test_client):
        """Test GET /visualization/algorithms with unknown algorithm."""
        response = test_client.get("/visualization/algorithms/unknown_algorithm")
        assert response.status_code == 404

    def test_get_algorithm_info_no_info(self, test_client):
        """Test GET /visualization/algorithms with algorithm that has no info."""
        # dijkstra exists in AlgorithmType but not in ALGORITHM_INFO
        response = test_client.get("/visualization/algorithms/dijkstra")
        assert response.status_code == 404

    def test_generate_random_array(self, test_client):
        """Test GET /visualization/random-array."""
        response = test_client.get("/visualization/random-array")
        assert response.status_code == 200

        data = response.json()
        assert "array" in data["data"]
        assert "size" in data["data"]
        assert data["data"]["size"] == 10  # default size

    def test_generate_random_array_custom_size(self, test_client):
        """Test random array with custom size."""
        response = test_client.get("/visualization/random-array?size=5")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["size"] == 5

    def test_generate_random_array_custom_range(self, test_client):
        """Test random array with custom value range."""
        response = test_client.get("/visualization/random-array?size=10&min_val=1&max_val=10")
        assert response.status_code == 200

        data = response.json()
        arr = data["data"]["array"]
        assert all(1 <= x <= 10 for x in arr)

    def test_generate_random_array_invalid_range(self, test_client):
        """Test random array with invalid range (min > max)."""
        response = test_client.get("/visualization/random-array?min_val=100&max_val=1")
        assert response.status_code == 400

    def test_post_sorting_visualization(self, test_client):
        """Test POST /visualization/sorting."""
        response = test_client.post(
            "/visualization/sorting",
            json={
                "algorithm": "bubble_sort",
                "data": [5, 3, 8, 1],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "bubble_sort"
        assert data["data"]["final_data"] == [1, 3, 5, 8]

    def test_post_sorting_visualization_with_size(self, test_client):
        """Test POST /visualization/sorting with random data."""
        response = test_client.post(
            "/visualization/sorting",
            json={
                "algorithm": "selection_sort",
                "size": 6,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "selection_sort"
        assert len(data["data"]["initial_data"]) == 6

    def test_post_sorting_visualization_invalid_algorithm(self, test_client):
        """Test POST /visualization/sorting with invalid algorithm."""
        response = test_client.post(
            "/visualization/sorting",
            json={
                "algorithm": "bfs",  # Not a sorting algorithm
                "data": [1, 2, 3],
            },
        )
        assert response.status_code == 400

    def test_post_search_visualization(self, test_client):
        """Test POST /visualization/searching."""
        response = test_client.post(
            "/visualization/searching",
            json={
                "algorithm": "linear_search",
                "data": [5, 3, 8, 1],
                "target": 8,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "linear_search"

    def test_post_search_visualization_binary(self, test_client):
        """Test POST /visualization/searching with binary search."""
        response = test_client.post(
            "/visualization/searching",
            json={
                "algorithm": "binary_search",
                "data": [1, 3, 5, 7, 9],
                "target": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "binary_search"

    def test_post_search_visualization_invalid_algorithm(self, test_client):
        """Test POST /visualization/searching with invalid algorithm."""
        response = test_client.post(
            "/visualization/searching",
            json={
                "algorithm": "bubble_sort",  # Not a search algorithm
                "target": 5,
            },
        )
        assert response.status_code == 400

    def test_post_graph_visualization_bfs(self, test_client):
        """Test POST /visualization/graph with BFS."""
        response = test_client.post(
            "/visualization/graph",
            json={
                "algorithm": "bfs",
                "start_node": "A",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "bfs"
        assert len(data["data"]["nodes"]) == 7

    def test_post_graph_visualization_dfs(self, test_client):
        """Test POST /visualization/graph with DFS."""
        response = test_client.post(
            "/visualization/graph",
            json={
                "algorithm": "dfs",
                "start_node": "A",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "dfs"

    def test_post_graph_visualization_invalid_algorithm(self, test_client):
        """Test POST /visualization/graph with invalid algorithm."""
        response = test_client.post(
            "/visualization/graph",
            json={
                "algorithm": "bubble_sort",  # Not a graph algorithm
                "start_node": "A",
            },
        )
        assert response.status_code == 400

    def test_get_sorting_visualization_quick(self, test_client):
        """Test GET /visualization/sorting/{algorithm_id}."""
        response = test_client.get("/visualization/sorting/bubble_sort")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "bubble_sort"

    def test_get_sorting_visualization_with_size(self, test_client):
        """Test GET /visualization/sorting with custom size."""
        response = test_client.get("/visualization/sorting/quick_sort?size=8")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]["initial_data"]) == 8

    def test_get_sorting_visualization_not_found(self, test_client):
        """Test GET /visualization/sorting with unknown algorithm."""
        response = test_client.get("/visualization/sorting/unknown")
        assert response.status_code == 404

    def test_get_sorting_visualization_invalid_type(self, test_client):
        """Test GET /visualization/sorting with non-sorting algorithm."""
        response = test_client.get("/visualization/sorting/bfs")
        assert response.status_code == 400

    def test_get_search_visualization_quick(self, test_client):
        """Test GET /visualization/searching/{algorithm_id}."""
        response = test_client.get("/visualization/searching/linear_search")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "linear_search"

    def test_get_search_visualization_with_target(self, test_client):
        """Test GET /visualization/searching with target."""
        response = test_client.get("/visualization/searching/binary_search?target=5")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "binary_search"

    def test_get_search_visualization_not_found(self, test_client):
        """Test GET /visualization/searching with unknown algorithm."""
        response = test_client.get("/visualization/searching/unknown")
        assert response.status_code == 404

    def test_get_search_visualization_invalid_type(self, test_client):
        """Test GET /visualization/searching with non-search algorithm."""
        response = test_client.get("/visualization/searching/bubble_sort")
        assert response.status_code == 400

    def test_get_graph_visualization_quick(self, test_client):
        """Test GET /visualization/graph/{algorithm_id}."""
        response = test_client.get("/visualization/graph/bfs")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "bfs"

    def test_get_graph_visualization_with_start(self, test_client):
        """Test GET /visualization/graph with custom start node."""
        response = test_client.get("/visualization/graph/dfs?start_node=B")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["algorithm_type"] == "dfs"

    def test_get_graph_visualization_not_found(self, test_client):
        """Test GET /visualization/graph with unknown algorithm."""
        response = test_client.get("/visualization/graph/unknown")
        assert response.status_code == 404

    def test_get_graph_visualization_invalid_type(self, test_client):
        """Test GET /visualization/graph with non-graph algorithm."""
        response = test_client.get("/visualization/graph/bubble_sort")
        assert response.status_code == 400
