"""Visualization domain entities."""

from dataclasses import dataclass, field
from typing import Any

from code_tutor.visualization.domain.value_objects import (
    AlgorithmCategory,
    AlgorithmType,
    AnimationType,
    ElementState,
)


@dataclass
class VisualizationStep:
    """A single step in algorithm visualization."""

    step_number: int
    action: AnimationType
    indices: list[int]  # Indices involved in this step
    values: list[Any]  # Values at those indices
    array_state: list[Any]  # Full array state after this step
    element_states: list[ElementState]  # Visual state of each element
    description: str  # Korean description of what happened
    code_line: int | None = None  # Corresponding line in code
    auxiliary_data: dict[str, Any] = field(
        default_factory=dict
    )  # Extra data (e.g., pointers, ranges)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action.value,
            "indices": self.indices,
            "values": self.values,
            "array_state": self.array_state,
            "element_states": [s.value for s in self.element_states],
            "description": self.description,
            "code_line": self.code_line,
            "auxiliary_data": self.auxiliary_data,
        }


@dataclass
class TreeNode:
    """Node for tree visualization."""

    id: str
    value: Any
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    state: ElementState = ElementState.DEFAULT
    x: float = 0
    y: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "value": self.value,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "state": self.state.value,
            "x": self.x,
            "y": self.y,
        }


@dataclass
class GraphNode:
    """Node for graph visualization."""

    id: str
    value: Any
    x: float
    y: float
    state: ElementState = ElementState.DEFAULT

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "value": self.value,
            "x": self.x,
            "y": self.y,
            "state": self.state.value,
        }


@dataclass
class GraphEdge:
    """Edge for graph visualization."""

    source: str
    target: str
    weight: float = 1.0
    state: ElementState = ElementState.DEFAULT

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "state": self.state.value,
        }


@dataclass
class Visualization:
    """Complete visualization for an algorithm."""

    algorithm_type: AlgorithmType
    category: AlgorithmCategory
    initial_data: list[Any]
    steps: list[VisualizationStep] = field(default_factory=list)
    final_data: list[Any] = field(default_factory=list)
    code: str = ""
    total_comparisons: int = 0
    total_swaps: int = 0

    def add_step(self, step: VisualizationStep) -> None:
        """Add a step to visualization."""
        self.steps.append(step)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "algorithm_type": self.algorithm_type.value,
            "category": self.category.value,
            "initial_data": self.initial_data,
            "steps": [step.to_dict() for step in self.steps],
            "final_data": self.final_data,
            "code": self.code,
            "total_steps": len(self.steps),
            "total_comparisons": self.total_comparisons,
            "total_swaps": self.total_swaps,
        }


@dataclass
class TreeVisualization:
    """Visualization for tree-based algorithms."""

    algorithm_type: AlgorithmType
    root: TreeNode | None = None
    steps: list[dict] = field(default_factory=list)
    code: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "algorithm_type": self.algorithm_type.value,
            "root": self.root.to_dict() if self.root else None,
            "steps": self.steps,
            "code": self.code,
            "total_steps": len(self.steps),
        }


@dataclass
class GraphVisualization:
    """Visualization for graph algorithms."""

    algorithm_type: AlgorithmType
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    code: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "algorithm_type": self.algorithm_type.value,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "steps": self.steps,
            "code": self.code,
            "total_steps": len(self.steps),
        }
