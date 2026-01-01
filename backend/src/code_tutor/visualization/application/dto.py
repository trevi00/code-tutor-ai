"""Visualization DTOs."""

from pydantic import BaseModel, Field

from code_tutor.visualization.domain import AlgorithmCategory, AlgorithmType


class AlgorithmInfoResponse(BaseModel):
    """Algorithm information response."""

    id: str
    name: str
    name_en: str
    category: str
    time_complexity: str
    space_complexity: str
    description: str


class AlgorithmListResponse(BaseModel):
    """List of algorithms response."""

    algorithms: list[AlgorithmInfoResponse]
    total: int


class GenerateSortingRequest(BaseModel):
    """Request to generate sorting visualization."""

    algorithm: AlgorithmType
    data: list[int] | None = None
    size: int = Field(default=10, ge=3, le=50)


class GenerateSearchRequest(BaseModel):
    """Request to generate search visualization."""

    algorithm: AlgorithmType
    data: list[int] | None = None
    target: int | None = None
    size: int = Field(default=10, ge=3, le=50)


class GenerateGraphRequest(BaseModel):
    """Request to generate graph visualization."""

    algorithm: AlgorithmType
    start_node: str = "A"


class VisualizationStepResponse(BaseModel):
    """Single visualization step."""

    step_number: int
    action: str
    indices: list[int]
    values: list
    array_state: list
    element_states: list[str]
    description: str
    code_line: int | None = None
    auxiliary_data: dict = Field(default_factory=dict)


class SortingVisualizationResponse(BaseModel):
    """Sorting/Search visualization response."""

    algorithm_type: str
    category: str
    initial_data: list
    steps: list[VisualizationStepResponse]
    final_data: list
    code: str
    total_steps: int
    total_comparisons: int
    total_swaps: int


class GraphNodeResponse(BaseModel):
    """Graph node response."""

    id: str
    value: str
    x: float
    y: float
    state: str


class GraphEdgeResponse(BaseModel):
    """Graph edge response."""

    source: str
    target: str
    weight: float
    state: str


class GraphVisualizationResponse(BaseModel):
    """Graph visualization response."""

    algorithm_type: str
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    steps: list[dict]
    code: str
    total_steps: int
