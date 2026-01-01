"""Visualization API routes."""

from fastapi import APIRouter, HTTPException, Query

from code_tutor.shared.api_response import success_response
from code_tutor.visualization.application import (
    GenerateGraphRequest,
    GenerateSearchRequest,
    GenerateSortingRequest,
    VisualizationService,
)
from code_tutor.visualization.domain import AlgorithmCategory, AlgorithmType

router = APIRouter(prefix="/visualization", tags=["Visualization"])

# Dependency
visualization_service = VisualizationService()


@router.get("/algorithms")
async def list_algorithms(
    category: AlgorithmCategory | None = None,
):
    """Get list of available algorithms."""
    if category:
        algorithms = visualization_service.get_algorithms_by_category(category)
    else:
        algorithms = visualization_service.get_available_algorithms()

    return success_response({
        "algorithms": algorithms,
        "total": len(algorithms),
    })


@router.get("/algorithms/{algorithm_id}")
async def get_algorithm_info(algorithm_id: str):
    """Get information about a specific algorithm."""
    try:
        algo_type = AlgorithmType(algorithm_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    from code_tutor.visualization.domain import ALGORITHM_INFO

    if algo_type not in ALGORITHM_INFO:
        raise HTTPException(status_code=404, detail="Algorithm info not available")

    info = ALGORITHM_INFO[algo_type]
    return success_response({
        "id": algo_type.value,
        "name": info["name"],
        "name_en": info["name_en"],
        "category": info["category"].value,
        "time_complexity": info["time_complexity"],
        "space_complexity": info["space_complexity"],
        "description": info["description"],
    })


@router.get("/random-array")
async def generate_random_array(
    size: int = Query(default=10, ge=3, le=50),
    min_val: int = Query(default=1, ge=0, le=100),
    max_val: int = Query(default=100, ge=1, le=1000),
):
    """Generate a random array for visualization."""
    if min_val > max_val:
        raise HTTPException(status_code=400, detail="min_val must be <= max_val")

    arr = visualization_service.generate_random_array(size, min_val, max_val)
    return success_response({
        "array": arr,
        "size": len(arr),
    })


@router.post("/sorting")
async def generate_sorting_visualization(request: GenerateSortingRequest):
    """Generate visualization for a sorting algorithm."""
    try:
        viz = visualization_service.generate_sorting_visualization(
            algorithm=request.algorithm,
            data=request.data,
            size=request.size,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/searching")
async def generate_search_visualization(request: GenerateSearchRequest):
    """Generate visualization for a search algorithm."""
    try:
        viz = visualization_service.generate_search_visualization(
            algorithm=request.algorithm,
            data=request.data,
            target=request.target,
            size=request.size,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/graph")
async def generate_graph_visualization(request: GenerateGraphRequest):
    """Generate visualization for a graph algorithm."""
    try:
        viz = visualization_service.generate_graph_visualization(
            algorithm=request.algorithm,
            start_node=request.start_node,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sorting/{algorithm_id}")
async def get_sorting_visualization(
    algorithm_id: str,
    size: int = Query(default=10, ge=3, le=50),
):
    """Quick endpoint to get sorting visualization with random data."""
    try:
        algo_type = AlgorithmType(algorithm_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    try:
        viz = visualization_service.generate_sorting_visualization(
            algorithm=algo_type,
            size=size,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/searching/{algorithm_id}")
async def get_search_visualization(
    algorithm_id: str,
    size: int = Query(default=10, ge=3, le=50),
    target: int | None = None,
):
    """Quick endpoint to get search visualization with random data."""
    try:
        algo_type = AlgorithmType(algorithm_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    try:
        viz = visualization_service.generate_search_visualization(
            algorithm=algo_type,
            target=target,
            size=size,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/graph/{algorithm_id}")
async def get_graph_visualization(
    algorithm_id: str,
    start_node: str = "A",
):
    """Quick endpoint to get graph visualization."""
    try:
        algo_type = AlgorithmType(algorithm_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    try:
        viz = visualization_service.generate_graph_visualization(
            algorithm=algo_type,
            start_node=start_node,
        )
        return success_response(viz.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
