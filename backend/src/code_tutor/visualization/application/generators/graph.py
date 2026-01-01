"""Graph algorithm visualization generators."""

import math
from collections import deque

from code_tutor.visualization.domain import (
    AlgorithmType,
    ElementState,
    GraphEdge,
    GraphNode,
    GraphVisualization,
)


def create_sample_graph() -> tuple[list[GraphNode], list[GraphEdge], dict[str, list[str]]]:
    """Create a sample graph for visualization."""
    # Create nodes in a circle layout
    nodes = []
    num_nodes = 7
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    center_x, center_y = 200, 200
    radius = 150

    for i in range(num_nodes):
        angle = (2 * math.pi * i / num_nodes) - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        nodes.append(GraphNode(
            id=labels[i],
            value=labels[i],
            x=x,
            y=y,
        ))

    # Create edges (adjacency list)
    adjacency = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F", "G"],
        "F": ["C", "E"],
        "G": ["E"],
    }

    edges = []
    added_edges = set()
    for source, targets in adjacency.items():
        for target in targets:
            edge_key = tuple(sorted([source, target]))
            if edge_key not in added_edges:
                edges.append(GraphEdge(source=source, target=target))
                added_edges.add(edge_key)

    return nodes, edges, adjacency


def generate_bfs(start_node: str = "A") -> GraphVisualization:
    """Generate BFS visualization steps."""
    nodes, edges, adjacency = create_sample_graph()

    viz = GraphVisualization(
        algorithm_type=AlgorithmType.BFS,
        nodes=nodes,
        edges=edges,
        code="""def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return result""",
    )

    node_map = {n.id: n for n in nodes}
    visited = set()
    queue = deque([start_node])
    step_num = 0

    def get_node_states():
        states = {}
        for node in nodes:
            if node.id in visited:
                states[node.id] = ElementState.VISITED
            else:
                states[node.id] = ElementState.DEFAULT
        return states

    def get_edge_states(current_node=None, exploring_edge=None):
        states = {}
        for edge in edges:
            if exploring_edge and (edge.source, edge.target) == exploring_edge:
                states[(edge.source, edge.target)] = ElementState.ACTIVE
            elif edge.source in visited and edge.target in visited:
                states[(edge.source, edge.target)] = ElementState.VISITED
            else:
                states[(edge.source, edge.target)] = ElementState.DEFAULT
        return states

    # Initial state
    viz.steps.append({
        "step_number": step_num,
        "action": "init",
        "current_node": None,
        "queue": list(queue),
        "visited": list(visited),
        "node_states": {n.id: ElementState.DEFAULT.value for n in nodes},
        "edge_states": {f"{e.source}-{e.target}": ElementState.DEFAULT.value for e in edges},
        "description": f"BFS를 노드 {start_node}에서 시작합니다.",
    })
    step_num += 1

    while queue:
        current = queue.popleft()

        if current in visited:
            continue

        # Visit current node
        visited.add(current)
        node_map[current].state = ElementState.CURRENT

        node_states = get_node_states()
        node_states[current] = ElementState.CURRENT

        viz.steps.append({
            "step_number": step_num,
            "action": "visit",
            "current_node": current,
            "queue": list(queue),
            "visited": list(visited),
            "node_states": {k: v.value if isinstance(v, ElementState) else v for k, v in node_states.items()},
            "edge_states": {f"{e.source}-{e.target}": ElementState.DEFAULT.value for e in edges},
            "description": f"노드 {current}를 방문합니다. 큐: {list(queue)}",
        })
        step_num += 1

        # Explore neighbors
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                queue.append(neighbor)

                node_states = get_node_states()
                node_states[current] = ElementState.CURRENT
                node_states[neighbor] = ElementState.ACTIVE

                viz.steps.append({
                    "step_number": step_num,
                    "action": "explore",
                    "current_node": current,
                    "exploring": neighbor,
                    "queue": list(queue),
                    "visited": list(visited),
                    "node_states": {k: v.value if isinstance(v, ElementState) else v for k, v in node_states.items()},
                    "edge_states": {f"{e.source}-{e.target}": ElementState.ACTIVE.value if (e.source == current and e.target == neighbor) or (e.target == current and e.source == neighbor) else ElementState.DEFAULT.value for e in edges},
                    "description": f"이웃 노드 {neighbor}를 큐에 추가합니다. 큐: {list(queue)}",
                })
                step_num += 1

        node_map[current].state = ElementState.VISITED

    # Final state
    viz.steps.append({
        "step_number": step_num,
        "action": "complete",
        "current_node": None,
        "queue": [],
        "visited": list(visited),
        "node_states": {n.id: ElementState.VISITED.value for n in nodes},
        "edge_states": {f"{e.source}-{e.target}": ElementState.VISITED.value for e in edges},
        "description": f"BFS 완료! 방문 순서: {list(visited)}",
    })

    return viz


def generate_dfs(start_node: str = "A") -> GraphVisualization:
    """Generate DFS visualization steps."""
    nodes, edges, adjacency = create_sample_graph()

    viz = GraphVisualization(
        algorithm_type=AlgorithmType.DFS,
        nodes=nodes,
        edges=edges,
        code="""def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))

    return result""",
    )

    node_map = {n.id: n for n in nodes}
    visited = set()
    visit_order = []
    step_num = [0]  # Use list to allow modification in nested function

    def get_node_states(current=None, exploring=None):
        states = {}
        for node in nodes:
            if node.id == current:
                states[node.id] = ElementState.CURRENT.value
            elif node.id == exploring:
                states[node.id] = ElementState.ACTIVE.value
            elif node.id in visited:
                states[node.id] = ElementState.VISITED.value
            else:
                states[node.id] = ElementState.DEFAULT.value
        return states

    # Initial state
    viz.steps.append({
        "step_number": step_num[0],
        "action": "init",
        "current_node": None,
        "stack": [start_node],
        "visited": [],
        "node_states": {n.id: ElementState.DEFAULT.value for n in nodes},
        "edge_states": {f"{e.source}-{e.target}": ElementState.DEFAULT.value for e in edges},
        "description": f"DFS를 노드 {start_node}에서 시작합니다.",
    })
    step_num[0] += 1

    def dfs_recursive(current, stack_trace):
        visited.add(current)
        visit_order.append(current)
        node_map[current].state = ElementState.CURRENT

        # Visit step
        viz.steps.append({
            "step_number": step_num[0],
            "action": "visit",
            "current_node": current,
            "stack": stack_trace + [current],
            "visited": list(visited),
            "node_states": get_node_states(current=current),
            "edge_states": {f"{e.source}-{e.target}": ElementState.DEFAULT.value for e in edges},
            "description": f"노드 {current}를 방문합니다. 스택: {stack_trace + [current]}",
        })
        step_num[0] += 1

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                # Explore step
                viz.steps.append({
                    "step_number": step_num[0],
                    "action": "explore",
                    "current_node": current,
                    "exploring": neighbor,
                    "stack": stack_trace + [current],
                    "visited": list(visited),
                    "node_states": get_node_states(current=current, exploring=neighbor),
                    "edge_states": {f"{e.source}-{e.target}": ElementState.ACTIVE.value if (e.source == current and e.target == neighbor) or (e.target == current and e.source == neighbor) else ElementState.DEFAULT.value for e in edges},
                    "description": f"노드 {current}에서 이웃 {neighbor}로 탐색합니다.",
                })
                step_num[0] += 1

                dfs_recursive(neighbor, stack_trace + [current])

        # Backtrack step
        node_map[current].state = ElementState.VISITED
        if stack_trace:
            viz.steps.append({
                "step_number": step_num[0],
                "action": "backtrack",
                "current_node": current,
                "back_to": stack_trace[-1] if stack_trace else None,
                "stack": stack_trace,
                "visited": list(visited),
                "node_states": get_node_states(),
                "edge_states": {f"{e.source}-{e.target}": ElementState.DEFAULT.value for e in edges},
                "description": f"노드 {current}의 모든 이웃을 탐색했습니다. 백트래킹합니다.",
            })
            step_num[0] += 1

    dfs_recursive(start_node, [])

    # Final state
    viz.steps.append({
        "step_number": step_num[0],
        "action": "complete",
        "current_node": None,
        "stack": [],
        "visited": visit_order,
        "node_states": {n.id: ElementState.VISITED.value for n in nodes},
        "edge_states": {f"{e.source}-{e.target}": ElementState.VISITED.value for e in edges},
        "description": f"DFS 완료! 방문 순서: {visit_order}",
    })

    return viz
