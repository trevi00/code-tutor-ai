"""Search algorithm visualization generators."""

from code_tutor.visualization.domain import (
    AlgorithmCategory,
    AlgorithmType,
    AnimationType,
    ElementState,
    Visualization,
    VisualizationStep,
)


def generate_linear_search(data: list[int], target: int) -> Visualization:
    """Generate linear search visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.LINEAR_SEARCH,
        category=AlgorithmCategory.SEARCHING,
        initial_data=data.copy(),
        code="""def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Found
    return -1  # Not found""",
    )

    step_num = 0
    comparisons = 0
    found_idx = -1

    # Initial state
    viz.add_step(
        VisualizationStep(
            step_number=step_num,
            action=AnimationType.HIGHLIGHT,
            indices=list(range(n)),
            values=arr.copy(),
            array_state=arr.copy(),
            element_states=[ElementState.DEFAULT] * n,
            description=f"배열에서 {target}을 찾습니다.",
            code_line=1,
            auxiliary_data={"target": target},
        )
    )
    step_num += 1

    for i in range(n):
        # Compare step
        states = [
            ElementState.VISITED if idx < i else ElementState.DEFAULT
            for idx in range(n)
        ]
        states[i] = ElementState.CURRENT

        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.COMPARE,
                indices=[i],
                values=[arr[i]],
                array_state=arr.copy(),
                element_states=states,
                description=f"위치 {i}: {arr[i]}와 {target}을 비교합니다.",
                code_line=3,
                auxiliary_data={"i": i, "target": target, "current": arr[i]},
            )
        )
        step_num += 1
        comparisons += 1

        if arr[i] == target:
            found_idx = i
            states = [
                ElementState.VISITED if idx < i else ElementState.DEFAULT
                for idx in range(n)
            ]
            states[i] = ElementState.FOUND

            viz.add_step(
                VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.FOUND,
                    indices=[i],
                    values=[arr[i]],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"찾았습니다! {target}은 위치 {i}에 있습니다. (비교 횟수: {comparisons})",
                    code_line=4,
                    auxiliary_data={"found_index": i, "comparisons": comparisons},
                )
            )
            step_num += 1
            break

        # Not found at this position
        states = [
            ElementState.VISITED if idx <= i else ElementState.DEFAULT
            for idx in range(n)
        ]

        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.UNHIGHLIGHT,
                indices=[i],
                values=[arr[i]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[i]}은 {target}이 아닙니다. 다음으로 이동합니다.",
                code_line=2,
                auxiliary_data={"i": i},
            )
        )
        step_num += 1

    if found_idx == -1:
        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.NOT_FOUND,
                indices=[],
                values=[],
                array_state=arr.copy(),
                element_states=[ElementState.VISITED] * n,
                description=f"{target}을 찾지 못했습니다. (비교 횟수: {comparisons})",
                code_line=5,
                auxiliary_data={"comparisons": comparisons},
            )
        )

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons

    return viz


def generate_binary_search(data: list[int], target: int) -> Visualization:
    """Generate binary search visualization steps."""
    arr = sorted(data.copy())  # Binary search requires sorted array
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.BINARY_SEARCH,
        category=AlgorithmCategory.SEARCHING,
        initial_data=arr.copy(),
        code="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # Found
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # Not found""",
    )

    step_num = 0
    comparisons = 0
    found_idx = -1

    left, right = 0, n - 1

    # Initial state
    viz.add_step(
        VisualizationStep(
            step_number=step_num,
            action=AnimationType.HIGHLIGHT,
            indices=list(range(n)),
            values=arr.copy(),
            array_state=arr.copy(),
            element_states=[ElementState.ACTIVE] * n,
            description=f"정렬된 배열에서 {target}을 이진 탐색합니다.",
            code_line=1,
            auxiliary_data={"target": target, "left": left, "right": right},
        )
    )
    step_num += 1

    while left <= right:
        mid = (left + right) // 2

        # Show search range
        states = [ElementState.DEFAULT] * n
        for i in range(left, right + 1):
            states[i] = ElementState.ACTIVE
        states[mid] = ElementState.CURRENT

        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.HIGHLIGHT,
                indices=[left, mid, right],
                values=[arr[left], arr[mid], arr[right]],
                array_state=arr.copy(),
                element_states=states,
                description=f"탐색 범위: [{left}:{right}], 중간값: arr[{mid}] = {arr[mid]}",
                code_line=4,
                auxiliary_data={"left": left, "mid": mid, "right": right},
            )
        )
        step_num += 1

        # Compare
        states = [ElementState.DEFAULT] * n
        for i in range(left, right + 1):
            states[i] = ElementState.ACTIVE
        states[mid] = ElementState.COMPARING

        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.COMPARE,
                indices=[mid],
                values=[arr[mid]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[mid]}와 {target}을 비교합니다.",
                code_line=5,
                auxiliary_data={"mid": mid, "value": arr[mid], "target": target},
            )
        )
        step_num += 1
        comparisons += 1

        if arr[mid] == target:
            found_idx = mid
            states = [ElementState.DEFAULT] * n
            states[mid] = ElementState.FOUND

            viz.add_step(
                VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.FOUND,
                    indices=[mid],
                    values=[arr[mid]],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"찾았습니다! {target}은 위치 {mid}에 있습니다. (비교 횟수: {comparisons})",
                    code_line=6,
                    auxiliary_data={"found_index": mid, "comparisons": comparisons},
                )
            )
            step_num += 1
            break

        elif arr[mid] < target:
            # Move right
            old_left = left
            left = mid + 1

            states = [
                ElementState.VISITED
                if i < left
                else ElementState.ACTIVE
                if i <= right
                else ElementState.DEFAULT
                for i in range(n)
            ]

            viz.add_step(
                VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.UNHIGHLIGHT,
                    indices=list(range(old_left, left)),
                    values=[arr[i] for i in range(old_left, left)],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"{arr[mid]}이 {target}보다 작으므로 오른쪽 반을 탐색합니다. (left = {left})",
                    code_line=8,
                    auxiliary_data={"left": left, "right": right},
                )
            )
            step_num += 1

        else:
            # Move left
            old_right = right
            right = mid - 1

            states = [
                ElementState.ACTIVE
                if left <= i <= right
                else ElementState.VISITED
                if i > right
                else ElementState.DEFAULT
                for i in range(n)
            ]

            viz.add_step(
                VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.UNHIGHLIGHT,
                    indices=list(range(right + 1, old_right + 1)),
                    values=[arr[i] for i in range(right + 1, old_right + 1)],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"{arr[mid]}이 {target}보다 크므로 왼쪽 반을 탐색합니다. (right = {right})",
                    code_line=10,
                    auxiliary_data={"left": left, "right": right},
                )
            )
            step_num += 1

    if found_idx == -1:
        viz.add_step(
            VisualizationStep(
                step_number=step_num,
                action=AnimationType.NOT_FOUND,
                indices=[],
                values=[],
                array_state=arr.copy(),
                element_states=[ElementState.VISITED] * n,
                description=f"{target}을 찾지 못했습니다. (비교 횟수: {comparisons})",
                code_line=11,
                auxiliary_data={"comparisons": comparisons},
            )
        )

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons

    return viz
