"""Sorting algorithm visualization generators."""

from code_tutor.visualization.domain import (
    AlgorithmCategory,
    AlgorithmType,
    AnimationType,
    ElementState,
    Visualization,
    VisualizationStep,
)


def generate_bubble_sort(data: list[int]) -> Visualization:
    """Generate bubble sort visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.BUBBLE_SORT,
        category=AlgorithmCategory.SORTING,
        initial_data=data.copy(),
        code="""def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr""",
    )

    step_num = 0
    comparisons = 0
    swaps = 0

    # Initial state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.HIGHLIGHT,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.DEFAULT] * n,
        description="정렬을 시작합니다.",
        code_line=1,
    ))
    step_num += 1

    for i in range(n):
        for j in range(0, n - i - 1):
            # Compare step
            states = [ElementState.SORTED if idx >= n - i else ElementState.DEFAULT for idx in range(n)]
            states[j] = ElementState.COMPARING
            states[j + 1] = ElementState.COMPARING

            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.COMPARE,
                indices=[j, j + 1],
                values=[arr[j], arr[j + 1]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[j]}와 {arr[j+1]}을 비교합니다.",
                code_line=4,
                auxiliary_data={"i": i, "j": j},
            ))
            step_num += 1
            comparisons += 1

            if arr[j] > arr[j + 1]:
                # Swap step
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1

                states = [ElementState.SORTED if idx >= n - i else ElementState.DEFAULT for idx in range(n)]
                states[j] = ElementState.SWAPPING
                states[j + 1] = ElementState.SWAPPING

                viz.add_step(VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.SWAP,
                    indices=[j, j + 1],
                    values=[arr[j], arr[j + 1]],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"{arr[j+1]}이 {arr[j]}보다 크므로 교환합니다.",
                    code_line=5,
                    auxiliary_data={"i": i, "j": j},
                ))
                step_num += 1

        # Mark as sorted
        if n - i - 1 >= 0:
            states = [ElementState.SORTED if idx >= n - i - 1 else ElementState.DEFAULT for idx in range(n)]
            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.SORTED,
                indices=[n - i - 1],
                values=[arr[n - i - 1]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[n-i-1]}이 올바른 위치에 정렬되었습니다.",
                code_line=3,
            ))
            step_num += 1

    # Final state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.SORTED,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.SORTED] * n,
        description=f"정렬 완료! 비교: {comparisons}회, 교환: {swaps}회",
        code_line=6,
    ))

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons
    viz.total_swaps = swaps

    return viz


def generate_selection_sort(data: list[int]) -> Visualization:
    """Generate selection sort visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.SELECTION_SORT,
        category=AlgorithmCategory.SORTING,
        initial_data=data.copy(),
        code="""def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr""",
    )

    step_num = 0
    comparisons = 0
    swaps = 0

    # Initial state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.HIGHLIGHT,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.DEFAULT] * n,
        description="정렬을 시작합니다.",
        code_line=1,
    ))
    step_num += 1

    for i in range(n):
        min_idx = i

        # Mark current minimum
        states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
        states[i] = ElementState.PIVOT

        viz.add_step(VisualizationStep(
            step_number=step_num,
            action=AnimationType.HIGHLIGHT,
            indices=[i],
            values=[arr[i]],
            array_state=arr.copy(),
            element_states=states,
            description=f"위치 {i}부터 최소값을 찾습니다. 현재 최소값: {arr[i]}",
            code_line=4,
            auxiliary_data={"i": i, "min_idx": min_idx},
        ))
        step_num += 1

        for j in range(i + 1, n):
            # Compare step
            states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
            states[min_idx] = ElementState.PIVOT
            states[j] = ElementState.COMPARING

            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.COMPARE,
                indices=[min_idx, j],
                values=[arr[min_idx], arr[j]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[j]}와 현재 최소값 {arr[min_idx]}을 비교합니다.",
                code_line=5,
                auxiliary_data={"i": i, "j": j, "min_idx": min_idx},
            ))
            step_num += 1
            comparisons += 1

            if arr[j] < arr[min_idx]:
                min_idx = j
                states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
                states[min_idx] = ElementState.PIVOT

                viz.add_step(VisualizationStep(
                    step_number=step_num,
                    action=AnimationType.HIGHLIGHT,
                    indices=[min_idx],
                    values=[arr[min_idx]],
                    array_state=arr.copy(),
                    element_states=states,
                    description=f"새로운 최소값 발견: {arr[min_idx]}",
                    code_line=7,
                    auxiliary_data={"i": i, "min_idx": min_idx},
                ))
                step_num += 1

        # Swap if needed
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1

            states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
            states[i] = ElementState.SWAPPING
            states[min_idx] = ElementState.SWAPPING

            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.SWAP,
                indices=[i, min_idx],
                values=[arr[i], arr[min_idx]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[i]}와 {arr[min_idx]}을 교환합니다.",
                code_line=8,
            ))
            step_num += 1

        # Mark as sorted
        states = [ElementState.SORTED if idx <= i else ElementState.DEFAULT for idx in range(n)]
        viz.add_step(VisualizationStep(
            step_number=step_num,
            action=AnimationType.SORTED,
            indices=[i],
            values=[arr[i]],
            array_state=arr.copy(),
            element_states=states,
            description=f"{arr[i]}이 올바른 위치에 정렬되었습니다.",
            code_line=3,
        ))
        step_num += 1

    # Final state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.SORTED,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.SORTED] * n,
        description=f"정렬 완료! 비교: {comparisons}회, 교환: {swaps}회",
        code_line=9,
    ))

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons
    viz.total_swaps = swaps

    return viz


def generate_insertion_sort(data: list[int]) -> Visualization:
    """Generate insertion sort visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.INSERTION_SORT,
        category=AlgorithmCategory.SORTING,
        initial_data=data.copy(),
        code="""def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr""",
    )

    step_num = 0
    comparisons = 0
    swaps = 0

    # Initial state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.HIGHLIGHT,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.SORTED] + [ElementState.DEFAULT] * (n - 1),
        description="첫 번째 원소는 이미 정렬되어 있습니다.",
        code_line=1,
    ))
    step_num += 1

    for i in range(1, n):
        key = arr[i]

        # Pick current element
        states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
        states[i] = ElementState.CURRENT

        viz.add_step(VisualizationStep(
            step_number=step_num,
            action=AnimationType.HIGHLIGHT,
            indices=[i],
            values=[key],
            array_state=arr.copy(),
            element_states=states,
            description=f"{key}를 정렬된 부분에 삽입할 위치를 찾습니다.",
            code_line=3,
            auxiliary_data={"key": key, "i": i},
        ))
        step_num += 1

        j = i - 1
        while j >= 0 and arr[j] > key:
            # Compare step
            states = [ElementState.SORTED if idx < i else ElementState.DEFAULT for idx in range(n)]
            states[j] = ElementState.COMPARING
            states[i] = ElementState.CURRENT

            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.COMPARE,
                indices=[j, i],
                values=[arr[j], key],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[j]}가 {key}보다 크므로 오른쪽으로 이동합니다.",
                code_line=5,
                auxiliary_data={"key": key, "j": j},
            ))
            step_num += 1
            comparisons += 1

            # Shift
            arr[j + 1] = arr[j]
            swaps += 1

            viz.add_step(VisualizationStep(
                step_number=step_num,
                action=AnimationType.SET,
                indices=[j + 1],
                values=[arr[j + 1]],
                array_state=arr.copy(),
                element_states=states,
                description=f"{arr[j]}를 오른쪽으로 이동합니다.",
                code_line=6,
                auxiliary_data={"key": key, "j": j},
            ))
            step_num += 1

            j -= 1

        # Insert key
        arr[j + 1] = key

        states = [ElementState.SORTED if idx <= i else ElementState.DEFAULT for idx in range(n)]
        states[j + 1] = ElementState.ACTIVE

        viz.add_step(VisualizationStep(
            step_number=step_num,
            action=AnimationType.SET,
            indices=[j + 1],
            values=[key],
            array_state=arr.copy(),
            element_states=states,
            description=f"{key}를 위치 {j+1}에 삽입합니다.",
            code_line=8,
            auxiliary_data={"key": key, "position": j + 1},
        ))
        step_num += 1

    # Final state
    viz.add_step(VisualizationStep(
        step_number=step_num,
        action=AnimationType.SORTED,
        indices=list(range(n)),
        values=arr.copy(),
        array_state=arr.copy(),
        element_states=[ElementState.SORTED] * n,
        description=f"정렬 완료! 비교: {comparisons}회, 이동: {swaps}회",
        code_line=9,
    ))

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons
    viz.total_swaps = swaps

    return viz


def generate_quick_sort(data: list[int]) -> Visualization:
    """Generate quick sort visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.QUICK_SORT,
        category=AlgorithmCategory.SORTING,
        initial_data=data.copy(),
        code="""def quick_sort(arr, low, high):
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1""",
    )

    step_num = [0]  # Use list to allow modification in nested function
    comparisons = [0]
    swaps = [0]
    sorted_indices = set()

    def add_step(action, indices, values, states, description, code_line, aux=None):
        viz.add_step(VisualizationStep(
            step_number=step_num[0],
            action=action,
            indices=indices,
            values=values,
            array_state=arr.copy(),
            element_states=states,
            description=description,
            code_line=code_line,
            auxiliary_data=aux or {},
        ))
        step_num[0] += 1

    def get_states(low, high, pivot_idx=None, comparing=None):
        states = []
        for idx in range(n):
            if idx in sorted_indices:
                states.append(ElementState.SORTED)
            elif pivot_idx is not None and idx == pivot_idx:
                states.append(ElementState.PIVOT)
            elif comparing and idx in comparing:
                states.append(ElementState.COMPARING)
            elif low <= idx <= high:
                states.append(ElementState.ACTIVE)
            else:
                states.append(ElementState.DEFAULT)
        return states

    # Initial state
    add_step(
        AnimationType.HIGHLIGHT,
        list(range(n)),
        arr.copy(),
        [ElementState.DEFAULT] * n,
        "퀵 정렬을 시작합니다.",
        1,
    )

    def partition(low, high):
        pivot = arr[high]
        pivot_idx = high

        states = get_states(low, high, pivot_idx)
        add_step(
            AnimationType.PIVOT,
            [high],
            [pivot],
            states,
            f"피벗을 {pivot}로 선택합니다.",
            8,
            {"low": low, "high": high, "pivot": pivot},
        )

        i = low - 1

        for j in range(low, high):
            # Compare
            states = get_states(low, high, pivot_idx, [j])
            add_step(
                AnimationType.COMPARE,
                [j, pivot_idx],
                [arr[j], pivot],
                states,
                f"{arr[j]}와 피벗 {pivot}을 비교합니다.",
                10,
                {"i": i, "j": j},
            )
            comparisons[0] += 1

            if arr[j] <= pivot:
                i += 1
                if i != j:
                    arr[i], arr[j] = arr[j], arr[i]
                    swaps[0] += 1

                    states = get_states(low, high, pivot_idx)
                    states[i] = ElementState.SWAPPING
                    states[j] = ElementState.SWAPPING
                    add_step(
                        AnimationType.SWAP,
                        [i, j],
                        [arr[i], arr[j]],
                        states,
                        f"{arr[j]}이 피벗보다 작으므로 {arr[i]}와 교환합니다.",
                        13,
                        {"i": i, "j": j},
                    )

        # Move pivot to correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        swaps[0] += 1

        states = get_states(low, high)
        states[i + 1] = ElementState.SORTED
        sorted_indices.add(i + 1)

        add_step(
            AnimationType.SWAP,
            [i + 1, high],
            [arr[i + 1], arr[high]],
            states,
            f"피벗 {arr[i+1]}을 올바른 위치에 배치합니다.",
            14,
            {"pivot_position": i + 1},
        )

        return i + 1

    def quicksort(low, high):
        if low < high:
            pi = partition(low, high)
            quicksort(low, pi - 1)
            quicksort(pi + 1, high)
        elif low == high:
            sorted_indices.add(low)
            states = get_states(0, n - 1)
            add_step(
                AnimationType.SORTED,
                [low],
                [arr[low]],
                states,
                f"{arr[low]}은 정렬된 위치입니다.",
                2,
            )

    quicksort(0, n - 1)

    # Final state
    add_step(
        AnimationType.SORTED,
        list(range(n)),
        arr.copy(),
        [ElementState.SORTED] * n,
        f"정렬 완료! 비교: {comparisons[0]}회, 교환: {swaps[0]}회",
        5,
    )

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons[0]
    viz.total_swaps = swaps[0]

    return viz


def generate_merge_sort(data: list[int]) -> Visualization:
    """Generate merge sort visualization steps."""
    arr = data.copy()
    n = len(arr)
    viz = Visualization(
        algorithm_type=AlgorithmType.MERGE_SORT,
        category=AlgorithmCategory.SORTING,
        initial_data=data.copy(),
        code="""def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        merge(arr, left, right)

def merge(arr, left, right):
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1""",
    )

    step_num = [0]
    comparisons = [0]

    def add_step(action, indices, values, states, description, code_line, aux=None):
        viz.add_step(VisualizationStep(
            step_number=step_num[0],
            action=action,
            indices=indices,
            values=values,
            array_state=arr.copy(),
            element_states=states,
            description=description,
            code_line=code_line,
            auxiliary_data=aux or {},
        ))
        step_num[0] += 1

    # Initial state
    add_step(
        AnimationType.HIGHLIGHT,
        list(range(n)),
        arr.copy(),
        [ElementState.DEFAULT] * n,
        "병합 정렬을 시작합니다.",
        1,
    )

    def merge_sort_helper(left_idx, right_idx):
        if left_idx >= right_idx:
            return

        mid = (left_idx + right_idx) // 2

        # Divide step
        states = [ElementState.DEFAULT] * n
        for i in range(left_idx, mid + 1):
            states[i] = ElementState.ACTIVE
        for i in range(mid + 1, right_idx + 1):
            states[i] = ElementState.COMPARING

        add_step(
            AnimationType.DIVIDE,
            list(range(left_idx, right_idx + 1)),
            arr[left_idx:right_idx + 1],
            states,
            f"배열을 [{left_idx}:{mid}]와 [{mid+1}:{right_idx}]로 분할합니다.",
            3,
            {"left": left_idx, "mid": mid, "right": right_idx},
        )

        merge_sort_helper(left_idx, mid)
        merge_sort_helper(mid + 1, right_idx)

        # Merge
        merge(left_idx, mid, right_idx)

    def merge(left_idx, mid, right_idx):
        left = arr[left_idx:mid + 1]
        right = arr[mid + 1:right_idx + 1]

        states = [ElementState.DEFAULT] * n
        for i in range(left_idx, right_idx + 1):
            states[i] = ElementState.ACTIVE

        add_step(
            AnimationType.MERGE,
            list(range(left_idx, right_idx + 1)),
            arr[left_idx:right_idx + 1],
            states,
            f"[{left}]와 [{right}]를 병합합니다.",
            10,
            {"left": left, "right": right},
        )

        i = j = 0
        k = left_idx

        while i < len(left) and j < len(right):
            comparisons[0] += 1
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1

            states = [ElementState.DEFAULT] * n
            states[k] = ElementState.SORTED

            add_step(
                AnimationType.SET,
                [k],
                [arr[k]],
                states,
                f"{arr[k]}를 위치 {k}에 배치합니다.",
                13,
            )
            k += 1

        while i < len(left):
            arr[k] = left[i]
            states = [ElementState.DEFAULT] * n
            states[k] = ElementState.SORTED
            add_step(
                AnimationType.SET,
                [k],
                [arr[k]],
                states,
                f"남은 {arr[k]}를 위치 {k}에 배치합니다.",
                14,
            )
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            states = [ElementState.DEFAULT] * n
            states[k] = ElementState.SORTED
            add_step(
                AnimationType.SET,
                [k],
                [arr[k]],
                states,
                f"남은 {arr[k]}를 위치 {k}에 배치합니다.",
                17,
            )
            j += 1
            k += 1

    merge_sort_helper(0, n - 1)

    # Final state
    add_step(
        AnimationType.SORTED,
        list(range(n)),
        arr.copy(),
        [ElementState.SORTED] * n,
        f"정렬 완료! 비교: {comparisons[0]}회",
        8,
    )

    viz.final_data = arr.copy()
    viz.total_comparisons = comparisons[0]
    viz.total_swaps = 0

    return viz
