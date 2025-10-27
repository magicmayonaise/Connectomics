from connectomics import _matrix
from connectomics.effective import dense_matrix_power, slice_from_effective
from connectomics.paths import meet_in_the_middle


def build_graph():
    return {
        0: [1, 2],
        1: [3],
        2: [3, 4],
        3: [5],
        4: [5],
        5: [],
        6: [7],
        7: [],
    }


def build_adjacency(graph):
    n = max(graph) + 1
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for src, dsts in graph.items():
        for dst in dsts:
            matrix[src][dst] = 1.0
    return matrix


def test_meet_in_the_middle_identifies_nodes():
    graph = build_graph()
    result = meet_in_the_middle(graph, {0}, {5}, max_depth=3)
    expected = {0, 1, 2, 3, 4, 5}
    assert result["within_k"] == expected


def test_slice_from_effective_selects_bridges():
    graph = build_graph()
    adjacency = build_adjacency(graph)
    n = len(adjacency)
    effective = _matrix.zeros(n, n)
    for power in range(1, 4):
        effective = _matrix.add(effective, dense_matrix_power(adjacency, power))
    slice_result = slice_from_effective(effective, [0], [5], min_score=0.0)
    assert set(slice_result.selected) == {1, 2, 3, 4}
    scores = slice_result.to_mapping()
    for node in slice_result.selected:
        assert scores[node] > 0
    assert 6 not in scores and 7 not in scores
