import networkx as nx

from cx_connectome.functional_roles import (
    annotate_functional_roles,
    collect_functional_neurons,
    infer_layer_membership,
)


def _build_sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(1, cell_type="hDeltaF")
    graph.add_node(2, cell_type="PFNd")
    graph.add_node(3, cell_type="Unknown")
    graph.add_node(4, cell_type="PEN_b")

    graph.add_edge(1, 2, layer="N1->N2")
    graph.add_edge(2, 3, layer="N2->N3")
    graph.add_edge(4, 2, layer="N1->N2")
    return graph


def test_annotation_and_layer_membership():
    graph = _build_sample_graph()

    annotate_functional_roles(graph)
    membership = infer_layer_membership(graph)

    assert graph.nodes[1]["FunctionalRole"] == "Sleep-Promoting"
    assert graph.nodes[2]["FunctionalRole"] == "Navigation"
    assert graph.nodes[3]["FunctionalRole"] is None
    assert graph.nodes[4]["FunctionalRole"] == "Sleep-Promoting"

    assert membership["N1"] == {1, 4}
    assert membership["N2"] == {2}
    assert membership["N3"] == {3}


def test_collect_functional_neurons_with_layers():
    graph = _build_sample_graph()
    annotate_functional_roles(graph)
    membership = infer_layer_membership(graph)

    summary = collect_functional_neurons(graph, membership)

    sleep_entries = summary["Sleep-Promoting"]
    navigation_entries = summary["Navigation"]

    sleep_ids = {entry["id"] for entry in sleep_entries}
    navigation_ids = {entry["id"] for entry in navigation_entries}

    assert sleep_ids == {1, 4}
    assert navigation_ids == {2}

    pen_b_entry = next(entry for entry in sleep_entries if entry["id"] == 4)
    assert pen_b_entry["layers"] == ["N1"]

    pfnd_entry = navigation_entries[0]
    assert pfnd_entry["layers"] == ["N2"]
