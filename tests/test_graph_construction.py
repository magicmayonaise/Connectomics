import networkx as nx
import pandas as pd

from cx_connectome.cx_network import build_connectivity_graph
from tests.conftest import FakeCAVEclient


def _build_synthetic_client() -> FakeCAVEclient:
    n1_roots = list(range(1, 6))
    synapses: list[dict[str, int]] = []
    n2_start = 1000
    n3_start = 2000
    synapse_id = 1

    for n1 in n1_roots:
        for n2 in range(n2_start, n2_start + 50):
            for _ in range(5):
                synapses.append({
                    "pre_pt_root_id": n1,
                    "post_pt_root_id": n2,
                    "id": synapse_id,
                })
                synapse_id += 1

            n3 = n3_start + (n2 - n2_start)
            for _ in range(10):
                synapses.append({
                    "pre_pt_root_id": n2,
                    "post_pt_root_id": n3,
                    "id": synapse_id,
                })
                synapse_id += 1

    synapse_df = pd.DataFrame(synapses)
    roots = set(synapse_df["pre_pt_root_id"]).union(synapse_df["post_pt_root_id"]).union(n1_roots)
    cell_type_df = pd.DataFrame(
        {"pt_root_id": rid, "cell_type": f"Type{rid}"} for rid in roots
    )

    return FakeCAVEclient(synapse_df, cell_type_df)


def test_connectivity_graph_construction(tmp_path):
    client = _build_synthetic_client()
    seed_path = tmp_path / "seeds.txt"
    seed_path.write_text("\n".join(str(r) for r in range(1, 6)), encoding="utf8")

    graph, n1_n2, n2_n3 = build_connectivity_graph(
        seed_path,
        client=client,
        materialization=783,
        n1_n2_output=tmp_path / "n1n2.parquet",
        n2_n3_output=tmp_path / "n2n3.parquet",
        graph_output=tmp_path / "graph.gpickle",
    )

    assert graph.number_of_nodes() > 100
    assert nx.is_weakly_connected(graph)
    assert all("cell_type" in data for _, data in graph.nodes(data=True))
    assert not n1_n2.empty
    assert not n2_n3.empty
