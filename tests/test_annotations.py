import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectomics_gui.annotations import CellTypeCacheWarning, fetch_cell_types


class _FlakyMaterialize:
    def __init__(self, responses: List[Any]):
        self._responses = responses
        self.calls = 0

    def query_table(self, *args: Any, **kwargs: Any) -> Any:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class _StubClient:
    def __init__(self, responses: List[Any]):
        self.materialize = _FlakyMaterialize(responses)


def _cache_path(tmp_path: Path, materialization: int = 783) -> Path:
    return tmp_path / f"cell_types_m{materialization}.json"


def test_fetch_cell_types_uses_cache_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_payload = {
        "materialization": 783,
        "values": {
            "111": {"pt_root_id": 111, "cell_type": "KC"},
        },
    }
    cache_file = _cache_path(tmp_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache_payload))

    called = False

    def _unexpected_client(*_: Any, **__: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("Client creation should not be required for cached lookups")

    monkeypatch.setattr("connectomics_gui.annotations._create_client", _unexpected_client)

    result = fetch_cell_types([111], cache_dir=tmp_path)
    assert result == {111: {"pt_root_id": 111, "cell_type": "KC"}}
    assert called is False


def test_fetch_cell_types_retries_on_error(tmp_path: Path) -> None:
    responses: List[Any] = [RuntimeError("network"), [{"pt_root_id": 222, "cell_type": "MBON"}]]
    client = _StubClient(responses)

    result = fetch_cell_types([222], cache_dir=tmp_path, client=client, max_retries=2, retry_delay=0)

    assert client.materialize.calls == 2
    assert result == {222: {"pt_root_id": 222, "cell_type": "MBON"}}

    payload = json.loads(_cache_path(tmp_path).read_text())
    assert payload["values"]["222"]["cell_type"] == "MBON"


def test_fetch_cell_types_warns_on_materialization_mismatch(tmp_path: Path) -> None:
    cache_file = _cache_path(tmp_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "materialization": 100,
                "values": {"333": {"pt_root_id": 333, "cell_type": "ON"}},
            }
        )
    )

    client = _StubClient([[{"pt_root_id": 333, "cell_type": "PN"}]])

    with pytest.warns(CellTypeCacheWarning):
        result = fetch_cell_types([333], cache_dir=tmp_path, client=client, retry_delay=0)

    assert result == {333: {"pt_root_id": 333, "cell_type": "PN"}}

    payload = json.loads(_cache_path(tmp_path).read_text())
    assert payload["materialization"] == 783
    assert payload["values"]["333"]["cell_type"] == "PN"
