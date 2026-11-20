import json

import numpy as np

from tsp_ga import engine, load
from tsp_ga.export import FIXTURE_CASES, _plain, _round, fixtures, instances, write_json


def test_fixtures_cover_every_operator_and_schedule():
    used = {(c.crossover, c.mutation, c.schedule) for _, _, _, c in FIXTURE_CASES}
    assert {u[0] for u in used} == {"pmx", "ox", "er"}
    assert {u[1] for u in used} == {"swap", "inversion"}
    assert {u[2] for u in used} == {"funnel", "fixed", "reverse", "geometric"}
    assert any(c.effort_end > 0 for *_, c in FIXTURE_CASES)


def test_fixtures_are_self_consistent(tmp_path):
    data = json.loads(write_json(fixtures(), tmp_path / "f.json").read_text())
    assert data["columns"] == engine.N_COLS == 13
    cut = 0
    for case in data["cases"]:
        instance = load(case["instance"])
        tour = np.array(case["tour"], dtype=np.int32)
        assert engine.tour_length(instance.distance, tour) == case["length"]
        assert case["length"] == case["log"][-1][engine.BEST] == min(case["lengths"])
        assert len(case["sizes"]) == case["config"]["generations"]
        cut += len(case["log"]) < case["config"]["generations"]
    assert cut >= 1  # at least one run is stopped by its budget


def test_page_instances_carry_their_coordinates_and_optimum():
    data = _plain(instances())
    assert set(data) == {"dj38", "qa194"}
    assert len(data["qa194"]["coords"]) == 194 and data["qa194"]["optimum"] == 9352
    assert data["dj38"]["coords"][0] == [11003.6111, 42102.5]


def test_json_helpers():
    assert _plain({"a": float("nan"), "b": np.int64(3), "c": np.array([1.5])}) == {
        "a": None, "b": 3, "c": [1.5]}
    assert _round({"a": [1.23456789, 2], "b": "x"}, 3) == {"a": [1.23, 2], "b": "x"}
