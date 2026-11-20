import numpy as np
import pytest

from tsp_ga import from_coords, load
from tsp_ga.engine import tour_length
from tsp_ga.instance import OPTIMA, parse_tsplib


@pytest.mark.parametrize("name", sorted(OPTIMA))
def test_shipped_instances_load(name):
    instance = load(name)
    assert instance.n == int(name[2:])
    assert instance.optimum == OPTIMA[name]
    d = instance.distance
    assert (d == d.T).all() and (np.diag(d) == 0).all() and (d >= 0).all()
    # No tour can be shorter than the proven optimum.
    assert tour_length(d, np.arange(instance.n, dtype=np.int32)) >= instance.optimum


def test_distances_follow_the_tsplib_rounding_rule():
    instance = from_coords("t", [[0, 0], [3, 4], [0, 1.4], [0, 1.5], [10, 10]])
    assert instance.distance[0, 1] == 5
    assert instance.distance[0, 2] == 1  # 1.4 rounds down
    assert instance.distance[0, 3] == 2  # 1.5 rounds up


def test_neighbours_are_nearest_first_and_ties_go_to_the_lower_index():
    instance = from_coords("t", [[0, 0], [1, 0], [-1, 0], [0, 5], [0, -2]])
    assert instance.neighbours[0].tolist() == [1, 2, 4, 3]
    assert 0 not in instance.neighbours[0]


def test_parser_reads_both_header_styles_and_refuses_other_metrics():
    text = ("NAME : x\nTYPE: TSP\nEDGE_WEIGHT_TYPE : EUC_2D\n"
            "NODE_COORD_SECTION\n1 0 0\n2 1.5 2\nEOF\n")
    name, coords = parse_tsplib(text)
    assert name == "x" and coords.tolist() == [[0, 0], [1.5, 2]]
    with pytest.raises(ValueError):
        parse_tsplib(text.replace("EUC_2D", "GEO"))
    with pytest.raises(ValueError):
        parse_tsplib(text.replace("2 1.5 2", "2 1.5"))
    with pytest.raises(ValueError):
        from_coords("small", [[0, 0], [1, 1]])
    with pytest.raises(FileNotFoundError):
        load("atlantis")
