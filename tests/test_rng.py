import numpy as np

from tsp_ga import rng


def test_known_values_pin_the_algorithm():
    # The JavaScript implementation is tested against these same numbers.
    assert [rng.uint32(0, 0, i) for i in range(3)] == [1680814304, 1471256801, 3260664659]
    assert [rng.uint32(7, 100, i) for i in (0, 59)] == [29833046, 1845834364]


def test_the_compiled_and_plain_python_versions_agree():
    for args in [(0, 0, 0), (123456789, 3, 2**31), (2**32 - 1, 0, 2**32 - 1)]:
        assert rng.uint32(*args) == rng.uint32.py_func(*args)
        assert 0 <= rng.uint32(*args) < 2**32


def test_integers_are_in_range_and_uniform():
    ctr = np.zeros(1, dtype=np.int64)
    draws = np.array([rng.rand_int(5, ctr, 7) for _ in range(70_000)])
    assert ctr[0] == 70_000
    assert draws.min() == 0 and draws.max() == 6
    counts = np.bincount(draws)
    assert counts.min() > 9_500 and counts.max() < 10_500
    assert rng.rand_int(1, ctr, 1) == 0


def test_unit_draws_are_strictly_inside_zero_and_one():
    ctr = np.zeros(1, dtype=np.int64)
    u = np.array([rng.rand_unit(9, ctr) for _ in range(20_000)])
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.01


def test_different_seeds_give_unrelated_streams():
    a, b = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    x = [rng.rand_unit(1, a) for _ in range(20_000)]
    y = [rng.rand_unit(2, b) for _ in range(20_000)]
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03
