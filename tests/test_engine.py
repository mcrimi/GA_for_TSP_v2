import numpy as np
import pytest

from tsp_ga import Config, engine, from_coords, load, run
from tsp_ga.ga import CROSSOVERS, FULL_EFFORT

DJ = load("dj38")
QA = load("qa194")


def is_tour(tour, n):
    return sorted(tour.tolist()) == list(range(n))


def edges(tour):
    return {frozenset((int(u), int(v))) for u, v in zip(tour, np.roll(tour, -1), strict=True)}


def scratch(n):
    return (np.empty(n, np.int32), np.empty((n, 4), np.int32), np.empty(n, np.int32),
            np.empty(n, np.int32), np.empty(n, np.int32))


def random_pair(n, seed):
    ctr = np.zeros(1, np.int64)
    a, b = np.empty(n, np.int32), np.empty(n, np.int32)
    engine.random_tour(a, seed, ctr)
    engine.random_tour(b, seed, ctr)
    return a, b, ctr


@pytest.mark.parametrize("crossover", sorted(CROSSOVERS))
def test_crossovers_return_valid_tours(crossover):
    n = 23
    c1, c2 = np.empty(n, np.int32), np.empty(n, np.int32)
    for seed in range(100):
        a, b, ctr = random_pair(n, seed)
        engine.cross(CROSSOVERS[crossover], a, b, c1, c2, seed, ctr, *scratch(n))
        assert is_tour(c1, n) and is_tour(c2, n)


@pytest.mark.parametrize("crossover", sorted(CROSSOVERS))
def test_crossing_a_tour_with_itself_gives_it_back(crossover):
    n = 17
    a, _, ctr = random_pair(n, 4)
    c1, c2 = np.empty(n, np.int32), np.empty(n, np.int32)
    engine.cross(CROSSOVERS[crossover], a, a, c1, c2, 4, ctr, *scratch(n))
    assert edges(c1) == edges(a) and edges(c2) == edges(a)


def test_textbook_crossover_examples():
    # Goldberg and Lingle's example, zero-based: cut after the 3rd and 6th city.
    a = np.array([2, 3, 7, 1, 6, 0, 5, 4], np.int32)
    b = np.array([3, 1, 4, 0, 5, 7, 2, 6], np.int32)
    child, where = np.empty(8, np.int32), np.empty(8, np.int32)
    engine.pmx(b, a, child, 3, 6, where)
    assert child.tolist() == [2, 3, 1, 0, 5, 7, 6, 4]
    engine.ox(a, b, child, 3, 6, where)
    assert child[3:6].tolist() == [1, 6, 0] and is_tour(child, 8)
    # Order crossover keeps the other parent's order for the rest, starting after the cut.
    assert child.tolist() == [4, 5, 7, 1, 6, 0, 2, 3]


def test_edge_recombination_inherits_almost_every_edge():
    n = 60
    c1, c2 = np.empty(n, np.int32), np.empty(n, np.int32)
    share = []
    for seed in range(50):
        a, b, ctr = random_pair(n, seed)
        engine.cross(engine.ER, a, b, c1, c2, seed, ctr, *scratch(n))
        share.append(len(edges(c1) & (edges(a) | edges(b))) / n)
    assert np.mean(share) > 0.93


def test_mutations_keep_a_tour_a_tour():
    n = 30
    for kind in (engine.SWAP, engine.INVERSION):
        a, _, ctr = random_pair(n, kind)
        before = a.copy()
        for _ in range(50):
            engine.mutate(a, kind, 1, ctr)
            assert is_tour(a, n)
        assert not np.array_equal(a, before)


def test_two_opt_only_shortens_and_reports_the_change_exactly():
    pos = np.empty(QA.n, np.int32)
    out = np.zeros(3, np.int64)
    for seed in range(8):
        tour, _, _ = random_pair(QA.n, seed)
        before = engine.tour_length(QA.distance, tour)
        engine.two_opt(tour, pos, QA.distance, QA.neighbours, FULL_EFFORT, out)
        assert is_tour(tour, QA.n)
        assert engine.tour_length(QA.distance, tour) == before + out[0] < before
        assert (pos[tour] == np.arange(QA.n)).all()
        engine.two_opt(tour, pos, QA.distance, QA.neighbours, FULL_EFFORT, out)
        assert out[0] == 0 and out[1] == 0  # already a local optimum


def test_two_opt_respects_the_move_limit():
    tour, _, _ = random_pair(QA.n, 3)
    out = np.zeros(3, np.int64)
    engine.two_opt(tour, np.empty(QA.n, np.int32), QA.distance, QA.neighbours, 7, out)
    assert out[1] == 7 and out[2] >= 7


def test_two_opt_untangles_a_crossing():
    square = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
    inst = from_coords("square", square)
    tour = np.array([0, 2, 1, 3], np.int32)  # the two diagonals
    out = np.zeros(3, np.int64)
    engine.two_opt(tour, np.empty(4, np.int32), inst.distance, inst.neighbours, FULL_EFFORT, out)
    assert engine.tour_length(inst.distance, tour) == 40


def test_runs_are_reproducible_and_seeds_differ():
    config = Config(generations=15)
    a, b, c = run(DJ, config, 5), run(DJ, config, 5), run(DJ, config, 6)
    assert np.array_equal(a.log, b.log) and np.array_equal(a.tour, b.tour)
    assert not np.array_equal(a.log, c.log)


@pytest.mark.parametrize("crossover", sorted(CROSSOVERS))
def test_a_run_keeps_its_books_straight(crossover):
    config = Config(generations=25, crossover=crossover, mutation="inversion",
                    effort_start=0, effort_end=3)
    result = run(DJ, config, 2)
    assert is_tour(result.tour, DJ.n)
    assert engine.tour_length(DJ.distance, result.tour) == result.length
    assert result.length <= result.lengths.min()
    assert (np.diff(result.best) <= 0).all()  # the best never gets worse
    assert result.length >= DJ.optimum
    for tour, length in zip(result.population, result.lengths, strict=True):
        assert is_tour(tour, DJ.n) and engine.tour_length(DJ.distance, tour) == length
    log = result.log
    assert log[-1, engine.EVALS] == config.evaluations()
    assert (log[:, engine.KEPT] == np.minimum(config.sizes(), log[:, engine.N_POP])).all()
    assert (log[:, engine.SUM_SEL] / log[:, engine.KEPT]
            <= log[:, engine.SUM_POP] / log[:, engine.N_POP] + 1e-9).all()


def test_the_original_settings_spend_the_original_budget():
    config = Config()
    assert config.evaluations() == 269_800  # the README of 2020 reports 269,796
    assert config.sizes()[0] == 698 and config.sizes()[-1] == 200
    assert (np.diff(config.sizes()) <= 0).all()
    assert (np.diff(config.with_(schedule="reverse").sizes()) >= 0).all()
    steep = config.with_(schedule="geometric", base=20, extra=1480).sizes()
    assert steep[0] == 1500 and steep[-1] == 20


def test_the_budget_stops_a_run_within_one_mating():
    result = run(DJ, Config(budget=3000), 1)
    assert 3000 <= result.log[-1, engine.EVALS] <= 3002
    assert len(result.log) < 300
    with_search = run(QA, Config(schedule="fixed", base=10, extra=0, init_size=20,
                                 generations=500, effort_start=FULL_EFFORT,
                                 effort_end=FULL_EFFORT, budget=5000), 1)
    assert 5000 <= with_search.cost[-1] < 5000 + 200  # one child's local search at most


@pytest.mark.parametrize("bad", [{"crossover": "magic"}, {"mutation": "none"},
                                 {"schedule": "zigzag"}, {"generations": 0}, {"base": 1},
                                 {"mutation_rate": 1.5}, {"effort_end": -1}, {"budget": -5}])
def test_impossible_settings_are_refused(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_seed_must_fit_in_32_bits():
    with pytest.raises(ValueError):
        run(DJ, Config(generations=1), 2**32)
