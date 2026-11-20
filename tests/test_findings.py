"""The claims the page makes, checked on a small scale so the suite stays quick."""

import numpy as np

from tsp_ga import Config, engine, load, run
from tsp_ga.experiments import BUDGET, calibrate, compare, funnel_verdict, plain, summarise
from tsp_ga.ga import CROSSOVERS, FULL_EFFORT

DJ = load("dj38")
QA = load("qa194")
SEEDS = range(1, 7)


def mean_gap(instance, config):
    return np.mean([run(instance, config, seed).gap for seed in SEEDS])


def test_the_rebuild_reproduces_the_2020_result_on_qatar():
    # 2020 README: 26,013 on an open path. On closed tours the same settings give about 180%.
    gap = mean_gap(QA, Config())
    assert 1.5 < gap < 2.2


def test_edge_recombination_beats_partially_mapped_crossover():
    small = dict(schedule="fixed", base=100, extra=0, init_size=200, generations=150)
    assert mean_gap(QA, Config(crossover="er", **small)) < mean_gap(QA, Config(**small)) - 0.3


def test_local_search_is_worth_more_than_everything_else():
    config = Config(schedule="fixed", base=20, extra=0, init_size=40, generations=60,
                    crossover="er", mutation="inversion", effort_start=FULL_EFFORT,
                    effort_end=FULL_EFFORT)
    results = [run(QA, config, seed) for seed in SEEDS]
    assert np.mean([r.gap for r in results]) < 0.01
    assert max(r.cost[-1] for r in results) < BUDGET / 4  # at a fraction of the 2020 budget


def test_breeding_beats_local_search_alone_at_the_same_budget():
    budget = 20_000
    config = calibrate("qa194", Config(schedule="fixed", base=10, extra=0, init_size=20,
                                       generations=50, crossover="er", mutation="inversion",
                                       effort_start=FULL_EFFORT, effort_end=FULL_EFFORT), budget)
    bred = np.mean([run(QA, config, seed).length for seed in SEEDS])
    alone = np.mean([engine.restarts(QA.distance, QA.neighbours, budget, seed)[0]
                     for seed in SEEDS])
    assert bred < alone


def test_crossing_good_parents_costs_most_under_partially_mapped_crossover():
    parents = run(QA, Config(crossover="er", mutation="inversion", schedule="fixed", base=200,
                             extra=0, generations=60), 2)
    load_ = {}
    for name, code in CROSSOVERS.items():
        o = engine.offspring_of(QA.distance, parents.population, parents.lengths, code, 1500, 3)
        load_[name] = (o[:, 2] + o[:, 3]).mean() / (o[:, 0] + o[:, 1]).mean() - 1
        assert (o[:, 4] <= 2 * QA.n).all()
    assert load_["pmx"] > 2 * load_["ox"] > 0
    assert load_["pmx"] > 2 * load_["er"] > 0


def test_partially_mapped_populations_become_inbred():
    kin = {x: np.nanmean(run(QA, Config(crossover=x), 1).breeding()["kinship"][-50:])
           for x in ("pmx", "er")}
    assert kin["pmx"] > 0.9 > kin["er"]


def test_breeding_terms_are_well_formed():
    b = run(DJ, Config(generations=40, crossover="ox"), 3).breeding()
    assert (b["differential"] <= 0).all()  # the selected are never longer than average
    assert (b["sd"] >= 0).all() and (b["kept"] > 0).all()
    assert np.nanmin(b["kinship"]) >= 0 and np.nanmax(b["kinship"]) <= 1


def test_budget_arithmetic_and_summaries():
    for size in (10, 450, 1000):
        config = plain(size)
        assert config.budget == BUDGET and config.evaluations() >= BUDGET
        assert run(DJ, config.with_(generations=3), 1).log[-1, engine.EVALS] <= BUDGET
    s = summarise([110, 100, 120, 100], [1, 1, 1, 1], 100)
    assert s["gap"] == np.mean([0.1, 0, 0.2, 0]) and s["solved"] == 0.5 and s["best"] == 0
    assert s["lo"] <= s["gap"] <= s["hi"]
    better = summarise([100, 101, 100, 102], [1] * 4, 100)
    d = compare(better, s)
    assert d["difference"] < 0 and d["lo"] <= d["difference"] <= d["hi"]
    rows = [dict(arm="Fixed 10", schedule="fixed", **s),
            dict(arm="Fixed 450", schedule="fixed", **better),
            dict(arm="Funnel 700 to 200 (2020)", schedule="funnel", **s),
            dict(arm="Reverse funnel 200 to 700", schedule="reverse", **s)]
    verdict = funnel_verdict(rows)
    assert verdict["best_fixed"] == "Fixed 450"
    assert verdict["funnel_vs_best_fixed"]["difference"] > 0
