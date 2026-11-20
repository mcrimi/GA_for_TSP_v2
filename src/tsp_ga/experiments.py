"""The experiments behind the page and the report.

Every comparison is made at the same amount of work: the 269,800 tour
evaluations the 2020 program spent. Local search is charged too (a 2-opt
check reads four distances, a tour evaluation reads n), and runs that use it
have their number of generations calibrated so the total comes out the same.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import engine
from .ga import CROSSOVERS, FULL_EFFORT, Config, run
from .instance import TITLES, load

ORIGINAL = Config()
BUDGET = ORIGINAL.evaluations()  # 269,800
SEEDS = 40
CROSSOVER_NAMES = {"pmx": "Partially mapped (2020)", "ox": "Order", "er": "Edge recombination"}


def plain(size: int, **kw) -> Config:
    """A fixed population of ``size`` with as many generations as the budget allows."""
    generations = -(-(BUDGET - 2 * size) // (2 * size))
    return Config(schedule="fixed", base=size, extra=0, init_size=2 * size,
                  generations=generations, budget=BUDGET, **kw)


def schedule_arms(**kw) -> dict[str, Config]:
    """Fixed populations of several sizes against funnels, all at the 2020 budget."""
    arms = {f"Fixed {n}": plain(n, **kw) for n in (10, 20, 50, 100, 200, 450, 700, 1000)}
    arms["Funnel 700 to 200 (2020)"] = ORIGINAL.with_(**kw)
    arms["Reverse funnel 200 to 700"] = ORIGINAL.with_(schedule="reverse", **kw)
    # A steeper funnel, closer to the breeding picture.
    steep = Config(schedule="geometric", base=20, extra=1480, init_size=1500, **kw)
    arms["Steep funnel 1500 to 20"] = calibrate("dj38", steep)
    return arms


MEMETIC = dict(crossover="er", mutation="inversion", effort_start=FULL_EFFORT,
               effort_end=FULL_EFFORT)


def memetic_arms() -> dict[str, Config]:
    """Local search on every child; generations are calibrated to the budget later."""
    def fixed(n):
        return Config(schedule="fixed", base=n, extra=0, init_size=2 * n, generations=50,
                      **MEMETIC)

    arms = {f"Fixed {n}": fixed(n) for n in (5, 10, 20, 40, 80)}
    arms["Funnel 80 to 10"] = Config(schedule="funnel", base=10, extra=70, init_size=80,
                                     generations=50, **MEMETIC)
    arms["Steep funnel 200 to 5"] = Config(schedule="geometric", base=5, extra=195,
                                           init_size=200, generations=50, **MEMETIC)
    # The breeding funnel taken literally: many candidates screened cheaply,
    # a few evaluated thoroughly.
    arms["Steep funnel, effort 1 to full"] = Config(
        schedule="geometric", base=5, extra=195, init_size=200, generations=50,
        crossover="er", mutation="inversion", effort_start=1, effort_end=60)
    arms["Fixed 20, partially mapped"] = fixed(20).with_(crossover="pmx")
    arms["Fixed 20, order"] = fixed(20).with_(crossover="ox")
    return arms


def _one(task):
    name, config, seed = task
    result = run(load(name), config, seed)
    return result.length, float(result.cost[-1])


def _restart(task):
    name, budget, seed = task
    instance = load(name)
    best, cost, starts = engine.restarts(instance.distance, instance.neighbours, budget, seed)
    return int(best), float(cost), int(starts)


def _pool(workers=None):
    return ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1))


def calibrate(name: str, config: Config, budget: float = BUDGET) -> Config:
    """Choose the number of generations so that the run costs about ``budget``."""
    instance = load(name)
    config = config.with_(budget=0)
    for _ in range(10):
        cost = np.mean([run(instance, config, seed).cost[-1] for seed in (901, 902, 903)])
        if 1.0 <= cost / budget < 1.03:  # a little over: the budget then cuts the run off
            break
        config = config.with_(generations=max(2, round(config.generations * 1.015 * budget / cost)))
    return config.with_(budget=budget)


def summarise(lengths, costs, optimum) -> dict:
    gaps = np.asarray(lengths) / optimum - 1
    rng = np.random.default_rng(2020)
    boots = rng.choice(gaps, size=(4000, len(gaps))).mean(axis=1)
    return {"n": len(gaps), "gap": float(gaps.mean()), "sd": float(gaps.std(ddof=1)),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "best": float(gaps.min()), "worst": float(gaps.max()),
            "solved": float(np.mean(np.asarray(lengths) == optimum)),
            "cost": float(np.mean(costs)), "gaps": [float(g) for g in np.sort(gaps)]}


def compare(a: dict, b: dict) -> dict:
    """Difference in mean gap (a - b) with a bootstrap interval. Negative: a is better."""
    rng = np.random.default_rng(2020)
    x, y = np.array(a["gaps"]), np.array(b["gaps"])
    boots = (rng.choice(x, size=(4000, len(x))).mean(axis=1)
             - rng.choice(y, size=(4000, len(y))).mean(axis=1))
    return {"difference": float(x.mean() - y.mean()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}


def run_arms(name: str, arms: dict[str, Config], seeds: int, pool) -> list[dict]:
    optimum = load(name).optimum
    tasks = [(name, config, seed) for config in arms.values() for seed in range(1, seeds + 1)]
    results = list(pool.map(_one, tasks, chunksize=4))
    out = []
    for i, (label, config) in enumerate(arms.items()):
        chunk = results[i * seeds:(i + 1) * seeds]
        row = {"arm": label, "schedule": config.schedule, "generations": config.generations,
               "sizes": [int(config.sizes()[0]), int(config.sizes()[-1])],
               "config": config.to_dict()}
        row.update(summarise([r[0] for r in chunk], [r[1] for r in chunk], optimum))
        out.append(row)
    return out


def funnel_verdict(rows: list[dict]) -> dict:
    """The 2020 funnel against the best fixed population, and against each other shape."""
    by = {r["arm"]: r for r in rows}
    fixed = [r for r in rows if r["schedule"] == "fixed" and r["arm"].startswith("Fixed")
             and "," not in r["arm"]]
    best_fixed = min(fixed, key=lambda r: r["gap"])
    funnels = [r for r in rows if r["schedule"] in ("funnel", "geometric")
               and "effort" not in r["arm"]]
    best_funnel = min(funnels, key=lambda r: r["gap"])
    out = {"best_fixed": best_fixed["arm"], "best_funnel": best_funnel["arm"],
           "best_funnel_vs_best_fixed": compare(best_funnel, best_fixed)}
    if "Funnel 700 to 200 (2020)" in by:
        funnel = by["Funnel 700 to 200 (2020)"]
        out["funnel_vs_best_fixed"] = compare(funnel, best_fixed)
        out["funnel_vs_same_average"] = compare(funnel, by["Fixed 450"])
        out["funnel_vs_reverse"] = compare(funnel, by["Reverse funnel 200 to 700"])
    return out


def heritability(name: str = "qa194", pairs: int = 5000) -> list[dict]:
    """Cross random pairs from populations at four stages of improvement.

    For each crossover: the slope of offspring on mid-parent tour length
    (heritability), the share of an offspring's edges that came from a parent,
    and how much longer offspring are than their parents (the cost of crossing).
    """
    instance = load(name)
    n = instance.n
    stages = []
    ctr = np.zeros(1, dtype=np.int64)
    tours = np.empty((1000, n), dtype=np.int32)
    for row in tours:
        engine.random_tour(row, 9, ctr)
    lengths = np.array([engine.tour_length(instance.distance, t) for t in tours])
    stages.append(("Random tours", tours, lengths))
    for generations in (30, 100, 300):
        result = run(instance, Config(crossover="er", mutation="inversion", schedule="fixed",
                                      base=450, extra=0, generations=generations), 2)
        stages.append((f"After {generations} generations", result.population, result.lengths))
    out = []
    for stage, tours, lengths in stages:
        for key, code in CROSSOVERS.items():
            o = engine.offspring_of(instance.distance, tours, lengths, code, pairs, 3)
            mid = (o[:, 0] + o[:, 1]) / 2
            off = (o[:, 2] + o[:, 3]) / 2
            slope, intercept = np.polyfit(mid, off, 1)
            keep = np.linspace(0, pairs - 1, 400).astype(int)
            out.append({
                "stage": stage, "crossover": key, "label": CROSSOVER_NAMES[key],
                "parent_gap": float(lengths.mean() / instance.optimum - 1),
                "heritability": float(slope),
                "correlation": float(np.corrcoef(mid, off)[0, 1]),
                "inherited": float(o[:, 4].mean() / (2 * n)),
                "load": float(off.mean() / mid.mean() - 1),
                "better_than_parents": float(np.mean(np.minimum(o[:, 2], o[:, 3])
                                                     < np.minimum(o[:, 0], o[:, 1]))),
                "points": {"mid": (mid[keep] / instance.optimum - 1).tolist(),
                           "off": (off[keep] / instance.optimum - 1).tolist()},
            })
    return out


def traces(name: str = "qa194", seeds: int = 20) -> dict:
    """Curves over a run, averaged over seeds: progress, variation and kinship."""
    instance = load(name)
    grid = np.linspace(0, BUDGET, 120)
    arms = {
        "2020: partially mapped, funnel": ORIGINAL,
        "Order crossover": ORIGINAL.with_(crossover="ox"),
        "Edge recombination": ORIGINAL.with_(crossover="er"),
        "Edge recombination + local search": calibrate(
            name, Config(schedule="fixed", base=40, extra=0, init_size=80, generations=50,
                         **MEMETIC)),
    }
    out = {"cost": grid.tolist(), "arms": []}
    for label, config in arms.items():
        gap, kin, sd, load_, sizes = [], [], [], [], None
        for seed in range(1, seeds + 1):
            result = run(instance, config, seed)
            cost = result.cost
            best = result.best / instance.optimum - 1
            idx = np.searchsorted(cost, grid, side="right") - 1
            gap.append(np.where(idx >= 0, best[np.maximum(idx, 0)], np.nan))
            b = result.breeding()
            kin.append(b["kinship"])
            sd.append(b["sd"] / b["mean"])
            load_.append(b["offspring"] / b["selected"] - 1)
            sizes = b["kept"]
        ran = min(len(k) for k in kin)  # runs cut off by the budget differ by a generation or two
        pick = slice(0, ran, max(1, ran // 100))

        def mean(series, ran=ran, pick=pick):
            return np.mean([s[:ran] for s in series], axis=0)[pick].tolist()

        gap = np.array(gap)
        mean_gap = [None if np.isnan(col).all() else float(np.nanmean(col)) for col in gap.T]
        out["arms"].append({
            "arm": label, "generations": ran, "gap": mean_gap,
            "generation": list(range(1, ran + 1))[pick],
            "kinship": mean(kin), "variation": mean(sd), "load": mean(load_),
            "kept": [int(v) for v in sizes[:ran][pick]],
        })
    return out


def all_experiments(seeds: int = SEEDS, workers: int | None = None) -> dict:
    out = {"budget": BUDGET, "seeds": seeds, "original": ORIGINAL.to_dict(),
           "instances": {k: {"title": TITLES[k], "optimum": load(k).optimum, "n": load(k).n}
                         for k in TITLES}}
    with _pool(workers) as pool:
        out["schedules"] = {}
        for name in ("dj38", "qa194"):
            rows = run_arms(name, schedule_arms(), seeds, pool)
            out["schedules"][name] = {"arms": rows, "verdict": funnel_verdict(rows)}
        rows = run_arms("qa194", schedule_arms(crossover="er", mutation="inversion"), seeds, pool)
        out["schedules_er"] = {"qa194": {"arms": rows, "verdict": funnel_verdict(rows)}}

        operators = {f"{c}+{m}": plain(100, crossover=c, mutation=m)
                     for c in CROSSOVERS for m in ("swap", "inversion")}
        out["operators"] = {name: run_arms(name, operators, seeds, pool)
                            for name in ("dj38", "qa194")}

        # With local search: the full budget on two instances, and smaller budgets on
        # Qatar to see how the answer depends on how much work there is to spend.
        out["memetic"] = []
        for name, budget in (("qa194", BUDGET // 16), ("qa194", BUDGET // 4), ("qa194", BUDGET),
                             ("uy734", BUDGET)):
            arms = {label: calibrate(name, config, budget)
                    for label, config in memetic_arms().items()}
            rows = run_arms(name, arms, seeds, pool)
            optimum = load(name).optimum
            starts = list(pool.map(_restart, [(name, budget, s) for s in range(1, seeds + 1)]))
            restart = {"arm": "No breeding: 2-opt from random tours", "schedule": "none",
                       "starts": float(np.mean([s[2] for s in starts]))}
            restart.update(summarise([s[0] for s in starts], [s[1] for s in starts], optimum))
            out["memetic"].append({"instance": name, "budget": budget, "arms": rows,
                                   "restarts": restart, "verdict": funnel_verdict(rows)})
    out["heritability"] = heritability()
    out["traces"] = traces()
    return out
