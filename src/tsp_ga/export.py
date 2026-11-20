"""Write the JSON the web page reads, and the trajectories the browser engine must match."""

import json
import math
from pathlib import Path

import numpy as np

from . import engine
from .experiments import all_experiments
from .ga import FULL_EFFORT, Config, run
from .instance import TITLES, load

PAGE_INSTANCES = ("dj38", "qa194")

#: Runs replayed in JavaScript. Between them they use every operator, every
#: schedule, local search (capped and to a local optimum) and the budget cut-off.
FIXTURE_CASES = [
    ("the 2020 settings, shortened", "dj38", 7, Config(generations=12)),
    ("order crossover, inversion, fixed", "dj38", 11,
     Config(schedule="fixed", base=60, extra=0, init_size=120, generations=15,
            crossover="ox", mutation="inversion")),
    ("edge recombination, reverse funnel", "dj38", 3,
     Config(schedule="reverse", base=20, extra=60, init_size=50, generations=12,
            crossover="er", mutation_rate=0.5)),
    ("edge recombination with full local search", "qa194", 5,
     Config(schedule="fixed", base=12, extra=0, init_size=24, generations=8, crossover="er",
            mutation="inversion", effort_start=FULL_EFFORT, effort_end=FULL_EFFORT)),
    ("steep funnel with rising effort, cut off by the budget", "qa194", 9,
     Config(schedule="geometric", base=5, extra=55, init_size=60, generations=14,
            crossover="pmx", mutation="inversion", effort_start=1, effort_end=40, budget=600)),
    ("partially mapped on Qatar", "qa194", 2,
     Config(schedule="funnel", base=20, extra=40, init_size=40, generations=6)),
]


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def _round(value, digits=5):
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def write_json(value, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), separators=(",", ":"), allow_nan=False))
    return path


def instances() -> dict:
    """Coordinates of the instances the page offers, exactly as in the TSPLIB files."""
    out = {}
    for name in PAGE_INSTANCES:
        instance = load(name)
        out[name] = {"title": TITLES[name], "optimum": instance.optimum,
                     "coords": instance.coords}
    return out


def fixtures() -> dict:
    cases = []
    for label, name, seed, config in FIXTURE_CASES:
        instance = load(name)
        result = run(instance, config, seed)
        cases.append({
            "name": label, "instance": name, "seed": seed, "config": config.to_dict(),
            "sizes": config.sizes(), "effort": config.effort(),
            "log": result.log, "tour": result.tour, "length": result.length,
            "lengths": result.lengths,
            # Spot checks on the derived data, so a mismatch is easy to locate.
            "distance_sum": int(instance.distance.astype(np.int64).sum()),
            "neighbours_of_0": instance.neighbours[0],
        })
    return {"columns": engine.N_COLS, "cases": cases}


def build(out_dir, seeds: int, workers: int | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        write_json(instances(), out_dir / "data" / "instances.json"),
        write_json(fixtures(), out_dir / "test" / "fixtures.json"),
        write_json(_round(_plain(all_experiments(seeds, workers))),
                   out_dir / "data" / "experiments.json"),
    ]
