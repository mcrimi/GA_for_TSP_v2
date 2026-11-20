"""Command line: ``tsp-ga build`` and ``tsp-ga run``."""

import argparse
import time

import numpy as np

from .experiments import SEEDS
from .export import build, fixtures, instances, write_json
from .ga import CROSSOVERS, FULL_EFFORT, MUTATIONS, SCHEDULES, Config, run
from .instance import TITLES, load


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="tsp-ga", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="run the experiments and write the page's data")
    b.add_argument("--out", default="web")
    b.add_argument("--seeds", type=int, default=SEEDS, help="runs per setting")
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--fixtures-only", action="store_true",
                   help="write only instances.json and fixtures.json (fast)")

    r = sub.add_parser("run", help="run the algorithm and print the result")
    r.add_argument("instance", choices=sorted(TITLES))
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--schedule", choices=SCHEDULES, default="funnel")
    r.add_argument("--crossover", choices=sorted(CROSSOVERS), default="pmx")
    r.add_argument("--mutation", choices=sorted(MUTATIONS), default="swap")
    r.add_argument("--generations", type=int, default=300)
    r.add_argument("--base", type=int, default=200)
    r.add_argument("--extra", type=int, default=500)
    r.add_argument("--init-size", type=int, default=500)
    r.add_argument("--local-search", action="store_true", help="2-opt on every child")
    r.add_argument("--budget", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "build":
        start = time.time()
        if args.fixtures_only:
            paths = [write_json(instances(), f"{args.out}/data/instances.json"),
                     write_json(fixtures(), f"{args.out}/test/fixtures.json")]
        else:
            paths = build(args.out, args.seeds, args.workers)
        for path in paths:
            print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
        print(f"{time.time() - start:.0f} s")
        return

    effort = FULL_EFFORT if args.local_search else 0
    config = Config(generations=args.generations, schedule=args.schedule, base=args.base,
                    extra=0 if args.schedule == "fixed" else args.extra,
                    init_size=args.init_size, crossover=args.crossover, mutation=args.mutation,
                    effort_start=effort, effort_end=effort, budget=args.budget)
    instance = load(args.instance)
    start = time.time()
    results = [run(instance, config, seed) for seed in range(1, args.seeds + 1)]
    gaps = np.array([r.gap for r in results])
    print(f"{TITLES[args.instance]}; optimal tour {instance.optimum}")
    print(f"{args.seeds} runs, {results[0].cost[-1]:,.0f} tour evaluations' worth of work each, "
          f"{(time.time() - start) / args.seeds:.2f} s per run")
    print(f"best tour found   {min(r.length for r in results)}")
    print(f"above the optimum {100 * gaps.mean():.2f}% on average "
          f"({100 * gaps.min():.2f}% to {100 * gaps.max():.2f}%)")
