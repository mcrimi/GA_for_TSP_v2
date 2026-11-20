"""A friendly front end to the compiled engine."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from . import engine
from .instance import Instance

CROSSOVERS = {"pmx": engine.PMX, "ox": engine.OX, "er": engine.ER}
MUTATIONS = {"swap": engine.SWAP, "inversion": engine.INVERSION}
SCHEDULES = ("funnel", "fixed", "reverse", "geometric")
FULL_EFFORT = 1_000_000  # more 2-opt moves than any tour here can need


@dataclass(frozen=True)
class Config:
    """Settings of a run. The defaults are those of the 2020 program."""

    generations: int = 300
    schedule: str = "funnel"  # how many are kept each generation
    base: int = 200  # funnel: size at the end. fixed: the size
    extra: int = 500  # funnel: additional individuals at the start
    init_size: int = 500
    crossover: str = "pmx"
    mutation: str = "swap"
    mutation_rate: float = 0.15
    effort_start: int = 0  # 2-opt moves allowed per child in the first generation
    effort_end: int = 0  # ... and in the last; interpolated in between
    budget: int = 0  # stop once this many tour evaluations' worth of work is done (0: never)

    def __post_init__(self):
        if self.crossover not in CROSSOVERS:
            raise ValueError(f"crossover must be one of {sorted(CROSSOVERS)}")
        if self.mutation not in MUTATIONS:
            raise ValueError(f"mutation must be one of {sorted(MUTATIONS)}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.generations < 1 or self.base < 2 or self.extra < 0 or self.init_size < 2:
            raise ValueError("generations >= 1, base >= 2, extra >= 0 and init_size >= 2 needed")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.effort_start < 0 or self.effort_end < 0 or self.budget < 0:
            raise ValueError("effort and budget must not be negative")

    def with_(self, **changes) -> "Config":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    def sizes(self) -> np.ndarray:
        """Individuals kept in each generation. Integer arithmetic only (JS does the same)."""
        g = np.arange(1, self.generations + 1, dtype=np.int64)
        total = self.generations
        if self.schedule == "fixed":
            out = np.full(total, self.base, dtype=np.int64)
        elif self.schedule == "funnel":  # the 2020 rule: base + extra * (1 - g / G)
            out = self.base + (self.extra * (total - g)) // total
        elif self.schedule == "reverse":
            out = self.base + (self.extra * (g - 1)) // total
        else:  # geometric: from base + extra down to base by a constant ratio
            top = self.base + self.extra
            out = np.floor(top * (self.base / top) ** ((g - 1) / max(total - 1, 1)) + 0.5)
        return np.asarray(out, dtype=np.int64)

    def effort(self) -> np.ndarray:
        """2-opt moves allowed per child in each generation."""
        g = np.arange(self.generations, dtype=np.int64)
        span = max(self.generations - 1, 1)
        return self.effort_start + ((self.effort_end - self.effort_start) * g) // span

    def evaluations(self) -> int:
        """Tour evaluations the run will make: the first population, then two per mating."""
        return int(self.init_size + 2 * self.sizes().sum())


@dataclass(frozen=True)
class Result:
    config: Config
    seed: int
    log: np.ndarray  # (generations run, 13) integers; columns are named in engine.py
    tour: np.ndarray  # best tour found, as city indices
    population: np.ndarray
    lengths: np.ndarray
    n: int
    optimum: int | None

    @property
    def best(self) -> np.ndarray:
        """Best length found so far, by generation."""
        return self.log[:, engine.BEST]

    @property
    def length(self) -> int:
        return int(self.best[-1])

    @property
    def gap(self) -> float:
        """Excess over the optimal tour, as a fraction."""
        return self.length / self.optimum - 1 if self.optimum else float("nan")

    @property
    def cost(self) -> np.ndarray:
        """Work done so far, in tour evaluations. A 2-opt check reads 4 distances, a tour n."""
        return self.log[:, engine.EVALS] + 4 * self.log[:, engine.CHECKS] / self.n

    def breeding(self) -> dict[str, np.ndarray]:
        """The run in a breeder's terms, by generation. Lengths: smaller is better.

        selection differential S = mean of the selected - mean of the population
        response R = mean of all offspring - mean of the population
        realised heritability = R / S
        intensity = S / phenotypic standard deviation
        kinship = share of edges the survivors have in common with the best one
        """
        log = self.log.astype(float)
        n_pop = log[:, engine.N_POP]
        kept = log[:, engine.KEPT]
        size = log[:, engine.MADE]
        mean = log[:, engine.SUM_POP] / n_pop
        sd = np.sqrt(np.maximum(log[:, engine.SUMSQ_POP] / n_pop - mean**2, 0))
        selected = log[:, engine.SUM_SEL] / kept
        offspring = log[:, engine.SUM_OFF] / (2 * size)
        differential = selected - mean
        response = offspring - mean
        with np.errstate(divide="ignore", invalid="ignore"):
            heritability = np.where(differential < 0, response / differential, np.nan)
            intensity = np.where(sd > 0, differential / sd, np.nan)
            kinship = np.where(kept > 1, log[:, engine.SHARED] / ((kept - 1) * self.n), np.nan)
        return {"mean": mean, "sd": sd, "selected": selected, "offspring": offspring,
                "differential": differential, "response": response,
                "heritability": heritability, "intensity": intensity, "kinship": kinship,
                "kept": kept, "size": size}


def run(instance: Instance, config: Config | None = None, seed: int = 1) -> Result:
    config = config or Config()
    if not 0 <= seed < 2**32:
        raise ValueError("seed must fit in 32 bits")
    log, tour, population, lengths = engine.run(
        instance.distance, instance.neighbours, config.sizes(), config.effort(),
        config.init_size, CROSSOVERS[config.crossover], MUTATIONS[config.mutation],
        config.mutation_rate, config.budget, seed)
    return Result(config, seed, log, tour, population, lengths, instance.n, instance.optimum)
