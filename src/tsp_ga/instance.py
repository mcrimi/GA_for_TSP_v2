"""Travelling salesman instances in TSPLIB format."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"

#: Proven optimal tour lengths (Concorde; math.uwaterloo.ca/tsp/world/summary.html).
OPTIMA = {"dj38": 6656, "qa194": 9352, "uy734": 79114}
TITLES = {"dj38": "Djibouti, 38 cities", "qa194": "Qatar, 194 cities",
          "uy734": "Uruguay, 734 cities"}
NEIGHBOURS = 8


@dataclass(frozen=True)
class Instance:
    name: str
    coords: np.ndarray  # (n, 2) float64, as written in the file
    distance: np.ndarray  # (n, n) int32, TSPLIB EUC_2D: rounded to the nearest integer
    neighbours: np.ndarray  # (n, k) int32: each city's nearest cities, nearest first
    optimum: int | None = None

    @property
    def n(self) -> int:
        return len(self.coords)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB ``EUC_2D`` distances: floor(sqrt(dx^2 + dy^2) + 0.5)."""
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    return np.floor(np.sqrt(dx * dx + dy * dy) + 0.5).astype(np.int32)


def nearest_neighbours(distance: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
    """Each city's ``k`` nearest other cities; ties go to the lower index."""
    n = len(distance)
    k = min(k, n - 1)
    out = np.empty((n, k), dtype=np.int32)
    for city in range(n):
        d = distance[city].astype(np.int64)
        d[city] = np.iinfo(np.int64).max
        out[city] = np.argsort(d, kind="stable")[:k]
    return out


def from_coords(name: str, coords, optimum: int | None = None) -> Instance:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 4:
        raise ValueError("an instance needs at least four cities with two coordinates each")
    distance = distance_matrix(coords)
    return Instance(name, coords, distance, nearest_neighbours(distance), optimum)


def parse_tsplib(text: str) -> tuple[str, np.ndarray]:
    """Read the name and coordinates from a TSPLIB file with EUC_2D node coordinates."""
    name, kind, rows, in_coords = "", "", [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if in_coords:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"cannot read coordinate line: {raw!r}")
            rows.append((float(parts[1]), float(parts[2])))
        elif line.startswith("NODE_COORD_SECTION"):
            in_coords = True
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "NAME":
                name = value
            elif key == "EDGE_WEIGHT_TYPE":
                kind = value
    if kind != "EUC_2D":
        raise ValueError(f"only EUC_2D instances are supported, not {kind or 'unspecified'!r}")
    return name, np.array(rows, dtype=np.float64)


def load(name: str) -> Instance:
    """Load one of the instances shipped in ``data/`` by name, e.g. ``"dj38"``."""
    path = DATA / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(f"no instance called {name!r} in {DATA}")
    _, coords = parse_tsplib(path.read_text())
    return from_coords(name, coords, OPTIMA.get(name))
