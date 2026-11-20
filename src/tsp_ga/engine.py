"""The genetic algorithm, compiled with Numba.

Everything that decides the course of a run is integer arithmetic: distances
are TSPLIB integers and randomness comes from ``rng.py``. That makes a run
exactly reproducible in JavaScript, which ``web/test/engine.test.js`` checks
generation by generation. ``web/js/engine.js`` is a line-for-line port; keep
the two in step.

A generation, following the 2020 program:

1. Rank the population and keep the best ``sizes[g]`` (truncation selection).
2. ``sizes[g]`` times: pick two survivors at random, cross them into two
   offspring, mutate both with probability ``mutation_rate``, keep the shorter
   one. If ``effort[g] > 0`` improve it with up to that many 2-opt moves.
3. Survivors and children together are the next population.

``sizes`` and ``effort`` are arrays with one entry per generation, so a fixed
population, the funnel, and a funnel of local-search effort are all the same
code with different inputs.
"""

import numpy as np
from numba import njit

from .rng import rand_int, rand_unit

PMX, OX, ER = 0, 1, 2
SWAP, INVERSION = 0, 1

# Columns of the per-generation log.
(KEPT, EVALS, CHECKS, BEST, N_POP, SUM_POP, SUMSQ_POP, SUM_SEL, SUM_OFF, SUM_CHILD,
 SHARED, MOVES, MADE, N_COLS) = range(14)


@njit(cache=True)
def tour_length(distance, tour):
    n = len(tour)
    total = 0
    for i in range(n - 1):
        total += distance[tour[i], tour[i + 1]]
    return total + distance[tour[n - 1], tour[0]]


@njit(cache=True)
def random_tour(tour, seed, ctr):
    """Fisher-Yates shuffle of 0..n-1, in place."""
    n = len(tour)
    for i in range(n):
        tour[i] = i
    for i in range(n - 1, 0, -1):
        j = rand_int(seed, ctr, i + 1)
        tour[i], tour[j] = tour[j], tour[i]


@njit(cache=True)
def pmx(a, b, child, lo, hi, where):
    """Partially mapped crossover: ``a[lo:hi]`` in place, the rest from ``b``.

    Keeps cities at the *positions* they had in the parents.
    """
    n = len(a)
    for i in range(n):
        where[a[i]] = i
    for i in range(lo, hi):
        child[i] = a[i]
    for i in range(n):
        if lo <= i < hi:
            continue
        city = b[i]
        while lo <= where[city] < hi:
            city = b[where[city]]
        child[i] = city


@njit(cache=True)
def ox(a, b, child, lo, hi, where):
    """Order crossover: ``a[lo:hi]`` in place, the rest in the order they have in ``b``."""
    n = len(a)
    for i in range(n):
        where[a[i]] = i
    for i in range(lo, hi):
        child[i] = a[i]
    at = hi % n
    for k in range(n):
        city = b[(hi + k) % n]
        if lo <= where[city] < hi:
            continue
        child[at] = city
        at = (at + 1) % n


@njit(cache=True)
def _add_edge(adj, deg, u, v):
    for k in range(deg[u]):
        if adj[u, k] == v:
            return
    adj[u, deg[u]] = v
    deg[u] += 1


@njit(cache=True)
def er(a, b, child, start, seed, ctr, adj, deg, rest, slot):
    """Edge recombination: build the child from the parents' *edges*.

    From the current city go to the neighbour (in either parent) that has the
    fewest unused neighbours left; break ties at random; if there is none,
    jump to a random unused city.
    """
    n = len(a)
    for i in range(n):
        deg[i] = 0
        rest[i] = i
        slot[i] = i
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        _add_edge(adj, deg, a[i], a[j])
        _add_edge(adj, deg, a[j], a[i])
        _add_edge(adj, deg, b[i], b[j])
        _add_edge(adj, deg, b[j], b[i])
    left = n
    city = start
    for k in range(n):
        child[k] = city
        # Take the city off the list of unused cities (swap with the last).
        last = rest[left - 1]
        rest[slot[city]] = last
        slot[last] = slot[city]
        left -= 1
        # Take it off its neighbours' lists.
        for q in range(deg[city]):
            other = adj[city, q]
            for r in range(deg[other]):
                if adj[other, r] == city:
                    adj[other, r] = adj[other, deg[other] - 1]
                    deg[other] -= 1
                    break
        if left == 0:
            break
        fewest = 99
        ties = 0
        for q in range(deg[city]):
            d = deg[adj[city, q]]
            if d < fewest:
                fewest = d
                ties = 1
            elif d == fewest:
                ties += 1
        if ties == 0:
            city = rest[rand_int(seed, ctr, left)]
        else:
            pick = rand_int(seed, ctr, ties) if ties > 1 else 0
            for q in range(deg[city]):
                if deg[adj[city, q]] == fewest:
                    if pick == 0:
                        city = adj[city, q]
                        break
                    pick -= 1


@njit(cache=True)
def mutate(tour, kind, seed, ctr):
    n = len(tour)
    i = rand_int(seed, ctr, n)
    j = rand_int(seed, ctr, n)
    if kind == 0:  # swap two cities
        tour[i], tour[j] = tour[j], tour[i]
    else:  # reverse a stretch of the tour
        if i > j:
            i, j = j, i
        while i < j:
            tour[i], tour[j] = tour[j], tour[i]
            i += 1
            j -= 1


@njit(cache=True)
def _reverse(tour, pos, i, j):
    """Reverse the cyclic stretch of the tour from position i to position j."""
    n = len(tour)
    length = (j - i + n) % n + 1
    if 2 * length > n:  # reversing the rest of the cycle gives the same tour
        i, j = (j + 1) % n, (i - 1 + n) % n
        length = n - length
    for _ in range(length // 2):
        u = tour[i]
        v = tour[j]
        tour[i] = v
        pos[v] = i
        tour[j] = u
        pos[u] = j
        i = i + 1 if i + 1 < n else 0
        j = j - 1 if j > 0 else n - 1


@njit(cache=True)
def two_opt(tour, pos, distance, neighbours, max_moves, out):
    """Improve a tour in place with 2-opt moves restricted to near neighbours.

    Stops at a local optimum or after ``max_moves`` improving moves.
    ``out`` receives (change in length, moves made, moves examined).
    """
    n = len(tour)
    k_near = neighbours.shape[1]
    for i in range(n):
        pos[tour[i]] = i
    gain = 0
    moves = 0
    checks = 0
    improved = True
    while improved and moves < max_moves:
        improved = False
        for i in range(n):
            if moves >= max_moves:
                break
            a = tour[i]
            # Replace the edge to a's successor.
            b = tour[i + 1 if i + 1 < n else 0]
            d_ab = distance[a, b]
            done = False
            for k in range(k_near):
                c = neighbours[a, k]
                d_ac = distance[a, c]
                if d_ac >= d_ab:
                    break
                j = pos[c]
                d = tour[j + 1 if j + 1 < n else 0]
                checks += 1
                delta = d_ac + distance[b, d] - d_ab - distance[c, d]
                if delta < 0:
                    _reverse(tour, pos, i + 1 if i + 1 < n else 0, j)
                    gain += delta
                    moves += 1
                    improved = True
                    done = True
                    break
            if done:
                continue
            # Replace the edge to a's predecessor.
            b = tour[i - 1 if i > 0 else n - 1]
            d_ab = distance[a, b]
            for k in range(k_near):
                c = neighbours[a, k]
                d_ac = distance[a, c]
                if d_ac >= d_ab:
                    break
                j = pos[c]
                d = tour[j - 1 if j > 0 else n - 1]
                checks += 1
                delta = d_ac + distance[b, d] - d_ab - distance[c, d]
                if delta < 0:
                    _reverse(tour, pos, i, j - 1 if j > 0 else n - 1)
                    gain += delta
                    moves += 1
                    improved = True
                    break
    out[0] = gain
    out[1] = moves
    out[2] = checks


@njit(cache=True)
def cross(crossover, a, b, c1, c2, seed, ctr, where, adj, deg, rest, slot):
    """Make two offspring of ``a`` and ``b`` in ``c1`` and ``c2``."""
    n = len(a)
    if crossover == 2:
        er(a, b, c1, a[0], seed, ctr, adj, deg, rest, slot)
        er(b, a, c2, b[0], seed, ctr, adj, deg, rest, slot)
        return
    i = rand_int(seed, ctr, n)
    j = rand_int(seed, ctr, n)
    lo = min(i, j)
    hi = max(i, j) + 1
    if crossover == 0:
        pmx(a, b, c1, lo, hi, where)
        pmx(b, a, c2, lo, hi, where)
    else:
        ox(a, b, c1, lo, hi, where)
        ox(b, a, c2, lo, hi, where)


@njit(cache=True)
def shared_edges(tour, nxt, prv):
    """How many edges of ``tour`` are also in the tour described by nxt/prv."""
    n = len(tour)
    count = 0
    for i in range(n):
        u = tour[i]
        v = tour[i + 1 if i + 1 < n else 0]
        if nxt[u] == v or prv[u] == v:
            count += 1
    return count


@njit(cache=True)
def run(distance, neighbours, sizes, effort, init_size, crossover, mutation, mutation_rate,
        budget, seed):
    """Run the algorithm. Returns (log, best tour, final tours, final lengths).

    If ``budget`` is positive the run stops as soon as its cost reaches that
    many tour evaluations, where a 2-opt check counts as 4 / n of one.
    """
    n = distance.shape[0]
    generations = len(sizes)
    cap = init_size
    for g in range(generations):
        cap = max(cap, 2 * sizes[g])
    tours = np.empty((cap, n), dtype=np.int32)
    spare = np.empty((cap, n), dtype=np.int32)
    lengths = np.empty(cap, dtype=np.int64)
    spare_lengths = np.empty(cap, dtype=np.int64)
    c1 = np.empty(n, dtype=np.int32)
    c2 = np.empty(n, dtype=np.int32)
    where = np.empty(n, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    adj = np.empty((n, 4), dtype=np.int32)
    deg = np.empty(n, dtype=np.int32)
    rest = np.empty(n, dtype=np.int32)
    slot = np.empty(n, dtype=np.int32)
    nxt = np.empty(n, dtype=np.int32)
    prv = np.empty(n, dtype=np.int32)
    ls = np.zeros(3, dtype=np.int64)
    ctr = np.zeros(1, dtype=np.int64)
    log = np.zeros((generations, 13), dtype=np.int64)
    done = 0
    best = np.empty(n, dtype=np.int32)

    for r in range(init_size):
        random_tour(tours[r], seed, ctr)
        lengths[r] = tour_length(distance, tours[r])
    m = init_size
    evals = init_size
    checks = 0
    best_length = lengths[0]
    best[:] = tours[0]
    for r in range(1, m):
        if lengths[r] < best_length:
            best_length = lengths[r]
            best[:] = tours[r]

    for g in range(generations):
        size = sizes[g]
        order = np.argsort(lengths[:m], kind="mergesort")  # stable: ties keep their order
        kept = min(size, m)
        sum_pop = 0
        sumsq_pop = 0
        for r in range(m):
            sum_pop += lengths[r]
            sumsq_pop += lengths[r] * lengths[r]
        sum_sel = 0
        for r in range(kept):
            spare[r] = tours[order[r]]
            spare_lengths[r] = lengths[order[r]]
            sum_sel += spare_lengths[r]
        tours, spare = spare, tours
        lengths, spare_lengths = spare_lengths, lengths

        # Relatedness: edges the other survivors share with the best one.
        for i in range(n):
            u = tours[0, i]
            v = tours[0, i + 1 if i + 1 < n else 0]
            nxt[u] = v
            prv[v] = u
        shared = 0
        for r in range(1, kept):
            shared += shared_edges(tours[r], nxt, prv)

        sum_off = 0
        sum_child = 0
        moves = 0
        made = 0
        for c in range(size):
            if budget > 0 and evals * n + 4 * checks >= budget * n:
                break
            made += 1
            pa = rand_int(seed, ctr, kept)
            pb = rand_int(seed, ctr, kept)
            cross(crossover, tours[pa], tours[pb], c1, c2, seed, ctr, where, adj, deg, rest, slot)
            if rand_unit(seed, ctr) < mutation_rate:
                mutate(c1, mutation, seed, ctr)
                mutate(c2, mutation, seed, ctr)
            len1 = tour_length(distance, c1)
            len2 = tour_length(distance, c2)
            evals += 2
            sum_off += len1 + len2
            row = kept + c
            if len1 >= len2:
                tours[row] = c2
                lengths[row] = len2
            else:
                tours[row] = c1
                lengths[row] = len1
            if effort[g] > 0:
                two_opt(tours[row], pos, distance, neighbours, effort[g], ls)
                lengths[row] += ls[0]
                moves += ls[1]
                checks += ls[2]
            sum_child += lengths[row]
            if lengths[row] < best_length:
                best_length = lengths[row]
                best[:] = tours[row]
        m_before = m
        m = kept + made

        log[g, 0] = kept
        log[g, 1] = evals
        log[g, 2] = checks
        log[g, 3] = best_length
        log[g, 4] = m_before
        log[g, 5] = sum_pop
        log[g, 6] = sumsq_pop
        log[g, 7] = sum_sel
        log[g, 8] = sum_off
        log[g, 9] = sum_child
        log[g, 10] = shared
        log[g, 11] = moves
        log[g, 12] = made
        done = g + 1
        if made < size:
            break
    return log[:done], best, tours[:m].copy(), lengths[:m].copy()


@njit(cache=True)
def offspring_of(distance, tours, lengths, crossover, pairs, seed):
    """Cross random pairs without mutation or selection.

    Returns, per mating, the two parents' lengths and the two offspring's
    lengths, and how many of each offspring's edges came from a parent.
    Used to measure how heritable tour length is under each crossover.
    """
    m, n = tours.shape
    c1 = np.empty(n, dtype=np.int32)
    c2 = np.empty(n, dtype=np.int32)
    where = np.empty(n, dtype=np.int32)
    adj = np.empty((n, 4), dtype=np.int32)
    deg = np.empty(n, dtype=np.int32)
    rest = np.empty(n, dtype=np.int32)
    slot = np.empty(n, dtype=np.int32)
    nxt_a = np.empty(n, dtype=np.int32)
    prv_a = np.empty(n, dtype=np.int32)
    nxt_b = np.empty(n, dtype=np.int32)
    prv_b = np.empty(n, dtype=np.int32)
    ctr = np.zeros(1, dtype=np.int64)
    out = np.empty((pairs, 5), dtype=np.int64)
    for k in range(pairs):
        pa = rand_int(seed, ctr, m)
        pb = rand_int(seed, ctr, m)
        cross(crossover, tours[pa], tours[pb], c1, c2, seed, ctr, where, adj, deg, rest, slot)
        for i in range(n):
            j = i + 1 if i + 1 < n else 0
            nxt_a[tours[pa, i]] = tours[pa, j]
            prv_a[tours[pa, j]] = tours[pa, i]
            nxt_b[tours[pb, i]] = tours[pb, j]
            prv_b[tours[pb, j]] = tours[pb, i]
        inherited = 0
        for child in (c1, c2):
            for i in range(n):
                u = child[i]
                v = child[i + 1 if i + 1 < n else 0]
                if nxt_a[u] == v or prv_a[u] == v or nxt_b[u] == v or prv_b[u] == v:
                    inherited += 1
        out[k, 0] = lengths[pa]
        out[k, 1] = lengths[pb]
        out[k, 2] = tour_length(distance, c1)
        out[k, 3] = tour_length(distance, c2)
        out[k, 4] = inherited
    return out


@njit(cache=True)
def restarts(distance, neighbours, budget, seed):
    """No breeding at all: 2-opt from fresh random tours until the budget is spent.

    Returns (best length, cost in tour evaluations, number of starts).
    """
    n = distance.shape[0]
    tour = np.empty(n, dtype=np.int32)
    pos = np.empty(n, dtype=np.int32)
    ls = np.zeros(3, dtype=np.int64)
    ctr = np.zeros(1, dtype=np.int64)
    best = -1
    cost = 0.0
    starts = 0
    while cost < budget:
        random_tour(tour, seed, ctr)
        length = tour_length(distance, tour)
        two_opt(tour, pos, distance, neighbours, 1000000, ls)
        length += ls[0]
        cost += 1 + 4 * ls[2] / n
        starts += 1
        if best < 0 or length < best:
            best = length
    return best, cost, starts
