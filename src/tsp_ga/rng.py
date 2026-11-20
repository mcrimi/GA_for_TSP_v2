"""Counter-based random numbers that are bit-for-bit identical in JavaScript.

A draw is a pure function of ``(seed, stream, counter)``: three rounds of the
MurmurHash3 32-bit finaliser. Everything is done in 64-bit integers masked to
32 bits, which Numba compiles to plain machine arithmetic and which
``Math.imul`` reproduces exactly in the browser (see ``web/js/rng.js``).
"""

from numba import njit

MASK = 0xFFFFFFFF


@njit(cache=True)
def _mix(x):
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


@njit(cache=True)
def uint32(seed, stream, counter):
    """A uniform integer in [0, 2**32)."""
    h = _mix((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
    h = _mix(h ^ ((stream * 0x85EBCA77) & 0xFFFFFFFF))
    h = _mix(h ^ ((counter * 0xC2B2AE3D) & 0xFFFFFFFF))
    return h


@njit(cache=True)
def rand_int(seed, ctr, n):
    """A uniform integer in [0, n). ``ctr`` is a one-element array, advanced by one."""
    u = uint32(seed, 0, ctr[0])
    ctr[0] += 1
    return (u * n) >> 32


@njit(cache=True)
def rand_unit(seed, ctr):
    """A uniform float in (0, 1)."""
    u = uint32(seed, 0, ctr[0])
    ctr[0] += 1
    return (u + 0.5) / 4294967296.0
