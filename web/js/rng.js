// Counter-based random numbers, bit-for-bit identical to src/tsp_ga/rng.py.
// A draw is a pure function of (seed, stream, counter) in 32-bit integer maths.

function mix(x) {
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

/** A uniform integer in [0, 2**32). */
export function uint32(seed, stream, counter) {
  let h = mix((seed ^ 0x9e3779b9) >>> 0);
  h = mix((h ^ Math.imul(stream, 0x85ebca77)) >>> 0);
  h = mix((h ^ Math.imul(counter, 0xc2b2ae3d)) >>> 0);
  return h;
}

/** A stream of draws: { seed, counter }. */
export const makeRng = (seed) => ({ seed: seed >>> 0, counter: 0 });

/** A uniform integer in [0, n). Exact for n below 2**21. */
export function randInt(rng, n) {
  const u = uint32(rng.seed, 0, rng.counter);
  rng.counter += 1;
  return Math.floor((u * n) / 4294967296);
}

/** A uniform float in (0, 1). */
export function randUnit(rng) {
  const u = uint32(rng.seed, 0, rng.counter);
  rng.counter += 1;
  return (u + 0.5) / 4294967296;
}
