// Canvas drawings for the live run: the map with the best tour, the progress
// curve, and the funnel. Ink and paper are read from CSS at draw time, so the
// drawings follow the reader's choice of paper.

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  return { ink: get("--ink"), ink2: get("--ink-2"), line: get("--line"), paper: get("--paper"), sans: get("--sans") };
}

function prepare(canvas, ratio) {
  const width = Math.max(240, Math.round(canvas.getBoundingClientRect().width || 480));
  const height = Math.round(width * ratio);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const c = palette();
  ctx.fillStyle = c.paper;
  ctx.fillRect(0, 0, width, height);
  return { ctx, width, height, c };
}

/**
 * Fit the cities into a box, north up. The files give (latitude, longitude)
 * in thousandths of a degree, so longitude runs across and latitude up.
 */
export function projection(coords, width, height, pad = 16) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [lat, lon] of coords) {
    minX = Math.min(minX, lon);
    maxX = Math.max(maxX, lon);
    minY = Math.min(minY, lat);
    maxY = Math.max(maxY, lat);
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  const offX = (width - spanX * scale) / 2;
  const offY = (height - spanY * scale) / 2;
  return ([lat, lon]) => [offX + (lon - minX) * scale, height - offY - (lat - minY) * scale];
}

/** The cities and the best tour found so far. */
export function drawMap(canvas, coords, tour) {
  const { ctx, width, height, c } = prepare(canvas, 1);
  const at = projection(coords, width, height);
  const pts = coords.map(at);
  if (tour) {
    ctx.strokeStyle = c.ink;
    ctx.lineWidth = 1;
    ctx.lineJoin = "miter";
    ctx.beginPath();
    for (let i = 0; i <= tour.length; i += 1) {
      const [x, y] = pts[tour[i % tour.length]];
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  const side = coords.length > 100 ? 3 : 5;
  ctx.fillStyle = c.ink;
  for (const [x, y] of pts) ctx.fillRect(Math.round(x - side / 2), Math.round(y - side / 2), side, side);
}

const FLOOR = 0.0005; // gaps below 0.05% are drawn on the bottom line
const CEILING = 20;

/**
 * Excess over the optimal tour (log scale) against work done.
 * curves: [{ cost: [...], gap: [...], dashed }]
 */
export function drawProgress(canvas, curves, budget) {
  const { ctx, width, height, c } = prepare(canvas, 0.79);
  const m = { top: 12, right: 14, bottom: 34, left: 46 };
  const sx = (v) => m.left + (Math.min(v, budget) / budget) * (width - m.left - m.right);
  const lo = Math.log(FLOOR);
  const span = Math.log(CEILING) - lo;
  const sy = (v) => height - m.bottom - ((Math.log(Math.min(Math.max(v, FLOOR), CEILING)) - lo) / span) * (height - m.top - m.bottom);

  ctx.font = `11px ${c.sans}`;
  ctx.fillStyle = c.ink2;
  ctx.strokeStyle = c.line;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (const [v, label] of [[10, "1000%"], [1, "100%"], [0.1, "10%"], [0.01, "1%"], [0.001, "0.1%"], [FLOOR, "optimal"]]) {
    const y = Math.round(sy(v)) + 0.5;
    if (v > FLOOR) { ctx.beginPath(); ctx.moveTo(m.left, y); ctx.lineTo(width - m.right, y); ctx.stroke(); }
    ctx.fillText(label, m.left - 6, y + 4);
  }
  ctx.textAlign = "center";
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const v = budget * f;
    ctx.fillText(f === 0 ? "0" : `${Math.round(v / 1000)}k`, sx(v), height - m.bottom + 16);
  }
  ctx.textAlign = "left";
  ctx.fillText("Work done, in tour evaluations", m.left, height - 4);

  ctx.strokeStyle = c.ink;
  ctx.beginPath();
  ctx.moveTo(m.left + 0.5, m.top);
  ctx.lineTo(m.left + 0.5, height - m.bottom + 0.5);
  ctx.lineTo(width - m.right, height - m.bottom + 0.5);
  ctx.stroke();

  for (const curve of curves) {
    if (!curve || !curve.cost.length) continue;
    ctx.lineWidth = curve.dashed ? 1 : 1.75;
    ctx.setLineDash(curve.dashed ? [5, 4] : []);
    ctx.beginPath();
    for (let i = 0; i < curve.cost.length; i += 1) {
      const x = sx(curve.cost[i]);
      const y = sy(curve.gap[i]);
      if (i === 0) ctx.moveTo(x, y);
      else { ctx.lineTo(x, sy(curve.gap[i - 1])); ctx.lineTo(x, y); } // the best changes in steps
    }
    ctx.lineTo(sx(curve.costNow ?? curve.cost[curve.cost.length - 1]), sy(curve.gap[curve.gap.length - 1]));
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

/** The population kept in each generation, as a funnel on its side, filled up to now. */
export function drawFunnel(canvas, sizes, generation) {
  const { ctx, width, height, c } = prepare(canvas, 0.11);
  const peak = Math.max(...sizes);
  const mid = height / 2;
  const half = (v) => Math.max(1, (v / peak) * (height / 2 - 4));
  const n = sizes.length;
  const x = (g) => 1 + (g / Math.max(n - 1, 1)) * (width - 2);
  const outline = (upto) => {
    ctx.beginPath();
    for (let px = 0; px <= upto; px += 1) {
      const g = Math.min(n - 1, Math.round((px / Math.max(width - 2, 1)) * (n - 1)));
      const y = mid - half(sizes[g]);
      if (px === 0) ctx.moveTo(1 + px, y);
      else ctx.lineTo(1 + px, y);
    }
    for (let px = upto; px >= 0; px -= 1) {
      const g = Math.min(n - 1, Math.round((px / Math.max(width - 2, 1)) * (n - 1)));
      ctx.lineTo(1 + px, mid + half(sizes[g]));
    }
    ctx.closePath();
  };
  const done = Math.round(x(Math.min(generation, n - 1)) - 1);
  if (generation > 0) {
    ctx.fillStyle = c.ink;
    outline(done);
    ctx.fill();
  }
  ctx.strokeStyle = c.ink;
  ctx.lineWidth = 1;
  outline(width - 2);
  ctx.stroke();
}
