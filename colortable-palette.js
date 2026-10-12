// Colortable model shared by the editor, the WebGL lookup texture, and Python export.
// A palette is { vmin, vmax, mode: 'linear' | 'listed', n, points: [{ t, color, x? }] }.
// t is °C; x (optional) is the original normalized position from satcmaps.py, kept so
// exported presets reproduce the same floating-point bins at sharp transitions.

export function validate(p) {
  if (!p || !Number.isFinite(p.vmin) || !Number.isFinite(p.vmax) || p.vmin >= p.vmax)
    throw Error('Minimum must be less than maximum.');
  if (!Array.isArray(p.points) || p.points.length < 2 || p.points.length > 2048)
    throw Error('Use 2–2048 breakpoints.');
  for (const x of p.points) {
    if (!Number.isFinite(x.t) || x.t < p.vmin - 1e-6 || x.t > p.vmax + 1e-6 || !/^#[0-9a-f]{6}$/i.test(x.color))
      throw Error('Each breakpoint needs a temperature within the range and a six-digit hex color.');
  }
  if (!['linear', 'listed'].includes(p.mode)) throw Error('Unknown interpolation mode.');
  const s = sorted(p);
  if (p.mode === 'linear' && (Math.abs(s[0].t - p.vmin) > 1e-6 || Math.abs(s.at(-1).t - p.vmax) > 1e-6))
    throw Error('First and last breakpoints must match MIN and MAX.');
}

export const sorted = p => p.points.map(x => ({ ...x })).sort((a, b) => a.t - b.t);

// Normalized 0–1 position; prefers the original x when it still matches t.
export const position = (p, x) =>
  Number.isFinite(x.x) && Math.abs(p.vmin + x.x * (p.vmax - p.vmin) - x.t) < 1e-9
    ? x.x
    : (x.t - p.vmin) / (p.vmax - p.vmin);

export const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
export const hex = values => '#' + values.map(x => Math.round(x).toString(16).padStart(2, '0')).join('');

// Linear RGB interpolation. With duplicate temperatures, the first point is reached from the
// cold side and the second continues toward the warm side (a sharp jump).
export function colorAt(points, t) {
  if (t <= points[0].t) return rgb(points[0].color);
  if (t >= points.at(-1).t) return rgb(points.at(-1).color);
  const j = points.findIndex(x => x.t >= t), a = points[j - 1], b = points[j];
  const f = b.t === a.t ? 1 : (t - a.t) / (b.t - a.t);
  const ra = rgb(a.color), rb = rgb(b.color);
  return ra.map((v, i) => v + (rb[i] - v) * f);
}

// RGBA lookup table with the same bin count matplotlib uses (N, or one bin per listed color).
export function makeLUT(p) {
  validate(p);
  const pts = sorted(p).map(x => ({ ...x, t: position(p, x) }));
  const n = p.mode === 'listed' ? pts.length : (p.n || 256);
  const lut = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const c = p.mode === 'listed' ? rgb(pts[i].color) : colorAt(pts, i / (n - 1));
    lut.set([...c.map(Math.round), 255], i * 4);
  }
  return lut;
}

const PY_KEYWORDS = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class',
  'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import',
  'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'];

// satcmaps.py-style function returning newcmp, vmax, vmin.
export function pythonCode(p, name) {
  validate(p);
  if (!/^[a-zA-Z_]\w*$/.test(name) || PY_KEYWORDS.includes(name)) throw Error('Use a valid Python function name.');
  const pts = sorted(p), fmt = n => Number(n.toFixed(8)).toString();
  let s = `def ${name}():\n    vmin = ${fmt(p.vmin)}\n    vmax = ${fmt(p.vmax)}\n`;
  if (p.mode === 'listed') {
    s += '    newcmp = ListedColormap([\n' + pts.map(x => `        "${x.color}",`).join('\n') + `\n    ], name="${name}")\n`;
  } else {
    const pos = x => Number.isFinite(x.x) && position(p, x) === x.x ? String(x.x) : `(${fmt(x.t)} - vmin) / (vmax - vmin)`;
    s += `    newcmp = LinearSegmentedColormap.from_list("${name}", [\n`
      + pts.map(x => `        (${pos(x)}, "${x.color}"),`).join('\n')
      + `\n    ], N=${p.n || 256})\n`;
  }
  return s + '    return newcmp, vmax, vmin\n';
}
