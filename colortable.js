import { validate, sorted, hex, colorAt, makeLUT, pythonCode } from './colortable-palette.js';
import { Viewer, latlon } from './colortable-viewer.js';

// ?api=http://127.0.0.1:7860 points the page at a local api.py for testing
const LOCAL_API = new URLSearchParams(location.search).get('api');
const API_BASE = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(LOCAL_API || '') ? LOCAL_API : 'https://tcalert-cycloweb2.hf.space';
const FIRST_LOAD_PX = 4096;  // longest side of the first image; zooming in loads finer pixels for the view
const EXPORT_PX = 8192;      // longest side of a full-resolution copy / save
const STORE_KEY = 'cw_ct_palettes';

const $ = id => document.getElementById(id);
const clone = x => JSON.parse(JSON.stringify(x));
const val = id => $(id).value.trim();

let presets = {};          // { '13': [...IR tables], '9': [...WV tables], vis: [...reflectance tables] }
let kind = '13';           // which preset group the editor is showing
let palette = null;
let viewer = null;
let current = null;        // metadata of the displayed image
let request = null;        // parameters of the displayed image, reused for zoom-in requests
let selected = -1;
let undoStack = [], redoStack = [];
let lut, drawQueued = false, fullDisk = false, fullRes = false;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch { }

// ---------------------------------------------------------------------------
// Status helpers (same convention as index.html)
// ---------------------------------------------------------------------------

function setStatus(type, msg) {
  $('ct-status').className = 'status ' + type;
  $('ct-status').textContent = msg;
}

async function callAPI(endpoint, params = {}) {
  const url = new URL(API_BASE + endpoint);
  Object.entries(params).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== '') url.searchParams.set(k, v); });
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = Array.isArray(body.detail) ? body.detail.map(d => d.msg || JSON.stringify(d)).join('; ') : (body.detail || res.statusText);
    throw new Error(detail);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Rendering the palette
// ---------------------------------------------------------------------------

function remember() {
  undoStack.push(clone(palette));
  if (undoStack.length > 100) undoStack.shift();
  redoStack = [];
}

function persist() {
  saved[kind] = clone(palette);
  try { localStorage.setItem(STORE_KEY, JSON.stringify(saved)); } catch { }
}

function schedule() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; update(); });
}

function update() {
  try {
    validate(palette);
    $('ct-validation').textContent = '';
    lut = makeLUT(palette);
    viewer?.setPalette(lut, palette.vmin, palette.vmax);
    drawRamps();
    persist();
    $('ct-dirty').textContent = palette.custom ? 'EDITED' : 'PRESET';
    $('ct-dirty').classList.toggle('edited', !!palette.custom);
  } catch (e) {
    $('ct-validation').textContent = e.message;
  }
  $('ct-undo').disabled = !undoStack.length;
  $('ct-redo').disabled = !redoStack.length;
}

// Temperatures are edited in °C to 0.1; reflectances are unitless and edited to 0.001
const isVis = () => kind === 'vis';
const unit = () => isVis() ? '' : '°C';
const digits = () => isVis() ? 3 : 1;
const roundT = t => Number(t.toFixed(digits()));
const label = t => `${t}${unit()}`;

function niceStep(span) {
  for (const s of [.01, .02, .05, .1, .2, .25, .5, 1, 2, 5, 10, 20, 25, 50, 100]) if (span / s <= 10) return s;
  return 200;
}

// Colorbar tick values (shared by the on-screen colorbar and the PNG export)
function tickValues() {
  const { vmin, vmax } = palette, step = niceStep(vmax - vmin), out = [];
  for (let k = Math.ceil(vmin / step - 1e-9); k * step <= vmax + 1e-9; k++) out.push(Number((k * step).toFixed(4)));
  return out;
}

function drawRamps() {
  for (const id of ['ct-legend', 'ct-ramp']) {
    const c = $(id);
    c.width = Math.max(256, Math.round(c.clientWidth * (devicePixelRatio || 1)));
    c.height = 1;
    const ctx = c.getContext('2d'), im = ctx.createImageData(c.width, 1), n = lut.length / 4;
    for (let x = 0; x < c.width; x++) {
      const i = Math.min(n - 1, Math.floor(x / c.width * n)) * 4;
      im.data.set(lut.subarray(i, i + 4), x * 4);
    }
    ctx.putImageData(im, 0, 0);
  }
  const { vmin, vmax } = palette;
  $('ct-ticks').replaceChildren(...tickValues().map(t => {
    const s = document.createElement('span');
    s.style.left = 100 * (t - vmin) / (vmax - vmin) + '%';
    s.textContent = label(t);
    return s;
  }));
  for (const h of $('ct-handles').children) {
    const p = palette.points[Number(h.dataset.index)];
    h.style.left = 100 * (p.t - vmin) / (vmax - vmin) + '%';
  }
}

// ---------------------------------------------------------------------------
// Breakpoint editor
// ---------------------------------------------------------------------------

function edited() { palette.custom = true; }

function editor() {
  if (!palette) return;
  $('ct-vmin').value = palette.vmin;
  $('ct-vmax').value = palette.vmax;
  $('ct-vmin').step = $('ct-vmax').step = isVis() ? '0.01' : '0.1';
  $('ct-vmin-label').textContent = `MIN ${unit()}`.trim();
  $('ct-vmax-label').textContent = `MAX ${unit()}`.trim();
  $('ct-mode').value = palette.mode;
  const rows = [], handles = [];

  palette.points.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'point' + (i === selected ? ' selected' : '');
    // One undo step per focused field, taken on its first change
    let touched = false;
    const touch = () => { if (!touched) { remember(); touched = true; } };
    const untouch = () => { touched = false; };

    const temp = document.createElement('input');
    temp.type = 'number';
    temp.step = isVis() ? '0.01' : '0.1';
    temp.value = Number(p.t.toFixed(digits() + 1));
    temp.title = isVis() ? 'Reflectance' : 'Temperature (°C)';
    temp.onfocus = untouch;
    temp.oninput = () => { touch(); delete p.x; p.t = temp.value === '' ? NaN : Number(temp.value); edited(); schedule(); };
    temp.onchange = () => { if (Number.isFinite(p.t)) { palette.points.sort((a, b) => a.t - b.t); editor(); } };

    const swatch = document.createElement('input');
    swatch.type = 'color';
    swatch.value = p.color;
    swatch.onfocus = untouch;
    swatch.oninput = () => { touch(); p.color = swatch.value; text.value = p.color; edited(); schedule(); };

    const text = document.createElement('input');
    text.value = p.color;
    text.spellcheck = false;
    text.title = 'Hex color';
    text.onfocus = untouch;
    text.oninput = () => {
      if (/^#[0-9a-f]{6}$/i.test(text.value)) { touch(); p.color = text.value.toLowerCase(); swatch.value = p.color; edited(); schedule(); }
      else $('ct-validation').textContent = 'Hex colors need six digits, e.g. #405580.';
    };

    const dup = document.createElement('button');
    dup.textContent = '⧉';
    dup.title = 'Duplicate (same value = sharp jump)';
    dup.onclick = () => { remember(); palette.points.splice(i + 1, 0, { ...p }); edited(); selected = i + 1; editor(); };

    const del = document.createElement('button');
    del.textContent = '×';
    del.title = 'Remove';
    del.onclick = () => { if (palette.points.length <= 2) return; remember(); palette.points.splice(i, 1); edited(); selected = -1; editor(); };

    row.append(temp, swatch, text, dup, del);
    rows.push(row);

    const h = document.createElement('button');
    h.className = 'handle' + (i === selected ? ' selected' : '');
    h.dataset.index = i;
    h.title = `${label(roundT(p.t))} · ${p.color}`;
    h.onpointerdown = e => {
      e.preventDefault();
      remember();
      h.setPointerCapture(e.pointerId);
      selected = i;
      h.onpointermove = ev => {
        const r = $('ct-ramp').getBoundingClientRect();
        const t = palette.vmin + Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * (palette.vmax - palette.vmin);
        delete p.x;
        // Endpoints stay pinned to MIN/MAX in linear mode
        const last = palette.points.length - 1;
        p.t = palette.mode === 'linear' && i === 0 ? palette.vmin
          : palette.mode === 'linear' && i === last ? palette.vmax
          : roundT(t);
        temp.value = p.t;
        edited();
        schedule();
      };
      h.onpointerup = () => { h.onpointermove = null; palette.points.sort((a, b) => a.t - b.t); editor(); };
    };
    handles.push(h);
  });

  $('ct-points').replaceChildren(...rows);
  $('ct-handles').replaceChildren(...handles);
  update();
}

function addPoint(t) {
  try {
    validate(palette);
    remember();
    const pts = sorted(palette);
    if (t === undefined) {
      const gaps = pts.slice(1).map((p, i) => ({ size: p.t - pts[i].t, t: (p.t + pts[i].t) / 2 }));
      t = gaps.sort((a, b) => b.size - a.size)[0].t;
    }
    palette.points.push({ t: roundT(t), color: hex(colorAt(pts, t)) });
    palette.points.sort((a, b) => a.t - b.t);
    edited();
    editor();
  } catch (e) { $('ct-validation').textContent = e.message; }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

function presetOptions() {
  const group = presets[kind] || [], q = val('ct-search').toLowerCase();
  $('ct-preset').replaceChildren(...group.filter(p => p.name.toLowerCase().includes(q)).map(p => {
    const o = document.createElement('option');
    o.value = o.textContent = p.name;
    return o;
  }));
  $('ct-preset-info').textContent = isVis() ? `${group.length} built-in reflectance tables`
    : `${group.length} ${kind === '13' ? 'IR' : 'WV'} tables in satcmaps.py`;
}

function applyPreset() {
  const p = (presets[kind] || []).find(x => x.name === $('ct-preset').value);
  if (!p) return;
  if (palette) remember();
  palette = clone(p);
  palette.custom = false;
  selected = -1;
  editor();
}

// Switch the editor between the IR, WV and reflectance collections, restoring that group's last palette.
function setKind(next) {
  if (next === kind && palette) return;
  kind = next;
  undoStack = []; redoStack = [];
  $('ct-search').value = '';
  presetOptions();
  const group = presets[kind] || [];
  if (!group.length) return;
  const fallback = { '13': 'irg', '9': 'wv', vis: 'gray' }[kind];
  $('ct-preset').value = group.some(x => x.name === fallback) ? fallback : group[0].name;
  const name = val('ct-name'), names = { '13': 'custom_ir', '9': 'custom_wv', vis: 'custom_vis' };
  if (!name || Object.values(names).includes(name)) $('ct-name').value = names[kind];
  try {
    validate(saved[kind]);
    palette = clone(saved[kind]);
    if (group.some(x => x.name === palette.name)) $('ct-preset').value = palette.name;
    editor();
  } catch {
    palette = null;
    applyPreset();
  }
}

// Reflectance bands (vis; 1–6 on GOES-R / Himawari; 1 on GridSat) use the reflectance tables,
// water-vapor bands (8–10 on GOES-R / Himawari, 3 on GridSat) the WV tables
function kindFor(satellite, band) {
  const s = satellite.toUpperCase(), n = parseInt(band);
  const gridsat = s === 'B1' || (/^\d+$/.test(s) && Number(s) >= 8 && Number(s) <= 15);
  if (String(band).trim().toLowerCase() === 'vis' || (gridsat ? n === 1 : n >= 1 && n <= 6)) return 'vis';
  return (gridsat ? n === 3 : [8, 9, 10].includes(n)) ? '9' : '13';
}
const fieldKind = () => kindFor(val('ct-satellite') || '19', val('ct-band'));

// ---------------------------------------------------------------------------
// Loading imagery
// ---------------------------------------------------------------------------

async function run() {
  if (!viewer) { setStatus('error', 'WebGL 2 is required to display imagery.'); return; }
  const btn = $('ct-run');
  if (!val('ct-date') || !val('ct-time')) { setStatus('error', 'DATE and TIME are required.'); return; }
  if (!fullDisk && /^-?\d/.test(val('ct-zoom') || '2') && (!val('ct-lat') || !val('ct-lon'))) {
    setStatus('error', 'LAT and LON are required (or use a region key in ZOOM, or FULL DISK).');
    return;
  }
  btn.disabled = true;
  setStatus('loading', 'working...');
  try {
    const params = { satellite: val('ct-satellite') || '19', date: val('ct-date'), time: val('ct-time'), band: val('ct-band') };
    const meta = await callAPI('/colortable/load', {
      ...params,
      lat: val('ct-lat'),
      lon: val('ct-lon'),
      zoom: val('ct-zoom'),
      full: fullDisk ? 'true' : null,
      maxpx: Math.min(viewer.maxTexture, FIRST_LOAD_PX),
    });
    setStatus('loading', `transferring ${meta.width} × ${meta.height} px...`);
    const data = await pixels(meta);
    refineToken++;
    setKind(meta.kind);
    viewer.setImage(data, meta);
    current = meta;
    request = params;
    $('ct-placeholder').hidden = true;
    showTitle();
    $('ct-title').style.color = '#111';
    $('ct-source').hidden = !meta.source;
    if (meta.source) $('ct-source').href = meta.source;
    update();
    updateHash();
    FIELDS.forEach(id => { try { localStorage.setItem('cw_' + id, $(id).value); } catch { } });
    setStatus('success', 'done.');
  } catch (e) {
    setStatus('error', e.message);
  } finally {
    btn.disabled = false;
  }
}

async function pixels(meta) {
  const res = await fetch(`${API_BASE}/colortable/frame/${meta.frame}`);
  if (!res.ok) throw Error('Image expired on the server; run again.');
  const buffer = await res.arrayBuffer();
  if (buffer.byteLength !== meta.width * meta.height * 2) throw Error('Incomplete transfer; run again.');
  return new Uint16Array(buffer);
}

// Title: what is shown and at what resolution (every Nth native pixel until zoomed in far enough)
function showTitle(note = '') {
  const m = current, stride = viewer.detail ? viewer.detail.meta.stride : m.stride || 1;
  const res = stride > 1 ? `every ${stride}${[, 'st', 'nd', 'rd'][stride] || 'th'} native px` : 'native px';
  $('ct-title').textContent = `${m.title} · ${m.scan_time} · ${res}${note}`;
}

// When the view shows the image coarser than the screen can, load the visible area (plus a margin)
// at the resolution the screen needs, and draw it over the overview. So resolution follows zoom.
let refineTimer = null, refineToken = 0;

function scheduleRefine() {
  clearTimeout(refineTimer);
  refineTimer = setTimeout(refine, 350);
}

async function refine() {
  const m = current;
  if (!m || !m.refinable || !viewer?.hasImage || (m.stride || 1) <= 1) return;
  const [vx, vy, vw, vh] = viewer.view, s = m.stride, [C0, R0] = m.crop;
  const need = Math.max(1, Math.ceil(Math.max(vw * s / viewer.canvas.width, vh * s / viewer.canvas.height)));
  const d = viewer.detail;
  if (need >= s) {   // zoomed out: the overview is already as sharp as the screen
    if (d) { viewer.clearDetail(); showTitle(); }
    return;
  }
  if (d && d.meta.stride <= need && vx >= d.rect[0] && vy >= d.rect[1]
      && vx + vw <= d.rect[0] + d.rect[2] && vy + vh <= d.rect[1] + d.rect[3]) return;   // already covered
  const pad = .25;
  const c0 = Math.max(0, Math.floor(C0 + (vx - vw * pad) * s)), c1 = Math.min(m.grid_width, Math.ceil(C0 + (vx + vw * (1 + pad)) * s));
  const r0 = Math.max(0, Math.floor(R0 + (vy - vh * pad) * s)), r1 = Math.min(m.grid_height, Math.ceil(R0 + (vy + vh * (1 + pad)) * s));
  const token = ++refineToken;
  showTitle(' · sharpening…');
  try {
    const maxpx = Math.min(viewer.maxTexture, Math.ceil(Math.max(viewer.canvas.width, viewer.canvas.height) * (1 + 2 * pad)));
    const meta = await callAPI('/colortable/load', { ...request, rect: [c0, r0, c1, r1].join(','), maxpx });
    const data = await pixels(meta);
    if (token !== refineToken || current !== m) return;   // the view or image changed meanwhile
    const [dc0, dr0] = meta.crop;
    viewer.setDetail(data, meta, [(dc0 - C0) / s, (dr0 - R0) / s, meta.width * meta.stride / s, meta.height * meta.stride / s]);
  } catch (e) {
    if (token === refineToken) setStatus('error', e.message);
  }
  if (token === refineToken) showTitle();
}

// ---------------------------------------------------------------------------
// Inputs: localStorage + shareable hash, as on the main page
// ---------------------------------------------------------------------------

const FIELDS = ['ct-satellite', 'ct-date', 'ct-time', 'ct-lat', 'ct-lon', 'ct-band', 'ct-zoom'];

function updateHash() {
  const p = new URLSearchParams();
  FIELDS.forEach(id => { if (val(id)) p.set(id, val(id)); });
  if (fullDisk) p.set('full', '1');
  history.replaceState(null, '', location.pathname + location.search + '#' + p.toString());
}

function restoreInputs() {
  const hash = new URLSearchParams(location.hash.slice(1));
  const fromHash = FIELDS.some(id => hash.has(id));
  FIELDS.forEach(id => {
    let v = null;
    if (fromHash) v = hash.get(id);
    else { try { v = localStorage.getItem('cw_' + id); } catch { } }
    if (v !== null) $(id).value = v;
  });
  if (fromHash && hash.get('full') === '1') setFull(true);
}

function setFull(on) {
  if (on && !viewer) return;
  fullDisk = on;
  $('ct-full').classList.toggle('active', on);
  for (const id of ['ct-lat', 'ct-lon', 'ct-zoom']) $(id).disabled = on;
}

function download(blob, name) {
  const a = document.createElement('a'), url = URL.createObjectURL(blob);
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$('ct-run').onclick = run;
FIELDS.forEach(id => {
  $(id).addEventListener('input', () => { try { localStorage.setItem('cw_' + id, $(id).value); } catch { } });
  $(id).addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
});
for (const id of ['ct-satellite', 'ct-band']) $(id).addEventListener('input', () => { if (presets[fieldKind()]) setKind(fieldKind()); });
$('ct-full').onclick = () => setFull(!fullDisk);

$('ct-search').oninput = presetOptions;
$('ct-preset').onchange = applyPreset;
$('ct-apply').onclick = applyPreset;
$('ct-add').onclick = () => addPoint();
$('ct-ramp').ondblclick = e => {
  const r = $('ct-ramp').getBoundingClientRect();
  addPoint(palette.vmin + Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (palette.vmax - palette.vmin));
};
$('ct-reverse').onclick = () => {
  remember();
  palette.points = sorted(palette).reverse().map(p => {
    const q = { ...p, t: palette.vmin + palette.vmax - p.t };
    if (Number.isFinite(p.x)) q.x = 1 - p.x; else delete q.x;
    return q;
  });
  edited();
  editor();
};
for (const id of ['ct-vmin', 'ct-vmax']) $(id).onchange = () => {
  const a = Number($('ct-vmin').value), b = Number($('ct-vmax').value);
  if ($('ct-vmin').value === '' || $('ct-vmax').value === '' || a >= b) { $('ct-validation').textContent = 'Minimum must be less than maximum.'; return; }
  remember();
  // Rescale every breakpoint so the table keeps its shape over the new range
  const oldA = palette.vmin, oldB = palette.vmax;
  for (const p of palette.points) p.t = a + (p.t - oldA) / (oldB - oldA) * (b - a);
  palette.vmin = a;
  palette.vmax = b;
  edited();
  editor();
};
$('ct-mode').onchange = () => { remember(); palette.mode = $('ct-mode').value; palette.n = 256; edited(); editor(); };
$('ct-undo').onclick = () => { if (undoStack.length) { redoStack.push(clone(palette)); palette = undoStack.pop(); editor(); } };
$('ct-redo').onclick = () => { if (redoStack.length) { undoStack.push(clone(palette)); palette = redoStack.pop(); editor(); } };

$('ct-save').onclick = () => {
  try { validate(palette); download(new Blob([JSON.stringify(palette, null, 2)], { type: 'application/json' }), `${val('ct-name') || 'colortable'}.json`); }
  catch (e) { $('ct-validation').textContent = e.message; }
};
$('ct-import').onclick = () => $('ct-file').click();
$('ct-file').onchange = async () => {
  try {
    const file = $('ct-file').files[0];
    if (!file) return;
    if (file.size > 1024 * 1024) throw Error('JSON file is too large.');
    const p = JSON.parse(await file.text());
    validate(p);
    p.n = Number.isInteger(p.n) && p.n >= 2 && p.n <= 4096 ? p.n : 256;
    remember();
    palette = p;
    edited();
    editor();
  } catch (e) { $('ct-validation').textContent = e.message; }
  finally { $('ct-file').value = ''; }
};

$('ct-python').onclick = () => {
  try { $('ct-code').value = pythonCode(palette, val('ct-name')); $('ct-dialog').showModal(); }
  catch (e) { $('ct-validation').textContent = e.message; }
};
$('ct-close').onclick = () => $('ct-dialog').close();
$('ct-copy').onclick = async () => {
  try { await navigator.clipboard.writeText($('ct-code').value); $('ct-copy').textContent = 'copied'; setTimeout(() => $('ct-copy').textContent = 'copy', 1500); }
  catch { $('ct-code').select(); }
};
$('ct-download').onclick = () => {
  const header = `from matplotlib.colors import ${palette.mode === 'listed' ? 'ListedColormap' : 'LinearSegmentedColormap'}\n\n\n`;
  download(new Blob([header + $('ct-code').value], { type: 'text/x-python' }), `${val('ct-name')}.py`);
};

$('ct-reset').onclick = () => viewer?.full();
$('ct-zoom-in').onclick = () => viewer?.zoom(.8);
$('ct-zoom-out').onclick = () => viewer?.zoom(1.25);
// Packed value -> RGBA pixel (as one little-endian uint32) for all 65536 values, binned exactly like the
// viewer's shader, so a full-resolution export is colored the same as the screen.
function colorTable() {
  const m = current, n = lut.length / 4, colors = new Uint32Array(lut.buffer, lut.byteOffset, n);
  const { vmin, vmax } = palette, out = new Uint32Array(65536);
  for (let v = 0; v < 65535; v++) {
    const f = Math.min(1, Math.max(0, (v * m.scale + m.offset - vmin) / (vmax - vmin)));
    out[v] = colors[Math.min(n - 1, Math.floor(f * n))];
  }
  out[65535] = 0xffffffff;   // missing: page white, like the viewer
  return out;
}

// The current view at native pixels (up to EXPORT_PX on the long side), colored with the current table.
async function nativeImage() {
  const m = current, [vx, vy, vw, vh] = viewer.view, s = m.stride || 1;
  let data, w, h;
  if (s > 1 && m.refinable) {
    // the loaded overview is thinned: fetch the view's native pixels
    const [C0, R0] = m.crop;
    const c0 = Math.max(0, Math.floor(C0 + vx * s)), c1 = Math.min(m.grid_width, Math.ceil(C0 + (vx + vw) * s));
    const r0 = Math.max(0, Math.floor(R0 + vy * s)), r1 = Math.min(m.grid_height, Math.ceil(R0 + (vy + vh) * s));
    const meta = await callAPI('/colortable/load', { ...request, rect: [c0, r0, c1, r1].join(','), maxpx: EXPORT_PX });
    data = await pixels(meta);
    w = meta.width;
    h = meta.height;
  } else {
    // the loaded image already holds every pixel there is: cut the view out of it
    const x0 = Math.max(0, Math.floor(vx)), x1 = Math.min(m.width, Math.ceil(vx + vw));
    const y0 = Math.max(0, Math.floor(vy)), y1 = Math.min(m.height, Math.ceil(vy + vh));
    w = x1 - x0;
    h = y1 - y0;
    data = new Uint16Array(w * h);
    for (let y = 0; y < h; y++) data.set(viewer.data.subarray((y0 + y) * m.width + x0, (y0 + y) * m.width + x1), y * w);
  }
  const table = colorTable(), img = document.createElement('canvas');
  img.width = w;
  img.height = h;
  const ctx = img.getContext('2d'), im = ctx.createImageData(w, h), out = new Uint32Array(im.data.buffer);
  for (let i = 0; i < data.length; i++) out[i] = table[data[i]];
  ctx.putImageData(im, 0, 0);
  return img;
}

// What copy / save produce: the screen view, or the native-resolution view when FULL RES is on.
async function exportImage() {
  return labeledImage(fullRes ? await nativeImage() : viewer.snapshot());
}

function setFullRes(on) {
  fullRes = on;
  $('ct-fullres').classList.toggle('active', on);
  try { localStorage.setItem('cw_ct_fullres', on ? '1' : '0'); } catch { }
}

// Image with a title strip above and the colorbar below, laid out like the site's matplotlib plots:
// satellite/band and scan time on the left, author on the right.
async function labeledImage(img) {
  const W = img.width;
  const size = Math.max(12, Math.round(W / 60)), pad = Math.round(size * .6), line = Math.round(size * 1.25);
  const barH = Math.round(size * 1.1), tickH = Math.round(size * .35);
  const font = weight => `${weight} ${size}px "Courier Prime", "Courier New", monospace`;
  try { await Promise.all([document.fonts.load(font('bold')), document.fonts.load(font('normal'))]); } catch { }
  const top = 2 * line + pad, barTop = top + img.height + pad;
  const out = document.createElement('canvas');
  out.width = W;
  out.height = barTop + barH + tickH + line + Math.round(size * .3);
  const ctx = out.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = '#111';
  ctx.font = font('bold');
  ctx.fillText(`${current.title} ${current.quantity}`, 0, line - size * .25);
  ctx.fillText(`Time: ${current.scan_time}`, 0, 2 * line - size * .25);
  ctx.font = font('normal');
  ctx.textAlign = 'right';
  ctx.fillText('Deelan Jariwala', W, 2 * line - size * .25);
  ctx.drawImage(img, 0, top);

  // Colorbar: one column per output pixel, binned exactly like the image
  const bar = ctx.createImageData(W, 1), n = lut.length / 4;
  for (let x = 0; x < W; x++) {
    const i = Math.min(n - 1, Math.floor(x / W * n)) * 4;
    bar.data.set(lut.subarray(i, i + 4), x * 4);
  }
  const strip = document.createElement('canvas');
  strip.width = W;
  strip.height = 1;
  strip.getContext('2d').putImageData(bar, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(strip, 0, barTop, W, barH);
  ctx.strokeStyle = '#111';
  ctx.lineWidth = Math.max(1, Math.round(size / 14));
  ctx.strokeRect(ctx.lineWidth / 2, barTop + ctx.lineWidth / 2, W - ctx.lineWidth, barH - ctx.lineWidth);

  const { vmin, vmax } = palette;
  ctx.font = font('normal');
  ctx.textAlign = 'center';
  for (const t of tickValues()) {
    const x = Math.round((t - vmin) / (vmax - vmin) * (W - 1)) + .5, text = label(t), half = ctx.measureText(text).width / 2;
    ctx.fillRect(x - ctx.lineWidth / 2, barTop + barH, ctx.lineWidth, tickH);
    ctx.fillText(text, Math.min(W - half, Math.max(half, x)), barTop + barH + tickH + line - size * .25);
  }
  return out;
}

$('ct-png').onclick = async () => {
  if (!viewer?.hasImage) { setStatus('error', 'Run an image first.'); return; }
  const name = `${current.title.toLowerCase().replace(/[^a-z0-9]+/g, '')}_${current.scan_time.replace(/\D/g, '').slice(0, 12)}_${val('ct-name') || 'custom'}.png`;
  setStatus('loading', 'saving...');
  try {
    const img = await exportImage();
    img.toBlob(b => { if (b) download(b, name); setStatus('success', 'done.'); }, 'image/png');
  } catch (e) { setStatus('error', e.message); }
};
$('ct-fullres').onclick = () => setFullRes(!fullRes);

// The browser's own "Copy image" only sees the bare canvas, so copy the labeled image instead.
// The blob is passed as a promise so the clipboard write starts inside the click (Safari needs that).
async function copyImage() {
  if (!viewer?.hasImage) { setStatus('error', 'Run an image first.'); return; }
  setStatus('loading', 'copying...');
  try {
    const blob = exportImage().then(c => new Promise(r => c.toBlob(r, 'image/png')));
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    $('ct-copy-png').textContent = 'copied';
    setTimeout(() => { $('ct-copy-png').textContent = 'copy png'; }, 1500);
    setStatus('success', 'done.');
  } catch (e) {
    setStatus('error', `Couldn't copy (${e.message}); use save png instead.`);
  }
}
$('ct-copy-png').onclick = copyImage;

const canvas = $('ct-canvas');
let drag = null;
canvas.addEventListener('wheel', e => {
  if (!viewer?.hasImage) return;
  e.preventDefault();
  viewer.zoom(Math.exp(Math.max(-500, Math.min(500, e.deltaY)) * .0015), viewer.pixel(e.clientX, e.clientY));
}, { passive: false });
canvas.onpointerdown = e => {
  if (!viewer?.hasImage) return;
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, start: [...viewer.view] };
};
canvas.onpointerup = canvas.onpointercancel = () => { drag = null; };
canvas.onpointermove = e => {
  if (drag) viewer.pan(e.clientX - drag.x, e.clientY - drag.y, drag.start);
  const p = viewer?.pixel(e.clientX, e.clientY);
  if (!p) return;
  const m = viewer.meta, raw = viewer.valueAt(p.x, p.y), ll = latlon(m, p.x, p.y);
  const value = raw * (m.scale ?? .01) + (m.offset ?? -200);
  $('ct-hover').textContent = (raw === 65535 ? 'no data' : `${value.toFixed(isVis() ? 3 : 2)} ${unit()}`.trim())
    + (ll ? ` · ${ll.lat.toFixed(2)}°, ${ll.lon.toFixed(2)}°` : '');
};

window.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); $(e.shiftKey ? 'ct-redo' : 'ct-undo').click(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c' && !String(getSelection())) { e.preventDefault(); copyImage(); }
});
window.addEventListener('resize', () => { if (palette) schedule(); });

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

try { viewer = new Viewer(canvas); viewer.onview = scheduleRefine; }
catch (e) { setStatus('error', e.message); }
restoreInputs();
try { setFullRes(localStorage.getItem('cw_ct_fullres') === '1'); } catch { }
try {
  presets = await callAPI('/colortable/presets');
  setKind(fieldKind());
} catch (e) {
  setStatus('error', 'Could not load presets: ' + e.message);
}
