import { validate, sorted, hex, colorAt, makeLUT, pythonCode } from './colortable-palette.js';
import { Viewer, latlon } from './colortable-viewer.js';

// ?api=http://127.0.0.1:7860 points the page at a local api.py for testing
const LOCAL_API = new URLSearchParams(location.search).get('api');
const API_BASE = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(LOCAL_API || '') ? LOCAL_API : 'https://tcalert-cycloweb2.hf.space';
const FULL_DISK_PX = 5500;   // largest whole image (Himawari full disk)
const STORE_KEY = 'cw_ct_palettes';

const $ = id => document.getElementById(id);
const clone = x => JSON.parse(JSON.stringify(x));
const val = id => $(id).value.trim();

let presets = {};          // { '13': [...IR tables], '9': [...WV tables] }
let kind = '13';           // which preset group the editor is showing
let palette = null;
let viewer = null;
let current = null;        // metadata of the displayed image
let selected = -1;
let undoStack = [], redoStack = [];
let lut, drawQueued = false, fullDisk = false;
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

function niceStep(span) {
  for (const s of [1, 2, 5, 10, 20, 25, 50, 100]) if (span / s <= 10) return s;
  return 200;
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
  const { vmin, vmax } = palette, step = niceStep(vmax - vmin), ticks = [];
  for (let t = Math.ceil(vmin / step) * step; t <= vmax + 1e-9; t += step) {
    const s = document.createElement('span');
    s.style.left = 100 * (t - vmin) / (vmax - vmin) + '%';
    s.textContent = `${Math.round(t * 10) / 10}°C`;
    ticks.push(s);
  }
  $('ct-ticks').replaceChildren(...ticks);
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
    temp.step = '0.1';
    temp.value = Number(p.t.toFixed(2));
    temp.title = 'Temperature (°C)';
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
    dup.title = 'Duplicate (same temperature = sharp jump)';
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
    h.title = `${p.t.toFixed(1)} °C · ${p.color}`;
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
          : Math.round(t * 10) / 10;
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
    palette.points.push({ t: Number(t.toFixed(1)), color: hex(colorAt(pts, t)) });
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
  $('ct-preset-info').textContent = `${group.length} ${kind === '13' ? 'IR' : 'WV'} tables in satcmaps.py`;
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

// Switch the editor between the IR and WV collections, restoring that group's last palette.
function setKind(next) {
  if (next === kind && palette) return;
  kind = next;
  undoStack = []; redoStack = [];
  $('ct-search').value = '';
  presetOptions();
  const group = presets[kind] || [];
  if (!group.length) return;
  const fallback = kind === '13' ? 'irg' : 'wv';
  $('ct-preset').value = group.some(x => x.name === fallback) ? fallback : group[0].name;
  const name = val('ct-name');
  if (!name || name === 'custom_ir' || name === 'custom_wv') $('ct-name').value = kind === '13' ? 'custom_ir' : 'custom_wv';
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

// Water-vapor bands use the WV tables: 8–10 on GOES-R / Himawari, channel 3 on GridSat
function kindFor(satellite, band) {
  const s = satellite.toUpperCase(), n = parseInt(band);
  const gridsat = s === 'B1' || (/^\d+$/.test(s) && Number(s) >= 8 && Number(s) <= 15);
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
    const meta = await callAPI('/colortable/load', {
      satellite: val('ct-satellite') || '19',
      date: val('ct-date'),
      time: val('ct-time'),
      band: val('ct-band'),
      lat: val('ct-lat'),
      lon: val('ct-lon'),
      zoom: val('ct-zoom'),
      full: fullDisk ? 'true' : null,
    });
    if (meta.width > viewer.maxTexture || meta.height > viewer.maxTexture)
      throw Error(`Image is ${meta.width} × ${meta.height}; this GPU allows ${viewer.maxTexture} px. Use a smaller zoom.`);
    setStatus('loading', `transferring ${meta.width} × ${meta.height} px...`);
    const res = await fetch(`${API_BASE}/colortable/frame/${meta.frame}`);
    if (!res.ok) throw Error('Image expired on the server; run again.');
    const buffer = await res.arrayBuffer();
    if (buffer.byteLength !== meta.width * meta.height * 2) throw Error('Incomplete transfer; run again.');
    setKind(meta.kind);
    viewer.setImage(new Uint16Array(buffer), meta);
    current = meta;
    $('ct-placeholder').hidden = true;
    $('ct-title').textContent = `${meta.title} · ${meta.scan_time} · ${meta.width} × ${meta.height} native px`;
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
  if (on && !(viewer && viewer.maxTexture >= FULL_DISK_PX)) return;
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
$('ct-png').onclick = () => {
  if (!viewer?.hasImage) { setStatus('error', 'Run an image first.'); return; }
  viewer.draw();
  const name = `${current.title.toLowerCase().replace(/[^a-z0-9]+/g, '')}_${current.scan_time.replace(/\D/g, '').slice(0, 12)}_${val('ct-name') || 'custom'}.png`;
  $('ct-canvas').toBlob(b => { if (b) download(b, name); }, 'image/png');
};

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
  const ix = Math.floor(p.x), iy = Math.floor(p.y), raw = viewer.data[iy * viewer.meta.width + ix];
  const ll = latlon(viewer.meta, ix, iy);
  $('ct-hover').textContent = (raw === 65535 ? 'no data' : `${(raw * .01 - 200).toFixed(2)} °C`)
    + (ll ? ` · ${ll.lat.toFixed(2)}°, ${ll.lon.toFixed(2)}°` : '');
};

window.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); $(e.shiftKey ? 'ct-redo' : 'ct-undo').click(); }
});
window.addEventListener('resize', () => { if (palette) schedule(); });

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

try { viewer = new Viewer(canvas); }
catch (e) { setStatus('error', e.message); }
if (viewer && viewer.maxTexture < FULL_DISK_PX) {
  $('ct-full').disabled = true;
  $('ct-full-hint').textContent = `unavailable: GPU limit ${viewer.maxTexture} px`;
}
restoreInputs();
try {
  presets = await callAPI('/colortable/presets');
  setKind(fieldKind());
} catch (e) {
  setStatus('error', 'Could not load presets: ' + e.message);
}
