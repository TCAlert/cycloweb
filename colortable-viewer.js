// WebGL 2 viewer: native-grid uint16 values (temperature or reflectance, see `decode`) live in an
// integer texture, and the palette is a small RGBA lookup texture, so recoloring never touches the
// pixel data. A second, finer "detail" texture can cover part of the image: the page loads it for
// the area zoomed into, so the shown resolution follows the zoom.

const BACKGROUND = [1, 1, 1, 1];   // page white; also used for missing / space pixels

const VERTEX = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0., 1.); }`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp usampler2D;
uniform usampler2D dataTex;
uniform usampler2D detailTex;
uniform sampler2D colorTex;
uniform vec4 rect;          // image placement on the canvas (x, y, w, h) in device pixels
uniform vec4 view;          // visible data window (col, row, w, h)
uniform vec4 detailRect;    // area the detail texture covers, in base-image pixels (x, y, w, h)
uniform vec2 detailSize;    // detail texture size in texels
uniform int hasDetail;
uniform vec2 decode;        // value = texel * decode.x + decode.y
uniform vec2 range;         // vmin, vmax in the same units
uniform vec4 background;
uniform float canvasHeight;
uniform int colors;
out vec4 outColor;
void main() {
  vec2 screen = vec2(gl_FragCoord.x, canvasHeight - gl_FragCoord.y);
  vec2 uv = (screen - rect.xy) / rect.zw;
  if (any(lessThan(uv, vec2(0.))) || any(greaterThanEqual(uv, vec2(1.)))) discard;
  vec2 p = view.xy + uv * view.zw;
  // a 'contain' view can reach past the image edges: those parts are page background
  if (any(lessThan(p, vec2(0.))) || any(greaterThanEqual(p, vec2(textureSize(dataTex, 0))))) { outColor = background; return; }
  vec2 d = (p - detailRect.xy) / detailRect.zw;
  uint value = hasDetail == 1 && all(greaterThanEqual(d, vec2(0.))) && all(lessThan(d, vec2(1.)))
    ? texelFetch(detailTex, ivec2(floor(d * detailSize)), 0).r
    : texelFetch(dataTex, ivec2(floor(p)), 0).r;
  if (value == 65535u) { outColor = background; return; }
  float t = float(value) * decode.x + decode.y;
  float f = clamp((t - range.x) / (range.y - range.x), 0., 1.);
  int i = min(colors - 1, int(floor(f * float(colors))));   // matplotlib bin rule
  outColor = texelFetch(colorTex, ivec2(i, 0), 0);
}`;

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const g = this.gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false });
    if (!g) throw Error('WebGL 2 is required. Enable hardware acceleration in your browser.');
    this.maxTexture = g.getParameter(g.MAX_TEXTURE_SIZE);

    const shader = (type, source) => {
      const s = g.createShader(type);
      g.shaderSource(s, source);
      g.compileShader(s);
      if (!g.getShaderParameter(s, g.COMPILE_STATUS)) throw Error(g.getShaderInfoLog(s));
      return s;
    };
    this.program = g.createProgram();
    g.attachShader(this.program, shader(g.VERTEX_SHADER, VERTEX));
    g.attachShader(this.program, shader(g.FRAGMENT_SHADER, FRAGMENT));
    g.linkProgram(this.program);
    if (!g.getProgramParameter(this.program, g.LINK_STATUS)) throw Error(g.getProgramInfoLog(this.program));
    g.useProgram(this.program);

    g.bindBuffer(g.ARRAY_BUFFER, g.createBuffer());
    g.bufferData(g.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), g.STATIC_DRAW);
    const a = g.getAttribLocation(this.program, 'position');
    g.enableVertexAttribArray(a);
    g.vertexAttribPointer(a, 2, g.FLOAT, false, 0, 0);

    this.locations = {};
    for (const name of ['dataTex', 'detailTex', 'colorTex', 'rect', 'view', 'detailRect', 'detailSize', 'hasDetail',
                        'decode', 'range', 'background', 'canvasHeight', 'colors'])
      this.locations[name] = g.getUniformLocation(this.program, name);
    this.dataTex = g.createTexture();
    this.colorTex = g.createTexture();
    this.detailTex = g.createTexture();
    g.uniform1i(this.locations.dataTex, 0);
    g.uniform1i(this.locations.colorTex, 1);
    g.uniform1i(this.locations.detailTex, 2);
    g.uniform4fv(this.locations.background, BACKGROUND);
    this.bind(this.dataTex, 0);
    this.bind(this.colorTex, 1);
    this.bind(this.detailTex, 2);
    this.range = [-100, 30];
    this.hasImage = false;
    this.detail = null;
    this.onview = () => {};   // called after the visible area changes (zoom, pan, resize)
    new ResizeObserver(() => { this.draw(); this.onview(); }).observe(canvas);
  }

  bind(tex, unit) {
    const g = this.gl;
    g.activeTexture(g.TEXTURE0 + unit);
    g.bindTexture(g.TEXTURE_2D, tex);
    for (const [k, v] of [[g.TEXTURE_MIN_FILTER, g.NEAREST], [g.TEXTURE_MAG_FILTER, g.NEAREST],
                          [g.TEXTURE_WRAP_S, g.CLAMP_TO_EDGE], [g.TEXTURE_WRAP_T, g.CLAMP_TO_EDGE]])
      g.texParameteri(g.TEXTURE_2D, k, v);
  }

  setImage(data, meta) {
    const g = this.gl;
    if (meta.width > this.maxTexture || meta.height > this.maxTexture)
      throw Error(`This image is ${meta.width} × ${meta.height}; your GPU allows ${this.maxTexture} px. Use a smaller zoom.`);
    this.data = data;
    this.meta = meta;
    this.detail = null;
    this.bind(this.dataTex, 0);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.R16UI, meta.width, meta.height, 0, g.RED_INTEGER, g.UNSIGNED_SHORT, data);
    this.hasImage = true;
    this.full();
  }

  // Finer pixels for part of the image; rect = [x, y, w, h] in base-image pixels.
  setDetail(data, meta, rect) {
    const g = this.gl;
    if (meta.width > this.maxTexture || meta.height > this.maxTexture) return;
    this.bind(this.detailTex, 2);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.R16UI, meta.width, meta.height, 0, g.RED_INTEGER, g.UNSIGNED_SHORT, data);
    this.detail = { data, meta, rect };
    this.draw();
  }

  clearDetail() {
    this.detail = null;
    this.draw();
  }

  // Packed value under a base-image position, from the detail texture where it covers it.
  valueAt(x, y) {
    const d = this.detail;
    if (d) {
      const u = (x - d.rect[0]) / d.rect[2], v = (y - d.rect[1]) / d.rect[3];
      if (u >= 0 && u < 1 && v >= 0 && v < 1)
        return d.data[Math.floor(v * d.meta.height) * d.meta.width + Math.floor(u * d.meta.width)];
    }
    return this.data[Math.floor(y) * this.meta.width + Math.floor(x)];
  }

  setPalette(lut, min, max) {
    const g = this.gl;
    this.bind(this.colorTex, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.RGBA8, lut.length / 4, 1, 0, g.RGBA, g.UNSIGNED_BYTE, lut);
    this.colors = lut.length / 4;
    this.range = [min, max];
    this.draw();
  }

  full() { if (this.meta) this.setView([0, 0, this.meta.width, this.meta.height]); }

  setView(bounds) {
    this.view = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]];
    this.clamp();
    this.draw();
    this.onview();
  }

  // Keep the view at the canvas's aspect ratio so the image fills the viewer. Zoomed all the way out,
  // 'cover' images (the default) are cropped to fill it; 'contain' images (side-by-side mesosectors)
  // are shown whole, with background beside them. Keeps the view's centre.
  clamp() {
    const v = this.view, m = this.meta;
    if (!v || !m) return;
    let w = Math.max(2, v[2]), h = Math.max(2, v[3]);
    if (this.aspect) {
      const maxW = m.fit === 'contain' ? Math.max(m.width, m.height * this.aspect) : Math.min(m.width, m.height * this.aspect);
      w = Math.min(w, maxW);
      h = w / this.aspect;
    } else {
      w = Math.min(m.width, w);
      h = Math.min(m.height, h);
    }
    const place = (c, size, full) => size >= full ? (full - size) / 2 : Math.max(0, Math.min(full - size, c - size / 2));
    this.view = [place(v[0] + v[2] / 2, w, m.width), place(v[1] + v[3] / 2, h, m.height), w, h];
  }

  draw() {
    const c = this.canvas, g = this.gl, ratio = window.devicePixelRatio || 1;
    const w = Math.round(c.clientWidth * ratio), h = Math.round(c.clientHeight * ratio);
    if (!w || !h) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    g.viewport(0, 0, w, h);
    g.clearColor(...BACKGROUND);
    g.clear(g.COLOR_BUFFER_BIT);
    if (!this.hasImage || !this.colors) return;
    this.aspect = w / h;
    this.clamp();
    const [, , vw, vh] = this.view, s = Math.min(w / vw, h / vh);
    this.rect = [(w - vw * s) / 2, (h - vh * s) / 2, vw * s, vh * s];
    g.useProgram(this.program);
    g.uniform4fv(this.locations.rect, this.rect);
    g.uniform4fv(this.locations.view, this.view);
    g.uniform2fv(this.locations.range, this.range);
    g.uniform2fv(this.locations.decode, [this.meta.scale ?? .01, this.meta.offset ?? -200]);
    const d = this.detail;
    g.uniform1i(this.locations.hasDetail, d ? 1 : 0);
    if (d) {
      g.uniform4fv(this.locations.detailRect, d.rect);
      g.uniform2fv(this.locations.detailSize, [d.meta.width, d.meta.height]);
    }
    g.uniform1f(this.locations.canvasHeight, h);
    g.uniform1i(this.locations.colors, this.colors);
    g.drawArrays(g.TRIANGLES, 0, 6);
  }

  // Copy of the drawn image (not any background beside a 'contain' image), for PNG export.
  snapshot() {
    this.draw();
    const [vx, vy, vw, vh] = this.view, m = this.meta, [rx, ry, rw, rh] = this.rect;
    const x = Math.round(rx + Math.max(0, -vx) / vw * rw), y = Math.round(ry + Math.max(0, -vy) / vh * rh);
    const w = Math.round(rx + Math.min(vw, m.width - vx) / vw * rw) - x, h = Math.round(ry + Math.min(vh, m.height - vy) / vh * rh) - y;
    const out = document.createElement('canvas');
    out.width = w;
    out.height = h;
    out.getContext('2d').drawImage(this.canvas, x, y, w, h, 0, 0, w, h);
    return out;
  }

  // Data-space position under a client point, or null when outside the image.
  pixel(clientX, clientY) {
    if (!this.hasImage || !this.rect) return null;
    const r = this.canvas.getBoundingClientRect();
    const sx = (clientX - r.left) * this.canvas.width / r.width, sy = (clientY - r.top) * this.canvas.height / r.height;
    const [rx, ry, rw, rh] = this.rect, u = (sx - rx) / rw, v = (sy - ry) / rh;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    return { x: this.view[0] + u * this.view[2], y: this.view[1] + v * this.view[3], u, v };
  }

  zoom(factor, p) {
    if (!this.hasImage) return;
    const v = this.view, u = p?.u ?? .5, t = p?.v ?? .5;
    const w = Math.min(this.meta.width, Math.max(2, v[2] * factor)), h = Math.min(this.meta.height, Math.max(2, v[3] * factor));
    this.view = [v[0] + u * (v[2] - w), v[1] + t * (v[3] - h), w, h];
    this.clamp();
    this.draw();
    this.onview();
  }

  pan(dx, dy, start) {
    const r = this.canvas.getBoundingClientRect(), scale = this.canvas.width / r.width;
    this.view = [start[0] - dx * scale / this.rect[2] * start[2], start[1] - dy * scale / this.rect[3] * start[3], start[2], start[3]];
    this.clamp();
    this.draw();
    this.onview();
  }
}

// Pixel (col, row) -> { lat, lon }, or null off the disk.
// 'latlon' grids (GridSat) are regular; geostationary grids use PROJ's geos inverse, which
// covers both GOES-R (sweep x) and Himawari (sweep y).
export function latlon(m, x, y) {
  const wrap = lon => ((lon + 180) % 360 + 360) % 360 - 180;
  if (m.grid === 'latlon') return { lat: m.lat0 + y * m.dlat, lon: wrap(m.lon0 + x * m.dlon) };
  if (!m.projection) return null;
  const g = m.projection, X = m.x0 + x * m.dx, Y = m.y0 + y * m.dy;   // scan angles (radians)
  const a = g.semi_major_axis, rp = g.semi_minor_axis / a, rg = (g.perspective_point_height + a) / a;
  let vx = -1, vy, vz;
  if (g.sweep_angle_axis === 'x') { vz = Math.tan(Y); vy = Math.tan(X) * Math.hypot(1, vz); }
  else { vy = Math.tan(X); vz = Math.tan(Y) * Math.hypot(1, vy); }
  const A = vy * vy + (vz / rp) ** 2 + vx * vx, B = 2 * rg * vx, det = B * B - 4 * A * (rg * rg - 1);
  if (det < 0) return null;
  const k = (-B - Math.sqrt(det)) / (2 * A);
  vx = rg + k * vx; vy *= k; vz *= k;
  const lam = Math.atan2(vy, vx);
  const phi = Math.atan(Math.tan(Math.atan(vz * Math.cos(lam) / vx)) / (rp * rp));
  return { lat: phi * 180 / Math.PI, lon: wrap(g.longitude_of_projection_origin + lam * 180 / Math.PI) };
}
