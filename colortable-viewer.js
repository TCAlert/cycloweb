// WebGL 2 viewer: native-grid uint16 temperatures live in an integer texture, and the
// palette is a small RGBA lookup texture, so recoloring never touches the pixel data.

const BACKGROUND = [1, 1, 1, 1];   // page white; also used for missing / space pixels

const VERTEX = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0., 1.); }`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp usampler2D;
uniform usampler2D dataTex;
uniform sampler2D colorTex;
uniform vec4 rect;          // image placement on the canvas (x, y, w, h) in device pixels
uniform vec4 view;          // visible data window (col, row, w, h)
uniform vec2 range;         // vmin, vmax in °C
uniform vec4 background;
uniform float canvasHeight;
uniform int colors;
out vec4 outColor;
void main() {
  vec2 screen = vec2(gl_FragCoord.x, canvasHeight - gl_FragCoord.y);
  vec2 uv = (screen - rect.xy) / rect.zw;
  if (any(lessThan(uv, vec2(0.))) || any(greaterThanEqual(uv, vec2(1.)))) discard;
  uint value = texelFetch(dataTex, ivec2(floor(view.xy + uv * view.zw)), 0).r;
  if (value == 65535u) { outColor = background; return; }
  float t = float(value) * .01 - 200.;
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
    for (const name of ['dataTex', 'colorTex', 'rect', 'view', 'range', 'background', 'canvasHeight', 'colors'])
      this.locations[name] = g.getUniformLocation(this.program, name);
    this.dataTex = g.createTexture();
    this.colorTex = g.createTexture();
    g.uniform1i(this.locations.dataTex, 0);
    g.uniform1i(this.locations.colorTex, 1);
    g.uniform4fv(this.locations.background, BACKGROUND);
    this.bind(this.dataTex, 0);
    this.bind(this.colorTex, 1);
    this.range = [-100, 30];
    this.hasImage = false;
    new ResizeObserver(() => this.draw()).observe(canvas);
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
    this.bind(this.dataTex, 0);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.R16UI, meta.width, meta.height, 0, g.RED_INTEGER, g.UNSIGNED_SHORT, data);
    this.hasImage = true;
    this.full();
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
  }

  clamp() {
    const v = this.view, m = this.meta;
    if (!v || !m) return;
    v[2] = Math.min(m.width, Math.max(2, v[2]));
    v[3] = Math.min(m.height, Math.max(2, v[3]));
    v[0] = Math.max(0, Math.min(m.width - v[2], v[0]));
    v[1] = Math.max(0, Math.min(m.height - v[3], v[1]));
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
    const [, , vw, vh] = this.view, s = Math.min(w / vw, h / vh);
    this.rect = [(w - vw * s) / 2, (h - vh * s) / 2, vw * s, vh * s];
    g.useProgram(this.program);
    g.uniform4fv(this.locations.rect, this.rect);
    g.uniform4fv(this.locations.view, this.view);
    g.uniform2fv(this.locations.range, this.range);
    g.uniform1f(this.locations.canvasHeight, h);
    g.uniform1i(this.locations.colors, this.colors);
    g.drawArrays(g.TRIANGLES, 0, 6);
  }

  // Copy of just the drawn image (no letterbox margins), for PNG export.
  snapshot() {
    this.draw();
    const [x, y, w, h] = this.rect.map(Math.round), out = document.createElement('canvas');
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
  }

  pan(dx, dy, start) {
    const r = this.canvas.getBoundingClientRect(), scale = this.canvas.width / r.width;
    this.view = [start[0] - dx * scale / this.rect[2] * start[2], start[1] - dy * scale / this.rect[3] * start[3], start[2], start[3]];
    this.clamp();
    this.draw();
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
