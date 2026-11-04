// The person matte at display size, a port of snap/matte.py MatteEngine.update and core(): the low resolution
// person confidence refined by a color guided filter (radius matte_radius, eps matte_eps, run at matte_scale
// like the fast guided filter), floored at matte_floor, with the matte_on / matte_keep hysteresis. The core
// (matte support grown by zone_dilate_px) is computed at the mask's half resolution. The statistics use
// 32-bit floats: the covariance terms subtract nearly equal means and would turn to noise at half precision.

import configData from "../data/config.json";
import { ellipseRows } from "../ellipse";
import { Pass, Target, TargetSet } from "./gpu";

const C = configData.config;

// Guide (the frame averaged to the filter's resolution) and the products the filter needs.
const PRODUCTS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
uniform sampler2D mask;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
void main() {
  vec3 i = texture(frame, uv).rgb;
  float p = texture(mask, uv).r;
  o0 = vec4(i, p);
  o1 = vec4(i * p, 0.0);
  o2 = vec4(i.r * i.r, i.r * i.g, i.r * i.b, i.g * i.g);
  o3 = vec4(i.g * i.b, i.b * i.b, 0.0, 0.0);
}`;

// Normalized box filter along one axis, border reflect 101 (OpenCV's default).
function boxShader(count: number): string {
  const inputs = Array.from({ length: count }, (_, k) => `uniform sampler2D t${k};`).join("\n");
  const outputs = Array.from({ length: count }, (_, k) => `layout(location = ${k}) out vec4 o${k};`).join("\n");
  const sums = Array.from({ length: count }, (_, k) => `s${k} += texelFetch(t${k}, q, 0);`).join("\n    ");
  const decl = Array.from({ length: count }, (_, k) => `vec4 s${k} = vec4(0.0);`).join("\n  ");
  const out = Array.from({ length: count }, (_, k) => `o${k} = s${k} / n;`).join("\n  ");
  return `#version 300 es
precision highp float;
${inputs}
uniform ivec2 dir;
uniform int radius;
${outputs}
int reflect101(int i, int n) {
  if (i < 0) return -i;
  if (i >= n) return 2 * n - 2 - i;
  return i;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 size = textureSize(t0, 0);
  ${decl}
  for (int k = -radius; k <= radius; k++) {
    ivec2 q = p + dir * k;
    q = ivec2(reflect101(q.x, size.x), reflect101(q.y, size.y));
    ${sums}
  }
  float n = float(2 * radius + 1);
  ${out}
}`;
}

// The linear model per window: a = (Sigma + eps I)^-1 cov(I, p), b = mean(p) - a . mean(I).
const COEFFICIENTS = `#version 300 es
precision highp float;
uniform sampler2D m0;
uniform sampler2D m1;
uniform sampler2D m2;
uniform sampler2D m3;
uniform float eps;
out vec4 color;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a0 = texelFetch(m0, p, 0);
  vec4 a1 = texelFetch(m1, p, 0);
  vec4 a2 = texelFetch(m2, p, 0);
  vec4 a3 = texelFetch(m3, p, 0);
  vec3 mi = a0.rgb;
  float mp = a0.a;
  vec3 cov = a1.rgb - mi * mp;
  float rr = a2.r - mi.r * mi.r + eps;
  float rg = a2.g - mi.r * mi.g;
  float rb = a2.b - mi.r * mi.b;
  float gg = a2.a - mi.g * mi.g + eps;
  float gb = a3.r - mi.g * mi.b;
  float bb = a3.g - mi.b * mi.b + eps;
  mat3 sigma = mat3(rr, rg, rb, rg, gg, gb, rb, gb, bb);
  vec3 a = inverse(sigma) * cov;
  color = vec4(a, mp - dot(a, mi));
}`;

// q = mean(a) . I + mean(b) at display size (a and b upsampled bilinearly), then the floor and hysteresis.
const MATTE = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
uniform sampler2D coefficients;
uniform sampler2D previous;
uniform float hasPrevious;
uniform float floorValue;
uniform float onValue;
uniform float keepValue;
out vec4 color;
vec4 bilinear(sampler2D t, vec2 at) {
  ivec2 size = textureSize(t, 0);
  vec2 pos = at * vec2(size) - 0.5;
  ivec2 i0 = ivec2(floor(pos));
  vec2 f = pos - floor(pos);
  ivec2 hi = size - 1;
  vec4 a = texelFetch(t, clamp(i0, ivec2(0), hi), 0);
  vec4 b = texelFetch(t, clamp(i0 + ivec2(1, 0), ivec2(0), hi), 0);
  vec4 c = texelFetch(t, clamp(i0 + ivec2(0, 1), ivec2(0), hi), 0);
  vec4 d = texelFetch(t, clamp(i0 + ivec2(1, 1), ivec2(0), hi), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 i = texelFetch(frame, p, 0).rgb;
  vec4 ab = bilinear(coefficients, uv);
  float q = clamp(dot(ab.rgb, i) + ab.a, 0.0, 1.0);
  if (q < floorValue) q = 0.0;
  bool on = q > onValue || (hasPrevious > 0.5 && q > keepValue && texelFetch(previous, p, 0).g > 0.5);
  color = vec4(q, on ? 1.0 : 0.0, 0.0, 1.0);
}`;

// Matte support (any matte left) at the mask's resolution: a low pixel is support when any display pixel of
// its block has matte.
const SUPPORT = `#version 300 es
precision highp float;
uniform sampler2D matte;
uniform ivec2 low;
out vec4 color;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 full = textureSize(matte, 0);
  ivec2 f0 = p * full / low;
  ivec2 f1 = (p + 1) * full / low;
  float hit = 0.0;
  for (int y = f0.y; y < f1.y; y++) {
    for (int x = f0.x; x < f1.x; x++) {
      if (texelFetch(matte, ivec2(x, y), 0).r > 0.0) hit = 1.0;
    }
  }
  color = vec4(hit, 0.0, 0.0, 1.0);
}`;

const DILATE = `#version 300 es
precision highp float;
uniform sampler2D source;
uniform int rows[64];
uniform int radius;
out vec4 color;
void main() {
  ivec2 size = textureSize(source, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float hit = 0.0;
  for (int dy = -radius; dy <= radius && hit == 0.0; dy++) {
    int y = p.y + dy;
    if (y < 0 || y >= size.y) continue;
    int half_width = rows[dy + radius];
    for (int dx = -half_width; dx <= half_width; dx++) {
      int x = p.x + dx;
      if (x >= 0 && x < size.x && texelFetch(source, ivec2(x, y), 0).r > 0.5) {
        hit = 1.0;
        break;
      }
    }
  }
  color = vec4(hit, 0.0, 0.0, 1.0);
}`;

export class Matte {
  private products: TargetSet;
  private boxed: TargetSet;
  private means: TargetSet;
  private coefficients: Target;
  private coefficientsH: Target;
  private coefficientsV: Target;
  private mattes: Target[];
  private side = 0;
  private hasPrevious = false;
  private support: Target;
  readonly core: Target;
  private passes: Record<"products" | "box4" | "coefficients" | "box1" | "matte" | "support" | "dilate", Pass>;
  private filterRadius: number;
  private coreRows: Int32Array;
  private coreRadius: number;

  constructor(gl: WebGL2RenderingContext, quadVao: WebGLVertexArrayObject, width: number, height: number,
              maskWidth: number, maskHeight: number) {
    // The filter runs at matte_scale of the display size.
    const fw = Math.round(width * C.matte_scale);
    const fh = Math.round(height * C.matte_scale);
    const set = () => new TargetSet(gl, [0, 1, 2, 3].map(() => new Target(gl, fw, fh, "rgba32f")));
    this.products = set();
    this.boxed = set();
    this.means = set();
    this.coefficientsH = new Target(gl, fw, fh, "rgba32f");
    this.coefficientsV = new Target(gl, fw, fh, "rgba32f");
    this.coefficients = new Target(gl, fw, fh, "rgba32f");
    this.mattes = [new Target(gl, width, height, "rgba16f", false), new Target(gl, width, height, "rgba16f", false)];
    this.support = new Target(gl, maskWidth, maskHeight, "r8", false);
    this.core = new Target(gl, maskWidth, maskHeight, "r8", false);
    this.filterRadius = Math.max(1, Math.round(C.matte_radius * C.matte_scale));
    this.coreRadius = Math.max(1, Math.round(C.zone_dilate_px * maskWidth / width));
    this.coreRows = new Int32Array(64);
    this.coreRows.set(ellipseRows(this.coreRadius));
    this.passes = {
      products: new Pass(gl, PRODUCTS, quadVao),
      box4: new Pass(gl, boxShader(4), quadVao),
      coefficients: new Pass(gl, COEFFICIENTS, quadVao),
      box1: new Pass(gl, boxShader(1), quadVao),
      matte: new Pass(gl, MATTE, quadVao),
      support: new Pass(gl, SUPPORT, quadVao),
      dilate: new Pass(gl, DILATE, quadVao),
    };
  }

  dispose(): void {
    [this.products, this.boxed, this.means].forEach((s) => s.dispose());
    [this.coefficients, this.coefficientsH, this.coefficientsV, ...this.mattes, this.support, this.core]
      .forEach((t) => t.dispose());
  }

  /** The current matte: r = matte 0..1, g = person (after hysteresis). */
  get matte(): Target {
    return this.mattes[this.side];
  }

  reset(): void {
    this.hasPrevious = false;
  }

  update(frame: Target, mask: WebGLTexture): void {
    const t = this.products.targets;
    const b = this.boxed.targets;
    const m = this.means.targets;
    this.passes.products.run(this.products, { frame, mask });
    const r = this.filterRadius;
    this.passes.box4.run(this.boxed, { t0: t[0], t1: t[1], t2: t[2], t3: t[3], dir: [1, 0], radius: r });
    this.passes.box4.run(this.means, { t0: b[0], t1: b[1], t2: b[2], t3: b[3], dir: [0, 1], radius: r });
    this.passes.coefficients.run(this.coefficients, { m0: m[0], m1: m[1], m2: m[2], m3: m[3], eps: C.matte_eps });
    this.passes.box1.run(this.coefficientsH, { t0: this.coefficients, dir: [1, 0], radius: r });
    this.passes.box1.run(this.coefficientsV, { t0: this.coefficientsH, dir: [0, 1], radius: r });
    const next = this.mattes[1 - this.side];
    this.passes.matte.run(next, {
      frame, coefficients: this.coefficientsV, previous: this.mattes[this.side], hasPrevious: this.hasPrevious,
      floorValue: C.matte_floor, onValue: C.matte_on, keepValue: C.matte_keep,
    });
    this.side = 1 - this.side;
    this.hasPrevious = true;
    this.passes.support.run(this.support, { matte: this.matte, low: [this.support.width, this.support.height] });
    this.passes.dilate.run(this.core, { source: this.support, rows: this.coreRows, radius: this.coreRadius });
  }
}
