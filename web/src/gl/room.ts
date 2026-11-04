// Learns the empty room, a port of snap/background.py BackgroundEngine (learning, automatic and manual
// capture, coverage). The per pixel work runs in shaders on display size textures; the decisions run here,
// in the same order as the Python update(). Not ported yet: the camera moved check and the capture mismatch
// check (phase correlation), the exposure gain map and the sensor noise synthesis.

import configData from "../data/config.json";
import texts from "../data/texts.json";
import { ellipseRows, lowResRadius } from "../ellipse";
import { Pass, Target, TargetSet } from "./gpu";

const C = configData.config;
export type CaptureResult = "ok" | "person" | "moving" | "mismatch";

// Expands the low resolution person mask by the safe margin (dilate_to in background.py).
const UNSAFE = `#version 300 es
precision highp float;
uniform sampler2D mask;
uniform int rows[64];
uniform int radius;
uniform float conf;
out vec4 color;
void main() {
  ivec2 size = textureSize(mask, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float hit = 0.0;
  for (int dy = -radius; dy <= radius && hit == 0.0; dy++) {
    int y = p.y + dy;
    if (y < 0 || y >= size.y) continue;
    int half_width = rows[dy + radius];
    for (int dx = -half_width; dx <= half_width; dx++) {
      int x = p.x + dx;
      if (x >= 0 && x < size.x && texelFetch(mask, ivec2(x, y), 0).r >= conf) {
        hit = 1.0;
        break;
      }
    }
  }
  color = vec4(hit, 0.0, 0.0, 1.0);
}`;

// One learning step per pixel. plate: the room (0..255 per channel); state.r: samples seen (count),
// state.g: consecutive safe frames (_safe_run), state.b: safe this frame.
const LEARN = `#version 300 es
precision highp float;
uniform sampler2D frame;
uniform sampler2D plate;
uniform sampler2D state;
uniform sampler2D unsafe;
uniform float resetRun;
uniform float safeFrames;
uniform float medianStep;
layout(location = 0) out vec4 plateOut;
layout(location = 1) out vec4 stateOut;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 full = textureSize(frame, 0);
  ivec2 low = textureSize(unsafe, 0);
  vec3 f = round(texelFetch(frame, p, 0).rgb * 255.0);
  vec3 room = round(texelFetch(plate, p, 0).rgb * 255.0);
  vec4 s = round(texelFetch(state, p, 0) * 255.0);
  float count = s.r;
  float run = resetRun > 0.5 ? 0.0 : s.g;
  // INTER_NEAREST upsample: low resolution pixel floor(x * low / full).
  bool notSafe = texelFetch(unsafe, p * low / full, 0).r > 0.5;
  run = notSafe ? 0.0 : min(run + 1.0, 255.0);
  bool safe = run > safeFrames;
  if (safe) {
    if (count == 0.0) room = f;
    // Approximate running median: step toward the frame by at most medianStep.
    room += min(max(f - room, 0.0), medianStep) - min(max(room - f, 0.0), medianStep);
    count = min(count + 1.0, 255.0);
  }
  plateOut = vec4(room / 255.0, 1.0);
  stateOut = vec4(count / 255.0, run / 255.0, safe ? 1.0 : 0.0, 1.0);
}`;

// The bg_moved_grid copy of the frame (INTER_AREA for an exact 2:1 is the 2x2 average, which a linear
// sample between four texels gives).
const SMALL = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
out vec4 color;
void main() {
  color = vec4(texture(frame, uv).rgb, 1.0);
}`;

// Per block: room pixels known, zone pixels, both, and the summed motion against the previous small frame.
const STATS = `#version 300 es
precision highp float;
uniform sampler2D state;
uniform sampler2D core;
uniform sampler2D small;
uniform sampler2D previous;
uniform float hasPrevious;
uniform float knownSamples;
out vec4 color;
void main() {
  ivec2 b = ivec2(gl_FragCoord.xy);
  ivec2 blocks = ivec2(80, 45);
  ivec2 full = textureSize(state, 0);
  ivec2 low = textureSize(core, 0);
  ivec2 f0 = b * full / blocks;
  ivec2 f1 = (b + 1) * full / blocks;
  float known = 0.0;
  float zone = 0.0;
  float both = 0.0;
  for (int y = f0.y; y < f1.y; y++) {
    for (int x = f0.x; x < f1.x; x++) {
      bool k = round(texelFetch(state, ivec2(x, y), 0).r * 255.0) >= knownSamples;
      bool z = texelFetch(core, ivec2(x, y) * low / full, 0).r > 0.5;
      known += k ? 1.0 : 0.0;
      zone += z ? 1.0 : 0.0;
      both += (k && z) ? 1.0 : 0.0;
    }
  }
  float motion = 0.0;
  if (hasPrevious > 0.5) {
    ivec2 grid = textureSize(small, 0);
    ivec2 s0 = b * grid / blocks;
    ivec2 s1 = (b + 1) * grid / blocks;
    for (int y = s0.y; y < s1.y; y++) {
      for (int x = s0.x; x < s1.x; x++) {
        vec3 d = abs(texelFetch(small, ivec2(x, y), 0).rgb - texelFetch(previous, ivec2(x, y), 0).rgb) * 255.0;
        motion += (d.r + d.g + d.b) / 3.0;
      }
    }
  }
  color = vec4(known, zone, both, motion);
}`;

const ACCUMULATE = `#version 300 es
precision highp float;
uniform sampler2D frame;
uniform sampler2D sum;
uniform float first;
out vec4 color;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 previous = first > 0.5 ? vec4(0.0) : texelFetch(sum, p, 0);
  color = previous + vec4(round(texelFetch(frame, p, 0).rgb * 255.0), 0.0);
}`;

// The averaged capture becomes the plate, and every pixel counts as known (_accept).
const ACCEPT = `#version 300 es
precision highp float;
uniform sampler2D sum;
uniform sampler2D state;
uniform float frames;
uniform float knownSamples;
layout(location = 0) out vec4 plateOut;
layout(location = 1) out vec4 stateOut;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  // convertScaleAbs: round, then saturate.
  vec3 average = clamp(round(texelFetch(sum, p, 0).rgb / frames), 0.0, 255.0);
  vec4 s = round(texelFetch(state, p, 0) * 255.0);
  plateOut = vec4(average / 255.0, 1.0);
  stateOut = vec4(max(s.r, knownSamples) / 255.0, s.g / 255.0, s.b / 255.0, 1.0);
}`;

const BLOCKS: [number, number] = [80, 45];

export interface RoomStats {
  knownFraction: number;
  /** Known share of the person zone, or of the whole frame when nobody is in view. */
  coverage: number;
  still: boolean;
  absent: boolean;
}

interface Capture {
  auto: boolean;
  n: number;
  bad: CaptureResult | null;
}

export class Room {
  private plates: TargetSet[];
  private side = 0;
  private unsafe: Target;
  private smalls: Target[];
  private smallSide = 0;
  private stats: Target;
  private sums: Target[];
  private sumSide = 0;
  private readback: Float32Array;
  private passes: Record<"unsafe" | "learn" | "small" | "stats" | "accumulate" | "accept", Pass>;
  private unsafeRows: Int32Array;
  private unsafeRadius: number;
  private hasPrevious = false;
  private lastUpdate: number | null = null;
  private absentSince: number | null = null;
  private autoArmed = true;
  private capture: Capture | null = null;
  last: RoomStats = { knownFraction: 0, coverage: 0, still: false, absent: false };

  constructor(private gl: WebGL2RenderingContext, quadVao: WebGLVertexArrayObject, readonly width: number,
              readonly height: number, maskWidth: number, maskHeight: number) {
    const set = () => new TargetSet(gl, [new Target(gl, width, height, "rgba8", false),
                                         new Target(gl, width, height, "rgba8", false)]);
    this.plates = [set(), set()];
    this.unsafe = new Target(gl, maskWidth, maskHeight, "r8", false);
    const [gw, gh] = C.bg_moved_grid;
    this.smalls = [new Target(gl, gw, gh, "rgba8", false), new Target(gl, gw, gh, "rgba8", false)];
    this.stats = new Target(gl, BLOCKS[0], BLOCKS[1], "rgba32f");
    this.sums = [new Target(gl, width, height, "rgba32f"), new Target(gl, width, height, "rgba32f")];
    this.readback = new Float32Array(BLOCKS[0] * BLOCKS[1] * 4);
    this.unsafeRadius = lowResRadius(C.bg_safe_margin_px, width / maskWidth);
    const rows = ellipseRows(this.unsafeRadius);
    if (rows.length > 64) throw new Error("Safe margin too large for the shader.");
    this.unsafeRows = new Int32Array(64);
    this.unsafeRows.set(rows);
    this.passes = {
      unsafe: new Pass(gl, UNSAFE, quadVao),
      learn: new Pass(gl, LEARN, quadVao),
      small: new Pass(gl, SMALL, quadVao),
      stats: new Pass(gl, STATS, quadVao),
      accumulate: new Pass(gl, ACCUMULATE, quadVao),
      accept: new Pass(gl, ACCEPT, quadVao),
    };
  }

  dispose(): void {
    this.plates.forEach((p) => p.dispose());
    [this.unsafe, ...this.smalls, this.stats, ...this.sums].forEach((t) => t.dispose());
  }

  get plate(): Target {
    return this.plates[this.side].targets[0];
  }

  get state(): Target {
    return this.plates[this.side].targets[1];
  }

  /** Learn from one display frame (update() in background.py). Returns messages for the user. */
  update(now: number, frame: Target, mask: WebGLTexture, maskData: Float32Array, core: Target,
         allowAuto = true): string[] {
    const messages: string[] = [];
    let resetRun = false;
    if (this.lastUpdate !== null && now - this.lastUpdate > C.bg_frame_gap_s) {
      // Updates paused; frames from before the pause are not consecutive.
      resetRun = true;
      this.hasPrevious = false;
    }
    this.lastUpdate = now;
    this.passes.unsafe.run(this.unsafe, {
      mask, rows: this.unsafeRows, radius: this.unsafeRadius, conf: C.bg_safe_conf,
    });
    const from = this.plates[this.side];
    const to = this.plates[1 - this.side];
    this.passes.learn.run(to, {
      frame, plate: from.targets[0], state: from.targets[1], unsafe: this.unsafe, resetRun,
      safeFrames: C.bg_safe_frames, medianStep: C.bg_median_step,
    });
    this.side = 1 - this.side;
    const { still, absent } = this.observe(frame, maskData, core);

    if (this.capture?.auto) {
      if (absent && still) {
        const result = this.addCaptureFrame(frame);
        if (result !== null) {
          this.autoArmed = false;
          messages.push(result === "ok" ? texts.background.learned : texts.background.capture[result]);
        }
      } else {
        this.capture = null;
        this.absentSince = null;
      }
    } else if (this.capture === null) {
      if (!absent) this.autoArmed = true;
      if (absent && still) {
        if (this.absentSince === null) this.absentSince = now;
        else if (allowAuto && this.autoArmed && now - this.absentSince >= C.bg_auto_capture_s) {
          this.capture = { auto: true, n: 0, bad: null };
        }
      } else {
        this.absentSince = null;
      }
    }
    return messages;
  }

  /** Feed one frame of a manual capture (capture_frame). null while recording, then the result. */
  captureFrame(frame: Target, maskData: Float32Array, core: Target): CaptureResult | null {
    if (this.capture === null) this.startCapture();
    const capture = this.capture!;
    const { still, absent } = this.observe(frame, maskData, core);
    if (!absent) capture.bad = "person";
    else if (!still && capture.n > 0 && capture.bad === null) capture.bad = "moving";
    return this.addCaptureFrame(frame);
  }

  startCapture(): void {
    this.capture = { auto: false, n: 0, bad: null };
  }

  cancelCapture(): void {
    this.capture = null;
    this.absentSince = null;
  }

  private addCaptureFrame(frame: Target): CaptureResult | null {
    const capture = this.capture!;
    const into = this.sums[1 - this.sumSide];
    this.passes.accumulate.run(into, { frame, sum: this.sums[this.sumSide], first: capture.n === 0 });
    this.sumSide = 1 - this.sumSide;
    capture.n += 1;
    const needed = capture.auto ? C.bg_auto_capture_frames : C.plate_jumlah_frame;
    if (capture.n < needed) return null;
    this.capture = null;
    if (capture.bad !== null) return capture.bad;
    const from = this.plates[this.side];
    this.passes.accept.run(this.plates[1 - this.side], {
      sum: this.sums[this.sumSide], state: from.targets[1], frames: capture.n, knownSamples: C.bg_known_samples,
    });
    this.side = 1 - this.side;
    return "ok";
  }

  /** Motion and absence of this frame (_observe), plus coverage of the person zone. */
  private observe(frame: Target, maskData: Float32Array, core: Target): RoomStats {
    const small = this.smalls[this.smallSide];
    this.passes.small.run(small, { frame });
    this.passes.stats.run(this.stats, {
      state: this.state, core, small, previous: this.smalls[1 - this.smallSide], hasPrevious: this.hasPrevious,
      knownSamples: C.bg_known_samples,
    });
    this.smallSide = 1 - this.smallSide;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.stats.framebuffer);
    gl.readPixels(0, 0, BLOCKS[0], BLOCKS[1], gl.RGBA, gl.FLOAT, this.readback);
    let known = 0;
    let zone = 0;
    let both = 0;
    let motion = 0;
    for (let i = 0; i < this.readback.length; i += 4) {
      known += this.readback[i];
      zone += this.readback[i + 1];
      both += this.readback[i + 2];
      motion += this.readback[i + 3];
    }
    const [gw, gh] = C.bg_moved_grid;
    const still = this.hasPrevious && motion / (gw * gh) < C.bg_still_diff;
    this.hasPrevious = true;
    let person = 0;
    for (let i = 0; i < maskData.length; i++) if (maskData[i] > C.mask_ambang) person++;
    const absent = person / maskData.length < C.bg_absent_frac;
    const knownFraction = known / (this.width * this.height);
    const coverage = zone > 0 ? both / zone : knownFraction;
    this.last = { knownFraction, coverage, still, absent };
    return this.last;
  }
}
