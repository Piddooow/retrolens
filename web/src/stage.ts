// The GPU side of one camera frame: the picture mirrored into a display size texture (like the Python app
// flips the camera frame first), the person mask uploaded, and the result drawn on the canvas with the debug
// views of main.py (D cycles them).

import configData from "./data/config.json";
import { Pass, Target, quad, uploadTexture } from "./gl/gpu";
import { Matte } from "./gl/matte";
import { Room } from "./gl/room";
import type { ResultMessage } from "./protocol";

const C = configData.config;

/** The D key cycles through these (snap mode's DEBUG_VIEWS without the stats and gain views for now). */
export const VIEWS = ["off", "matte", "zone", "coverage"] as const;
export type View = (typeof VIEWS)[number];

const FLIP = `#version 300 es
precision highp float;
uniform sampler2D raw;
out vec4 color;
void main() {
  ivec2 size = textureSize(raw, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  color = vec4(texelFetch(raw, ivec2(size.x - 1 - p.x, p.y), 0).rgb, 1.0);
}`;

// Debug colors from main.py (BGR there): matte (0, 200, 0), zone (200, 0, 200), unseen room (0, 0, 220).
const DISPLAY = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
uniform sampler2D matte;
uniform sampler2D core;
uniform sampler2D state;
uniform int view;
uniform float alpha;
uniform float knownSamples;
out vec4 color;
void main() {
  vec3 rgb = texture(frame, uv).rgb;
  if (view == 1) {
    rgb = mix(rgb, vec3(0.0, 0.784, 0.0), alpha * texture(matte, uv).r);
  } else if (view == 2) {
    rgb = mix(rgb, vec3(0.784, 0.0, 0.784), alpha * step(0.5, texture(core, uv).r));
  } else if (view == 3) {
    ivec2 size = textureSize(state, 0);
    ivec2 p = min(ivec2(uv * vec2(size)), size - 1);
    bool known = round(texelFetch(state, p, 0).r * 255.0) >= knownSamples;
    rgb = mix(rgb, vec3(0.863, 0.0, 0.0), known ? 0.0 : alpha);
  }
  color = vec4(rgb, 1.0);
}`;

export class Engine {
  readonly gl: WebGL2RenderingContext;
  private vao: WebGLVertexArrayObject;
  private raw: WebGLTexture;
  readonly mask: WebGLTexture;
  frame: Target | null = null;
  room: Room | null = null;
  matte: Matte | null = null;
  maskData: Float32Array = new Float32Array(0);
  private flip: Pass;
  private display: Pass;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL2 is not available.");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("This GPU cannot render to float textures.");
    this.gl = gl;
    this.vao = quad(gl);
    this.raw = uploadTexture(gl, false);
    this.mask = uploadTexture(gl, true);
    this.flip = new Pass(gl, FLIP, this.vao);
    this.display = new Pass(gl, DISPLAY, this.vao, true);
  }

  /** Upload one result's frame and mask. Builds the room and matte for a new frame size (returns true then). */
  load(result: ResultMessage): boolean {
    const gl = this.gl;
    const { bitmap } = result;
    let fresh = false;
    if (!this.frame || this.frame.width !== bitmap.width || this.frame.height !== bitmap.height) {
      this.forget();
      this.frame = new Target(gl, bitmap.width, bitmap.height, "rgba8");
      this.room = new Room(gl, this.vao, bitmap.width, bitmap.height, result.maskWidth, result.maskHeight);
      this.matte = new Matte(gl, this.vao, bitmap.width, bitmap.height, result.maskWidth, result.maskHeight);
      this.canvas.width = bitmap.width;
      this.canvas.height = bitmap.height;
      fresh = true;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.raw);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    this.flip.run(this.frame, { raw: this.raw });
    gl.bindTexture(gl.TEXTURE_2D, this.mask);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, result.maskWidth, result.maskHeight, 0, gl.RED, gl.FLOAT, result.mask);
    this.maskData = result.mask;
    return fresh;
  }

  /** Drop the learned room and the matte; the next frame starts them over. */
  forget(): void {
    this.frame?.dispose();
    this.room?.dispose();
    this.matte?.dispose();
    this.frame = null;
    this.room = null;
    this.matte = null;
  }

  draw(view: View): void {
    if (!this.frame || !this.room || !this.matte) return;
    this.display.run(null, {
      frame: this.frame, matte: this.matte.matte, core: this.matte.core, state: this.room.state,
      view: VIEWS.indexOf(view), alpha: C.debug_alpha, knownSamples: C.bg_known_samples,
    }, this.frame.width, this.frame.height);
  }
}
