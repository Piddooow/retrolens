// Draws the camera picture (mirrored, like the Python app) with WebGL2, plus the debug view: the person
// mask as a green tint and the hand landmarks. The effects of the later phases draw here too.

import type { ResultMessage } from "./protocol";

const VERTEX = `#version 300 es
in vec2 corner;
out vec2 uv;
void main() {
  // Texture rows start at the top of the picture, so the top of the screen samples v = 0.
  uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

// The frame is the camera as captured, so it is sampled mirrored; the mask was made from the mirrored
// picture already.
const FRAGMENT = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
uniform sampler2D mask;
uniform float showMask;
out vec4 color;
void main() {
  vec3 rgb = texture(frame, vec2(1.0 - uv.x, uv.y)).rgb;
  float person = texture(mask, uv).r;
  color = vec4(mix(rgb, mix(rgb, vec3(0.0, 0.63, 0.0), 0.4), showMask * step(0.5, person)), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "shader");
  return shader;
}

export class Stage {
  showDebug = false;
  private gl: WebGL2RenderingContext;
  private overlay: CanvasRenderingContext2D;
  private frame: WebGLTexture;
  private mask: WebGLTexture;
  private showMaskAt: WebGLUniformLocation;

  constructor(private canvas: HTMLCanvasElement, private overlayCanvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL2 is not available.");
    this.gl = gl;
    this.overlay = overlayCanvas.getContext("2d")!;
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "link");
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(program, "corner");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    this.frame = this.texture(0);
    this.mask = this.texture(1);
    gl.uniform1i(gl.getUniformLocation(program, "frame"), 0);
    gl.uniform1i(gl.getUniformLocation(program, "mask"), 1);
    this.showMaskAt = gl.getUniformLocation(program, "showMask")!;
  }

  draw(result: ResultMessage): void {
    const gl = this.gl;
    const { bitmap } = result;
    if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
      this.canvas.width = this.overlayCanvas.width = bitmap.width;
      this.canvas.height = this.overlayCanvas.height = bitmap.height;
    }
    gl.viewport(0, 0, bitmap.width, bitmap.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frame);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.mask);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, result.maskWidth, result.maskHeight, 0, gl.RED, gl.FLOAT, result.mask);
    gl.uniform1f(this.showMaskAt, this.showDebug ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.drawOverlay(result);
  }

  private drawOverlay(result: ResultMessage): void {
    const ctx = this.overlay;
    const { width, height } = this.overlayCanvas;
    ctx.clearRect(0, 0, width, height);
    if (!this.showDebug) return;
    ctx.fillStyle = "#33D1F7";
    for (const hand of result.hands) {
      for (const i of [0, 4, 8, 9, 12, 16, 20]) {
        ctx.beginPath();
        ctx.arc(hand.landmarks[2 * i] * width, hand.landmarks[2 * i + 1] * height, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  private texture(unit: number): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }
}
