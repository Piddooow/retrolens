// Small WebGL2 helpers: full screen passes into render targets.

// Every texture keeps the picture's top row as row 0, and uv.y = row / height. A pass into a texture writes
// row y at gl_FragCoord.y = y + 0.5, so uv follows gl_FragCoord directly; only the canvas (whose row 0 is the
// bottom of the screen) needs uv flipped, so the picture shows upright.
const VERTEX = `#version 300 es
in vec2 corner;
out vec2 uv;
void main() {
  uv = corner * 0.5 + 0.5;
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

const VERTEX_SCREEN = `#version 300 es
in vec2 corner;
out vec2 uv;
void main() {
  uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

export type Format = "rgba8" | "r8" | "r16f" | "rgba16f" | "rgba32f";

const FORMATS: Record<Format, [number, number, number]> = {
  rgba8: [WebGL2RenderingContext.RGBA8, WebGL2RenderingContext.RGBA, WebGL2RenderingContext.UNSIGNED_BYTE],
  r8: [WebGL2RenderingContext.R8, WebGL2RenderingContext.RED, WebGL2RenderingContext.UNSIGNED_BYTE],
  r16f: [WebGL2RenderingContext.R16F, WebGL2RenderingContext.RED, WebGL2RenderingContext.HALF_FLOAT],
  rgba16f: [WebGL2RenderingContext.RGBA16F, WebGL2RenderingContext.RGBA, WebGL2RenderingContext.HALF_FLOAT],
  rgba32f: [WebGL2RenderingContext.RGBA32F, WebGL2RenderingContext.RGBA, WebGL2RenderingContext.FLOAT],
};

/** A texture with its framebuffer. 32-bit float textures are sampled NEAREST (they are not filterable). */
export class Target {
  readonly texture: WebGLTexture;
  readonly framebuffer: WebGLFramebuffer;

  constructor(private gl: WebGL2RenderingContext, readonly width: number, readonly height: number,
              readonly format: Format, linear = true) {
    const [internal, layout, type] = FORMATS[format];
    this.texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, layout, type, null);
    const filter = linear && format !== "rgba32f" ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`This GPU cannot render to ${format} (${status}).`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
    this.gl.deleteFramebuffer(this.framebuffer);
  }
}

/** Several targets written by one pass (multiple render targets). */
export class TargetSet {
  readonly framebuffer: WebGLFramebuffer;

  constructor(private gl: WebGL2RenderingContext, readonly targets: Target[]) {
    this.framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    targets.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D,
                                                       t.texture, 0));
    gl.drawBuffers(targets.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Cannot render to these targets (${status}).`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  get width(): number {
    return this.targets[0].width;
  }

  dispose(): void {
    this.targets.forEach((t) => t.dispose());
    this.gl.deleteFramebuffer(this.framebuffer);
  }

  get height(): number {
    return this.targets[0].height;
  }
}

type Uniform = number | boolean | number[] | Float32Array | Int32Array | WebGLTexture | Target;

/** One fragment shader run over a whole target. Textures bind to units in the order they are given. */
export class Pass {
  private program: WebGLProgram;
  private locations = new Map<string, WebGLUniformLocation | null>();
  private kinds = new Map<string, number>();

  /** screen: the pass draws on the canvas (uv flipped so the picture is upright). */
  constructor(private gl: WebGL2RenderingContext, fragment: string, private quad: WebGLVertexArrayObject,
              screen = false) {
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, screen ? VERTEX_SCREEN : VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
    gl.bindAttribLocation(program, 0, "corner");
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Shader link failed: ${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i)!;
      const name = info.name.replace(/\[0\]$/, "");
      this.kinds.set(name, info.type);
      this.locations.set(name, gl.getUniformLocation(program, info.name));
    }
  }

  /** Draw into target (null = the canvas, at width x height). */
  run(target: Target | TargetSet | null, uniforms: Record<string, Uniform>, width = 0, height = 0): void {
    const gl = this.gl;
    gl.useProgram(this.program);
    let unit = 0;
    for (const [name, value] of Object.entries(uniforms)) {
      const location = this.locations.get(name);
      if (location === undefined || location === null) continue;
      const kind = this.kinds.get(name)!;
      if (kind === gl.SAMPLER_2D) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, value instanceof Target ? value.texture : (value as WebGLTexture));
        gl.uniform1i(location, unit++);
      } else if (kind === gl.FLOAT) {
        if (typeof value === "number" || typeof value === "boolean") gl.uniform1f(location, Number(value));
        else gl.uniform1fv(location, value as Float32Array);
      } else if (kind === gl.INT || kind === gl.BOOL) {
        if (typeof value === "number" || typeof value === "boolean") gl.uniform1i(location, Number(value));
        else gl.uniform1iv(location, value as Int32Array);
      } else if (kind === gl.INT_VEC2) gl.uniform2iv(location, value as number[]);
      else if (kind === gl.FLOAT_VEC2) gl.uniform2fv(location, value as number[]);
      else if (kind === gl.FLOAT_VEC3) gl.uniform3fv(location, value as number[]);
      else if (kind === gl.FLOAT_VEC4) gl.uniform4fv(location, value as number[]);
    }
    if (target === null) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
    }
    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

export function quad(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

/** A texture filled from the CPU each frame (camera bitmap or the person mask). */
export function uploadTexture(gl: WebGL2RenderingContext, linear: boolean): WebGLTexture {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const filter = linear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}
