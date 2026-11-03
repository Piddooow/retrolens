// Runs the hand and person models off the main thread (their calls block the thread they run on).
// Like snap/worker.py: one frame at a time, and every result goes back with the very frame it came from.

import { FilesetResolver, GestureRecognizer, ImageSegmenter } from "@mediapipe/tasks-vision";
import type { FrameMessage, FromWorker, Hand, InitMessage, ToWorker } from "./protocol";

const MODELS = ["models/gesture_recognizer.task", "models/selfie_segmenter.tflite"];
const WASM = "wasm/vision_wasm_module_internal.wasm";
const LOADER = "wasm/vision_wasm_module_internal.js";

let gestures: GestureRecognizer | null = null;
let segmenter: ImageSegmenter | null = null;
let maskEma = 0.6;
let mask: Float32Array | null = null;
let lastTs = -1;
// The models see the mirrored picture, like the Python app (it flips the camera frame first).
const full = new OffscreenCanvas(1, 1);
const half = new OffscreenCanvas(1, 1);
const fullCtx = full.getContext("2d", { willReadFrequently: false })!;
const halfCtx = half.getContext("2d")!;

function post(message: FromWorker, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

/** Download the files with one progress bar; the wasm request warms the cache for the loader's own fetch.
 * The sizes come from assets.json (written at build time), as servers do not always send a length. */
async function download(base: string): Promise<Uint8Array[]> {
  const paths = [...MODELS, WASM];
  const sizesResponse = await fetch(`${base}/assets.json`);
  if (!sizesResponse.ok) throw new Error(`Could not download ${base}/assets.json (${sizesResponse.status})`);
  const known = (await sizesResponse.json()) as Record<string, number>;
  const urls = paths.map((path) => `${base}/${path}`);
  const responses = await Promise.all(urls.map((url) => fetch(url)));
  responses.forEach((r, i) => {
    if (!r.ok) throw new Error(`Could not download ${urls[i]} (${r.status})`);
  });
  const total = paths.reduce((sum, path) => sum + (known[path] ?? 0), 0);
  const loaded = paths.map(() => 0);
  return Promise.all(
    responses.map(async (r, i) => {
      const reader = r.body!.getReader();
      const parts: Uint8Array[] = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        loaded[i] += value.length;
        post({ type: "progress", loaded: loaded.reduce((a, b) => a + b, 0), total });
      }
      const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
      let at = 0;
      for (const p of parts) {
        out.set(p, at);
        at += p.length;
      }
      return out;
    }),
  );
}

/**
 * The module build of the wasm loader sets globalThis.ModuleFactory when it is first evaluated, and every
 * task creation clears it after use. A second task imports the same module again, but the browser only
 * evaluates a module once, so MediaPipe would fail with "ModuleFactory not set". Importing it here once and
 * putting the factory back before each task keeps both tasks working (checked against tasks-vision 1.1.1,
 * the pinned version).
 */
let factory: unknown = null;

async function withFactory<T>(base: string, make: () => Promise<T>): Promise<T> {
  if (!factory) factory = (await import(/* @vite-ignore */ `${base}/${LOADER}`)).default;
  (globalThis as unknown as { ModuleFactory: unknown }).ModuleFactory = factory;
  return make();
}

async function create(base: string, buffers: Uint8Array[], delegate: "GPU" | "CPU"): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(`${base}/wasm`, true);
  gestures = await withFactory(base, () => GestureRecognizer.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: buffers[0], delegate },
    runningMode: "VIDEO",
    numHands: 2,
    // The same thresholds as the Python app (main.py buka_model_tangan).
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,
  }));
  segmenter = await withFactory(base, () => ImageSegmenter.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: buffers[1], delegate },
    runningMode: "VIDEO",
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  }));
}

async function init(message: InitMessage): Promise<void> {
  maskEma = message.maskEma;
  try {
    const buffers = await download(message.base);
    try {
      await create(message.base, buffers, "GPU");
      post({ type: "ready", delegate: "GPU" });
    } catch {
      // Some GPUs or browsers cannot run the models on the GPU; the CPU is slower but works everywhere.
      gestures?.close();
      segmenter?.close();
      await create(message.base, buffers, "CPU");
      post({ type: "ready", delegate: "CPU" });
    }
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
}

function run(message: FrameMessage): void {
  if (!gestures || !segmenter) return;
  const { bitmap } = message;
  const w = bitmap.width;
  const h = bitmap.height;
  if (full.width !== w || full.height !== h) {
    full.width = w;
    full.height = h;
    half.width = Math.round(w / 2);
    half.height = Math.round(h / 2);
    mask = null;
  }
  fullCtx.setTransform(-1, 0, 0, 1, w, 0);
  fullCtx.drawImage(bitmap, 0, 0);
  halfCtx.drawImage(full, 0, 0, half.width, half.height);
  // Each task needs strictly increasing timestamps.
  const ts = Math.max(lastTs + 1, Math.round(message.ts));
  lastTs = ts;

  let start = performance.now();
  const recognized = gestures.recognizeForVideo(full, ts);
  const handMs = performance.now() - start;
  const hands: Hand[] = recognized.landmarks.map((points, i) => {
    const landmarks = new Float32Array(42);
    points.forEach((p, j) => {
      landmarks[2 * j] = p.x;
      landmarks[2 * j + 1] = p.y;
    });
    const top = recognized.gestures[i]?.[0];
    return { landmarks, gesture: top?.categoryName || "None", score: top?.score ?? 0 };
  });

  start = performance.now();
  const segmented = segmenter.segmentForVideo(half, ts);
  const confidence = segmented.confidenceMasks![0].getAsFloat32Array();
  if (!mask || mask.length !== confidence.length) {
    mask = new Float32Array(confidence);
  } else {
    for (let i = 0; i < mask.length; i++) mask[i] = maskEma * confidence[i] + (1 - maskEma) * mask[i];
  }
  segmented.close();
  const segmenterMs = performance.now() - start;

  const copy = new Float32Array(mask);
  post(
    {
      type: "result", id: message.id, ts: message.ts, bitmap, hands, mask: copy,
      maskWidth: half.width, maskHeight: half.height, handMs, segmenterMs,
    },
    [bitmap, copy.buffer, ...hands.map((hand) => hand.landmarks.buffer)],
  );
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  if (message.type === "init") void init(message);
  else if (message.type === "frame") {
    try {
      run(message);
    } catch (error) {
      // The page waits for every frame it sent; say what failed instead of leaving it waiting.
      message.bitmap.close();
      post({ type: "error", message: error instanceof Error ? error.message : String(error) });
    }
  } else if (message.type === "reset") mask = null;
};
