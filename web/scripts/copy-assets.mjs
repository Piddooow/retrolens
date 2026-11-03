// Copies the MediaPipe wasm loader (module build) into public/wasm, so the app serves it itself instead of
// from a CDN, and writes public/assets.json with the byte size of every file the app downloads (servers do
// not always send a content-length, and the progress bar needs the total). The models are committed in
// public/models.
import { copyFileSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = dirname(dirname(fileURLToPath(import.meta.url)));
const pub = join(web, "public");
const from = join(web, "node_modules", "@mediapipe", "tasks-vision", "wasm");
mkdirSync(join(pub, "wasm"), { recursive: true });
for (const name of ["vision_wasm_module_internal.js", "vision_wasm_module_internal.wasm"]) {
  copyFileSync(join(from, name), join(pub, "wasm", name));
}
const files = [
  "models/gesture_recognizer.task",
  "models/selfie_segmenter.tflite",
  "wasm/vision_wasm_module_internal.wasm",
];
const sizes = Object.fromEntries(files.map((path) => [path, statSync(join(pub, path)).size]));
writeFileSync(join(pub, "assets.json"), JSON.stringify(sizes, null, 1) + "\n");
