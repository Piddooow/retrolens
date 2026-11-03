// The guided flow: landing, camera permission, model download, then the camera stage.

import configData from "./data/config.json";
import { cameraProblem, openCamera, stopCamera } from "./camera";
import { Inference } from "./inference";
import type { ResultMessage } from "./protocol";
import { Stage } from "./stage";
import { missingFeature, platform } from "./support";

const CONFIG = configData.config;

function $<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const pages = ["landing", "camera-step", "loading", "stage"].map((id) => $(id));

function show(id: string): void {
  for (const page of pages) page.hidden = page.id !== id;
  // Move focus to the new page's heading or first control, so keyboard and screen reader users follow along.
  const target = $(id).querySelector<HTMLElement>("h1, h2, button");
  if (target) {
    if (!target.matches("button")) target.tabIndex = -1;
    target.focus({ preventScroll: true });
  }
}

const where = platform();
const video = $<HTMLVideoElement>("video");
let stream: MediaStream | null = null;
let inference: Inference | null = null;
let stage: Stage | null = null;
let modelsReady = false;
let modelsError: string | null = null;
let delegate = "";
let running = false;
let resultTimes: number[] = [];

function startModels(): void {
  if (inference) return;
  const base = new URL(import.meta.env.BASE_URL, location.href).href.replace(/\/$/, "");
  inference = new Inference(base, CONFIG.mask_ema_baru, {
    onProgress(loaded, total) {
      const percent = total > 0 ? Math.min(100, Math.round((100 * loaded) / total)) : 0;
      $("bar-fill").style.width = `${percent}%`;
      $("loading").querySelector(".bar")!.setAttribute("aria-valuenow", String(percent));
      $("loading-text").textContent = total > 0
        ? `${(loaded / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`
        : `${(loaded / 1e6).toFixed(1)} MB`;
    },
    onReady(usedDelegate) {
      modelsReady = true;
      delegate = usedDelegate;
      $("loading-text").textContent = "Ready";
      if (stream) enterStage();
    },
    onError(message) {
      modelsError = message;
      if (running) {
        stopRunning();
        show("loading");
      }
      showLoadingProblem();
    },
    onResult(result) {
      draw(result);
    },
  });
}

function showLoadingProblem(): void {
  if (!modelsError) return;
  $("loading-error").textContent = modelsError;
  $("loading-problem").hidden = false;
}

async function askForCamera(): Promise<void> {
  show("camera-step");
  $("camera-problem").hidden = true;
  $("camera-ask").hidden = false;
  try {
    stream = await openCamera(CONFIG.kamera_lebar, CONFIG.kamera_tinggi);
    video.srcObject = stream;
    await video.play();
  } catch (error) {
    stopCamera(stream);
    stream = null;
    const problem = cameraProblem(error, where);
    $("problem-title").textContent = problem.title;
    $("problem-steps").replaceChildren(...problem.steps.map((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      return li;
    }));
    $("camera-ask").hidden = true;
    $("camera-problem").hidden = false;
    $("retry").focus();
    return;
  }
  if (modelsReady) enterStage();
  else {
    show("loading");
    showLoadingProblem();
  }
}

function enterStage(): void {
  show("stage");
  if (!stage) stage = new Stage($("gl"), $("overlay"));
  running = true;
  resultTimes = [];
  inference?.reset();
  requestAnimationFrame(tick);
}

let lastVideoTime = -1;

function tick(): void {
  if (!running) return;
  // Offer each new camera frame once; while the worker is busy the frame is skipped (latest frame wins).
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    void inference?.offer(video, performance.now());
  }
  requestAnimationFrame(tick);
}

function draw(result: ResultMessage): void {
  if (!running || !stage) {
    result.bitmap.close();
    return;
  }
  stage.draw(result);
  result.bitmap.close();
  const now = performance.now();
  resultTimes.push(now);
  while (resultTimes.length && resultTimes[0] < now - 1000) resultTimes.shift();
  const model = Math.round(result.handMs + result.segmenterMs);
  $("status").textContent = `${resultTimes.length} fps · models ${model} ms on the ${delegate}`;
  const hands = $("hands");
  hands.hidden = !stage.showDebug;
  if (stage.showDebug) {
    hands.textContent = result.hands.length
      ? result.hands.map((h, i) => `Hand ${i + 1}: ${h.gesture} ${h.score.toFixed(2)}`).join(" · ")
      : "No hands in view";
  }
}

function stopRunning(): void {
  running = false;
  stopCamera(stream);
  stream = null;
  video.srcObject = null;
}

function closeCamera(): void {
  stopRunning();
  show("landing");
}

function toggleDebug(): void {
  if (!stage) return;
  stage.showDebug = !stage.showDebug;
  $("debug").setAttribute("aria-pressed", String(stage.showDebug));
}

function setUp(): void {
  const missing = missingFeature();
  if (missing) {
    $("start-area").hidden = true;
    $("unsupported-text").textContent = missing;
    $("unsupported").hidden = false;
  } else if (where.handheld) {
    $("start-area").hidden = true;
    $("handheld").hidden = false;
  }
  $("handheld-anyway").addEventListener("click", () => {
    $("handheld").hidden = true;
    $("start-area").hidden = false;
    $("start").focus();
  });
  $("start").addEventListener("click", () => {
    startModels();
    void askForCamera();
  });
  $("retry").addEventListener("click", () => void askForCamera());
  $("camera-back").addEventListener("click", () => show("landing"));
  $("reload").addEventListener("click", () => location.reload());
  $("close").addEventListener("click", closeCamera);
  $("debug").addEventListener("click", toggleDebug);
  document.addEventListener("keydown", (event) => {
    if (!running || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "d" || event.key === "D") toggleDebug();
    else if (event.key === "Escape") closeCamera();
  });
}

setUp();
