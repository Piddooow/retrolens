// The guided flow: landing, camera permission, model download, then the camera stage.

import configData from "./data/config.json";
import { cameraProblem, openCamera, stopCamera } from "./camera";
import { Controller } from "./controller";
import { Inference } from "./inference";
import type { ResultMessage } from "./protocol";
import { Engine, VIEWS, type View } from "./stage";
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
let engine: Engine | null = null;
let controller: Controller | null = null;
let view: View = "off";
let modelsReady = false;
let modelsError: string | null = null;
let delegate = "";
let running = false;
let resultTimes: number[] = [];
/** Capture time of the last processed frame, in seconds: the clock of the controller. */
let lastNow = 0;

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
      process(result);
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
  if (!engine) {
    try {
      engine = new Engine($("gl"));
    } catch (error) {
      modelsError = error instanceof Error ? error.message : String(error);
      stopRunning();
      show("loading");
      showLoadingProblem();
      return;
    }
  }
  running = true;
  resultTimes = [];
  inference?.reset();
  // A new session learns the room from scratch: the camera may point somewhere else now.
  controller = null;
  engine.forget();
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

function process(result: ResultMessage): void {
  if (!running || !engine) {
    result.bitmap.close();
    return;
  }
  const now = result.ts / 1000;
  lastNow = now;
  if (engine.load(result) || !controller) controller = new Controller(engine.room!, engine.matte!);
  result.bitmap.close();
  controller.observe(now, engine.frame!, engine.mask, engine.maskData);
  engine.draw(view);
  drawHands(result);
  updateHud(now, result);
}

function drawHands(result: ResultMessage): void {
  const canvas = $<HTMLCanvasElement>("overlay");
  const ctx = canvas.getContext("2d")!;
  if (canvas.width !== result.bitmap.width || canvas.height !== result.bitmap.height) {
    canvas.width = result.bitmap.width;
    canvas.height = result.bitmap.height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (view === "off") return;
  ctx.fillStyle = "#33D1F7";
  for (const hand of result.hands) {
    for (const i of [0, 4, 8, 9, 12, 16, 20]) {
      ctx.beginPath();
      ctx.arc(hand.landmarks[2 * i] * canvas.width, hand.landmarks[2 * i + 1] * canvas.height, 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function updateHud(now: number, result: ResultMessage): void {
  const c = controller!;
  const time = performance.now();
  resultTimes.push(time);
  while (resultTimes.length && resultTimes[0] < time - 1000) resultTimes.shift();
  const status = $("status");
  const room = document.createElement("span");
  room.className = c.bgReady ? "ready" : "learning";
  room.textContent = c.bgReady ? "Ready" : `Learning ${c.bgPercent}%`;
  status.replaceChildren(room, ` · ${resultTimes.length} fps`);

  const label = $("view-label");
  label.hidden = view === "off";
  if (view !== "off") {
    const hands = result.hands.length
      ? result.hands.map((h, i) => `hand ${i + 1}: ${h.gesture} ${h.score.toFixed(2)}`).join(", ")
      : "no hands";
    const model = Math.round(result.handMs + result.segmenterMs);
    label.textContent = `Debug view: ${view} (D for next) · ${hands} · models ${model} ms on the ${delegate}`;
  }

  const center = $("center");
  if (c.state === "PLATE_CAPTURE") {
    const left = c.countdownLeft(now);
    center.textContent = left > 0 ? `Step out of the frame ${left}` : "Learning the room...";
    center.hidden = false;
  } else {
    center.hidden = true;
  }

  const toast = $("toast");
  const message = c.message(now);
  toast.hidden = !message;
  if (message && toast.textContent !== message) toast.textContent = message;

  $("guide-title").textContent = c.bgReady ? "Step 1 of 3: done" : "Step 1 of 3: learn the room";
  $("guide-text").textContent = c.bgReady
    ? "The room is learned. Step back in. Picking an effect and vanishing come in the next update."
    : "Step out of the camera's view and keep still for a second. Retrolens remembers the room behind you, "
      + "so it can show the room when you vanish.";
  $("room-progress").hidden = c.bgReady;
  $("room-fill").style.width = `${c.bgPercent}%`;
  $("room-progress").querySelector(".bar")!.setAttribute("aria-valuenow", String(c.bgPercent));
  $("room-percent").textContent = `Room learned: ${c.bgPercent}%`;
  const learn = $<HTMLButtonElement>("learn");
  learn.hidden = c.bgReady;
  learn.disabled = c.state !== "LIVE";
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

function nextView(): void {
  view = VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length];
}

function learnRoom(): void {
  controller?.startRoomCapture(lastNow);
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
  $("debug").addEventListener("click", nextView);
  $("learn").addEventListener("click", learnRoom);
  document.addEventListener("keydown", (event) => {
    if (!running || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "d") nextView();
    else if (key === "b") learnRoom();
    else if (key === "escape") closeCamera();
  });
}

setUp();
