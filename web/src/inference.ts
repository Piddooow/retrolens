// The page side of the model worker. One frame is in the worker at a time; while it is busy, newer camera
// frames are skipped, so results never lag behind (the latest frame wins, like snap/worker.py).

import type { FromWorker, ResultMessage, ToWorker } from "./protocol";

export interface InferenceEvents {
  onProgress(loaded: number, total: number): void;
  onReady(delegate: string): void;
  onError(message: string): void;
  onResult(result: ResultMessage): void;
}

export class Inference {
  private worker: Worker;
  private busy = false;
  private nextId = 0;

  constructor(base: string, maskEma: number, events: InferenceEvents) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<FromWorker>) => {
      const message = event.data;
      if (message.type === "progress") events.onProgress(message.loaded, message.total);
      else if (message.type === "ready") events.onReady(message.delegate);
      else if (message.type === "error") {
        this.busy = false;
        events.onError(message.message);
      } else {
        this.busy = false;
        events.onResult(message);
      }
    };
    this.worker.onerror = (event) => events.onError(event.message || "The model worker stopped.");
    this.send({ type: "init", base, maskEma });
  }

  /** Hand a camera frame to the worker unless it is still busy with the previous one. */
  async offer(video: HTMLVideoElement, ts: number): Promise<boolean> {
    if (this.busy || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return false;
    this.busy = true;
    try {
      const bitmap = await createImageBitmap(video);
      this.send({ type: "frame", id: this.nextId++, ts, bitmap }, [bitmap]);
      return true;
    } catch {
      this.busy = false;
      return false;
    }
  }

  reset(): void {
    this.send({ type: "reset" });
  }

  close(): void {
    this.worker.terminate();
  }

  private send(message: ToWorker, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }
}
