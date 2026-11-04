// The SNAP state machine, a port of snap/controller.py SnapController. This first part covers LIVE and
// PLATE_CAPTURE (learning the room, automatically or after a countdown); the effect states come with the
// effects. All times are seconds, from the frame's capture time.

import configData from "./data/config.json";
import texts from "./data/texts.json";
import type { Matte } from "./gl/matte";
import type { Room } from "./gl/room";
import type { Target } from "./gl/gpu";

const C = configData.config;

export type State = "LIVE" | "PLATE_CAPTURE";

export class Controller {
  state: State = "LIVE";
  /** The room behind the person is known well enough to vanish (coverage of the zone, bg_coverage_min). */
  bgReady = false;
  bgPercent = 0;
  private captureAt = 0;
  private text = "";
  private textUntil = -Infinity;

  constructor(private room: Room, private matte: Matte) {}

  setMessage(text: string, now: number, seconds: number = C.pesan_detik): void {
    this.text = text;
    this.textUntil = now + seconds;
  }

  message(now: number): string {
    return now < this.textUntil ? this.text : "";
  }

  /** B (or a thumbs up later): count down, then capture the empty room. Only from LIVE. */
  startRoomCapture(now: number): void {
    if (this.state !== "LIVE") return;
    this.state = "PLATE_CAPTURE";
    this.captureAt = now + C.plate_hitung_mundur;
    this.room.cancelCapture();
  }

  countdownLeft(now: number): number {
    return Math.max(0, Math.ceil(this.captureAt - now));
  }

  /** One frame (amati): refine the matte, then learn the room or record the capture. */
  observe(now: number, frame: Target, mask: WebGLTexture, maskData: Float32Array): void {
    this.matte.update(frame, mask);
    const core = this.matte.core;
    if (this.state === "LIVE") {
      for (const text of this.room.update(now, frame, mask, maskData, core)) this.setMessage(text, now, C.pesan_detik * 2);
    } else if (now < this.captureAt) {
      this.room.update(now, frame, mask, maskData, core, false);
    } else {
      const result = this.room.captureFrame(frame, maskData, core);
      if (result !== null) {
        this.setMessage(texts.background.capture[result], now, C.pesan_detik * 2);
        this.state = "LIVE";
      }
    }
    const coverage = this.room.last.coverage;
    this.bgReady = coverage >= C.bg_coverage_min;
    this.bgPercent = Math.floor(100 * coverage);
  }
}
