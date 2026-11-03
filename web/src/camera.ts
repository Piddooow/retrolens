// Opens the camera and turns every way it can fail into steps the user can follow.

import type { Platform } from "./support";

export interface CameraProblem {
  title: string;
  steps: string[];
}

export async function openCamera(width: number, height: number): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: 30 } },
  });
}

export function stopCamera(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/** What went wrong and what to do about it, for this browser and system. */
export function cameraProblem(error: unknown, where: Platform): CameraProblem {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    const steps: string[] = [];
    if (where.browser === "safari") {
      steps.push("In the menu bar, open Safari, then Settings for This Website, and set Camera to Allow.");
    } else if (where.browser === "firefox") {
      steps.push("Click the camera icon at the left end of the address bar and remove the block.");
    } else {
      steps.push("Click the camera icon at the right end of the address bar and choose Always allow.");
    }
    if (where.system === "mac") {
      steps.push("If that does not help: open System Settings, Privacy & Security, Camera, turn on your browser, "
        + "then quit and reopen the browser.");
    } else if (where.system === "windows") {
      steps.push("If that does not help: open Settings, Privacy & security, Camera, and turn on Camera access "
        + "and Let desktop apps access your camera.");
    }
    steps.push("Press Try again.");
    return { title: "The camera is blocked for this page", steps };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return { title: "No camera found", steps: ["Connect a camera, then press Try again."] };
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return {
      title: "Another app is using the camera",
      steps: ["Close the app that has it (a video call, for example), then press Try again."],
    };
  }
  return {
    title: "The camera could not start",
    steps: [`The browser said: ${error instanceof Error ? error.message : String(error)}`, "Press Try again."],
  };
}
