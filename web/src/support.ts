// What this browser can do, and which browser and system it is (for help texts that name the right menus).

export type BrowserName = "chrome" | "edge" | "safari" | "firefox" | "other";
export type SystemName = "mac" | "windows" | "other";

export interface Platform {
  browser: BrowserName;
  system: SystemName;
  /** A phone or tablet: the app is made for a laptop or desktop camera. */
  handheld: boolean;
}

export function platform(userAgent = navigator.userAgent): Platform {
  const ua = userAgent;
  let browser: BrowserName = "other";
  if (/Edg\//.test(ua)) browser = "edge";
  else if (/Firefox\//.test(ua)) browser = "firefox";
  else if (/Chrome\//.test(ua) || /CriOS\//.test(ua)) browser = "chrome";
  else if (/Safari\//.test(ua)) browser = "safari";
  const system: SystemName = /Windows/.test(ua) ? "windows" : /Macintosh|Mac OS X/.test(ua) ? "mac" : "other";
  const handheld = /Android|iPhone|iPad|iPod|Mobile/.test(ua);
  return { browser, system, handheld };
}

/** The first missing feature the app needs, as a sentence for the user, or null when everything is there. */
export function missingFeature(): string | null {
  if (!window.isSecureContext) return "This page has to be opened over https for the camera to work.";
  if (!navigator.mediaDevices?.getUserMedia) return "This browser cannot use a camera.";
  if (typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined" || typeof createImageBitmap === "undefined") {
    return "This browser is too old for Retrolens.";
  }
  const gl = document.createElement("canvas").getContext("webgl2");
  if (!gl) return "This browser has no WebGL2, which Retrolens draws its effects with.";
  return null;
}
