import { defineConfig } from "@playwright/test";

// The camera is a video file (.y4m), for example made with
//   ffmpeg -i ../uji_lokal/orang.mp4 -frames:v 90 -vf scale=1280:720 -pix_fmt yuv420p orang.y4m
// RETROLENS_FAKE_CAMERA: a person in view all the time. RETROLENS_FAKE_ROOM: the empty room first, then a
// person walks in (the clip loops).
const camera = process.env.RETROLENS_FAKE_CAMERA ?? "";
const room = process.env.RETROLENS_FAKE_ROOM ?? "";
// Headless Chromium draws with a software GPU unless told to use the real one (RETROLENS_REAL_GPU=1 on a Mac).
const realGpu = process.env.RETROLENS_REAL_GPU === "1"
  ? ["--headless=new", "--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=metal"]
  : [];

function chromium(clip: string) {
  return {
    browserName: "chromium" as const,
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        ...(clip ? [`--use-file-for-fake-video-capture=${clip}`] : []),
        ...realGpu,
      ],
    },
  };
}

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  // One browser at a time: several fake cameras and GPUs at once stall each other's frames.
  workers: 1,
  // RETROLENS_URL tests a deployed site instead of a local build.
  use: { baseURL: process.env.RETROLENS_URL ?? "http://localhost:4173" },
  webServer: process.env.RETROLENS_URL
    ? undefined
    : { command: "npm run build && npx vite preview --port 4173 --strictPort", port: 4173, reuseExistingServer: true },
  projects: [
    { name: "person-in-view", testMatch: /(flow|states|capture)\.spec\.ts/, use: chromium(camera) },
    { name: "room-then-person", testMatch: /room\.spec\.ts/, use: chromium(room) },
  ],
});
