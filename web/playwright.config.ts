import { defineConfig } from "@playwright/test";

// The camera is a video file: RETROLENS_FAKE_CAMERA points to a .y4m clip, for example made with
// ffmpeg -i ../uji_lokal/orang.mp4 -frames:v 90 -vf scale=1280:720 -pix_fmt yuv420p orang.y4m
const camera = process.env.RETROLENS_FAKE_CAMERA ?? "";
// Headless Chromium draws with a software GPU unless told to use the real one (RETROLENS_REAL_GPU=1 on a Mac).
const realGpu = process.env.RETROLENS_REAL_GPU === "1"
  ? ["--headless=new", "--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=metal"]
  : [];

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  use: { baseURL: "http://localhost:4173" },
  webServer: { command: "npm run build && npx vite preview --port 4173 --strictPort", port: 4173, reuseExistingServer: true },
  projects: [
    {
      name: "chromium-fake-camera",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            ...(camera ? [`--use-file-for-fake-video-capture=${camera}`] : []),
            ...realGpu,
          ],
        },
      },
    },
  ],
});
