import { describe, expect, it } from "vitest";
import config from "../src/data/config.json";
import glitch from "../src/data/glitch.json";

// The JSON is written by tools/export_web_data.py from the Python app; these checks catch a broken export.
describe("shared data", () => {
  it("has the settings the web app reads", () => {
    for (const key of ["kamera_lebar", "kamera_tinggi", "mask_ema_baru", "glitch_step_hz", "hold_metal_s"]) {
      expect(config.config).toHaveProperty(key);
    }
    expect(config.styles).toEqual(["DUST", "ASH", "EMBER", "GLITCH"]);
  });

  it("has the three glitch scores with known palette colors", () => {
    expect(glitch.score).toHaveLength(94);
    expect(glitch.vanishScore).toHaveLength(10);
    expect(glitch.reappearScore).toHaveLength(10);
    const palette = glitch.palette as Record<string, string>;
    for (const hex of Object.values(palette)) expect(hex).toMatch(/^#[0-9A-F]{6}$/);
    for (const step of [...glitch.score, ...glitch.vanishScore, ...glitch.reappearScore] as Record<string, unknown>[]) {
      for (const name of (step.fill as string[] | undefined) ?? []) expect(palette).toHaveProperty(name);
    }
  });
});
