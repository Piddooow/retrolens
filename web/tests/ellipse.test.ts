import { describe, expect, it } from "vitest";
import { ellipseRows, lowResRadius } from "../src/ellipse";

// Expected rows are cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2r+1, 2r+1)) row sums // 2, from OpenCV.
describe("ellipse kernel", () => {
  it("matches OpenCV for the room's safe margin (radius 9)", () => {
    expect(ellipseRows(9)).toEqual([0, 4, 6, 7, 7, 8, 8, 9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 4, 0]);
  });

  it("matches OpenCV for the zone at half resolution (radius 5)", () => {
    expect(ellipseRows(5)).toEqual([0, 3, 4, 5, 5, 5, 5, 5, 4, 3, 0]);
  });

  it("grows the low resolution radius like dilate_to", () => {
    // bg_safe_margin_px 16 at display size, mask at half size: ceil(16 / 2) + 1.
    expect(lowResRadius(16, 2)).toBe(9);
  });
});
