// OpenCV's MORPH_ELLIPSE structuring element (cv2.getStructuringElement), as one half-width per row, so a
// shader can dilate with exactly the kernel the Python app uses.

/** Half-width of each row (dy = -radius..radius) of the (2r+1) x (2r+1) ellipse. Row dy covers
 * dx = -halfWidth..halfWidth. Same arithmetic as OpenCV's getStructuringElement. */
export function ellipseRows(radius: number): number[] {
  const size = 2 * radius + 1;
  const r = Math.floor(size / 2);
  const c = Math.floor(size / 2);
  const invR2 = r ? 1 / (r * r) : 0;
  const rows: number[] = [];
  for (let i = 0; i < size; i++) {
    const dy = i - r;
    let j1 = 0;
    let j2 = 0;
    if (Math.abs(dy) <= r) {
      // OpenCV's saturate_cast<int> rounds half to even.
      const dx = roundHalfEven(c * Math.sqrt((r * r - dy * dy) * invR2));
      j1 = Math.max(c - dx, 0);
      j2 = Math.min(c + dx + 1, size);
    }
    rows.push((j2 - j1 - 1) / 2);
  }
  return rows;
}

function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Radius of a low resolution dilation that covers a display resolution radius, like snap/background.py
 * dilate_to: one extra low resolution pixel. */
export function lowResRadius(displayRadius: number, scale: number): number {
  return Math.ceil(displayRadius / scale) + 1;
}
