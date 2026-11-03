import { expect, test } from "@playwright/test";

test("a blocked camera explains what to do, and Try again asks again", async ({ page }) => {
  await page.addInitScript(() => {
    let calls = 0;
    navigator.mediaDevices.getUserMedia = async () => {
      calls += 1;
      (window as unknown as { cameraCalls: number }).cameraCalls = calls;
      throw new DOMException("Permission denied", "NotAllowedError");
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open the camera" }).click();
  await expect(page.getByRole("heading", { name: "The camera is blocked for this page" })).toBeVisible();
  await expect(page.getByText("Choose Always allow")).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeFocused();
  await page.screenshot({ path: "test-results/camera-blocked.png", fullPage: true });
  await page.keyboard.press("Enter");
  await expect.poll(() => page.evaluate(() => (window as unknown as { cameraCalls: number }).cameraCalls)).toBe(2);
  await page.getByRole("button", { name: "Back" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Retrolens" })).toBeVisible();
});

test("the landing page fits a phone and a laptop, and works by keyboard", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  await page.screenshot({ path: "test-results/landing-laptop.png" });
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Open the camera" })).toBeFocused();
  await page.setViewportSize({ width: 375, height: 740 });
  await page.screenshot({ path: "test-results/landing-phone.png", fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
