import { expect, test } from "@playwright/test";

// The fake camera shows the empty room for 2.5 s, then a person (the clip loops).
test("learns the room by itself while it is empty, then is ready with the person in view", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().startsWith("INFO: ")) errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Open the camera" }).click();
  await expect(page.locator("#stage")).toBeVisible({ timeout: 90_000 });
  await expect(page.getByRole("heading", { name: "Step 1 of 3: learn the room" })).toBeVisible();
  // Empty and still for bg_auto_capture_s, then 20 frames averaged into the room.
  await expect(page.locator("#toast")).toHaveText("Background learned", { timeout: 30_000 });
  await expect(page.locator("#status")).toContainText("Ready", { timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Step 1 of 3: done" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Count down, then learn/ })).toBeHidden();
  // Coverage view: nothing left unseen in the room.
  await page.keyboard.press("d");
  await page.keyboard.press("d");
  await page.keyboard.press("d");
  await expect(page.locator("#view-label")).toContainText("Debug view: coverage");
  await page.screenshot({ path: "test-results/room-coverage.png" });
  expect(errors).toEqual([]);
});
