import { expect, test } from "@playwright/test";

test("landing, camera, models, then the live stage", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => {
    // MediaPipe writes its start up log ("INFO: Created TensorFlow Lite XNNPACK delegate...") to stderr,
    // which the browser shows as a console error; it is not one.
    if (message.type() === "error" && !message.text().startsWith("INFO: ")) errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Retrolens" })).toBeVisible();
  await expect(page.getByText("Created by Piddooow @david.antonoo")).toBeVisible();
  await page.getByRole("button", { name: "Open the camera" }).click();

  // The models download and start, then the stage shows the camera with a live frame rate.
  await expect(page.locator("#stage")).toBeVisible({ timeout: 90_000 });
  await expect(page.locator("#status")).toContainText("fps", { timeout: 30_000 });
  await page.waitForTimeout(5000);
  const status = await page.locator("#status").textContent();
  console.log("status after 5 s:", status);

  // D cycles the debug views: the matte first, with what the models saw.
  await page.keyboard.press("d");
  await expect(page.locator("#view-label")).toContainText("Debug view: matte");
  await expect(page.locator("#view-label")).toContainText("models");
  await page.screenshot({ path: "test-results/stage-matte.png" });
  await page.getByRole("button", { name: "Debug view (D)" }).click();
  await expect(page.locator("#view-label")).toContainText("Debug view: zone");
  await page.screenshot({ path: "test-results/stage-zone.png" });

  await page.getByRole("button", { name: "Close the camera" }).click();
  await expect(page.locator("#landing")).toBeVisible();
  expect(errors).toEqual([]);
});
