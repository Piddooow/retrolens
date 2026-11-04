import { expect, test } from "@playwright/test";

// The fake camera always shows a person, so the room is never empty.
test("B counts down, then refuses a room with someone in it", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open the camera" }).click();
  await expect(page.locator("#stage")).toBeVisible({ timeout: 90_000 });
  await expect(page.locator("#status")).toContainText("Learning", { timeout: 30_000 });
  await page.keyboard.press("b");
  await expect(page.locator("#center")).toHaveText(/Step out of the frame [1-3]/);
  await expect(page.getByRole("button", { name: /Count down, then learn/ })).toBeDisabled();
  await expect(page.locator("#toast")).toHaveText("Someone is still in the frame. Step out and press B again",
                                                    { timeout: 30_000 });
  await expect(page.locator("#center")).toBeHidden();
  await expect(page.getByRole("button", { name: /Count down, then learn/ })).toBeEnabled();
});
