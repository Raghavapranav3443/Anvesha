"""Render the EO console headlessly and capture screenshots for UI review."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("runs/ui")
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto("http://localhost:8000", wait_until="networkidle")
    page.wait_for_timeout(1200)
    # dismiss onboarding if present (fresh browser profile)
    try:
        page.get_by_role("button", name="Skip").click(timeout=3000)
        page.wait_for_timeout(400)
    except Exception:
        pass
    page.screenshot(path=str(OUT / "console_initial.png"), full_page=True)

    # select two samples + run a change query end-to-end
    page.get_by_text("demo_change_2020.tif").click()
    page.get_by_text("demo_change_2024.tif").click()
    page.get_by_role("button", name="Run analysis").click()
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / "console_running.png"))

    page.wait_for_selector("text=Visual evidence", timeout=60000)
    page.wait_for_timeout(900)
    page.screenshot(path=str(OUT / "console_results.png"), full_page=True)

    # provenance view
    page.get_by_role("button", name="Provenance").click()
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / "provenance.png"), full_page=True)

    # history view
    page.get_by_role("button", name="History").click()
    page.wait_for_timeout(900)
    page.screenshot(path=str(OUT / "history.png"), full_page=True)

    # evaluation view
    page.get_by_role("button", name="Evaluation").click()
    page.wait_for_timeout(900)
    page.screenshot(path=str(OUT / "evaluation.png"), full_page=True)

    print("JS errors:", errors if errors else "none")
    browser.close()
