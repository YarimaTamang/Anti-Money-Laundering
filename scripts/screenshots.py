"""Capture dashboard screenshots with Playwright + locally installed Chrome.

Start the dashboard first:  streamlit run dashboard/app.py --server.port 8501
Then:                        python scripts/screenshots.py [--url http://localhost:8501]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGES = [("", "01_overview"), ("Transaction_Investigation", "02_transaction_investigation"),
         ("Network_Explorer", "03_network_explorer"), ("Explainable_AI", "04_explainable_ai"),
         ("Model_Laboratory", "05_model_laboratory"), ("Investigation_Reports", "06_investigation_reports")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--wait", type=int, default=25000)
    a = ap.parse_args()
    out = ROOT / "reports" / "dashboard_screenshots"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1500, "height": 2300}, device_scale_factor=1)
        for path, name in PAGES:
            pg.goto(f"{a.url}/{path}", wait_until="networkidle", timeout=180000)
            pg.wait_for_selector("[data-testid='stAppViewContainer']", timeout=120000)
            # wait until Streamlit has finished running the script
            pg.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=240000)
            if name == "06_investigation_reports":
                btn = pg.get_by_role("button", name="Generate report")
                if btn.count():
                    btn.first.click()
                    pg.get_by_text("TRACE-X investigation report").first.wait_for(timeout=600000)
            pg.wait_for_timeout(a.wait)
            pg.screenshot(path=str(out / f"{name}.png"), full_page=False)
            print("saved", name, flush=True)
        b.close()


if __name__ == "__main__":
    main()
