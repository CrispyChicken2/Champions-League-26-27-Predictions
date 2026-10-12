"""Save raw pages for a Matchday Snapshot.

Usage: uv run scripts/capture_raw.py <out_dir> <name>=<url> [<name>=<url> ...]

Each page is saved as <name>.html (rendered DOM) and <name>.txt (visible text),
with a manifest.json recording the URL, HTTP status and capture time (UTC).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
PAUSE_SECONDS = 4


def capture(out_dir: Path, targets: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=USER_AGENT, locale="en-GB", viewport={"width": 1400, "height": 2000}
        )
        page = context.new_page()
        for name, url in targets.items():
            entry = {"url": url, "captured_at_utc": datetime.now(timezone.utc).isoformat()}
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
                try:
                    page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass
                page.mouse.wheel(0, 6000)
                time.sleep(PAUSE_SECONDS)
                html = page.content()
                text = page.inner_text("body")
                (out_dir / f"{name}.html").write_text(html, "utf-8")
                (out_dir / f"{name}.txt").write_text(text, "utf-8")
                entry.update(
                    status=response.status if response else None,
                    title=page.title(),
                    html_bytes=len(html),
                    text_chars=len(text),
                )
            except Exception as exc:
                entry["error"] = repr(exc)
            manifest[name] = entry
            print(name, json.dumps(entry, ensure_ascii=False))
        browser.close()

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    pairs = dict(arg.split("=", 1) for arg in sys.argv[2:])
    capture(out, pairs)
