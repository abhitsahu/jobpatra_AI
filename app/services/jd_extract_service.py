"""JD URL Extraction Service — 2-tier scraping pipeline.

Tier 1: httpx (async HTTP) + trafilatura (boilerplate removal)
         Handles static HTML job pages — fast, zero browser overhead.

Tier 2: playwright (headless Chromium)
         Handles JS-rendered pages (LinkedIn, Indeed, Naukri, etc.)
         Falls back automatically when Tier 1 yields < MIN_TEXT_CHARS.

No ATS logic lives here.  This module only retrieves and cleans text.
"""

from __future__ import annotations

import asyncio

from app.core.logging import logger
from app.core.errors import AppError
from app.schemas.jd_extract import JDExtractResponse

# Minimum extracted characters to consider Tier 1 a success
_MIN_TEXT_CHARS = 150

# Timeout constants (seconds)
_HTTPX_TIMEOUT = 12
_PLAYWRIGHT_NAV_TIMEOUT_MS = 25_000   # 25 s — page.goto timeout

# Browser user-agent mimic
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


async def extract_from_url(url: str) -> JDExtractResponse:
    """Extract the main job description text from a public URL.

    Tries Tier 1 (httpx + trafilatura) first.  If that yields too little
    text, falls back to Tier 2 (playwright headless Chromium).

    Raises:
        AppError: If both tiers fail to produce usable text.
    """
    url_str = str(url)
    logger.info("[JDExtract] Starting extraction for %s", url_str)

    # ── Tier 1: httpx + trafilatura ────────────────────────────────────────
    try:
        text = await _tier1_httpx(url_str)
        if text and len(text) >= _MIN_TEXT_CHARS:
            logger.info("[JDExtract] Tier 1 success — %d chars from %s", len(text), url_str)
            return JDExtractResponse(
                text=text,
                source="httpx",
                char_count=len(text),
                url=url_str,
            )
        logger.info("[JDExtract] Tier 1 insufficient (%d chars) — trying Tier 2", len(text) if text else 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[JDExtract] Tier 1 failed: %s — falling back to Tier 2", exc)

    # ── Tier 2: playwright headless browser ────────────────────────────────
    try:
        text = await _tier2_playwright(url_str)
        if text and len(text) >= _MIN_TEXT_CHARS:
            logger.info("[JDExtract] Tier 2 success — %d chars from %s", len(text), url_str)
            return JDExtractResponse(
                text=text,
                source="playwright",
                char_count=len(text),
                url=url_str,
            )
        logger.warning("[JDExtract] Tier 2 also returned insufficient text (%d chars)", len(text) if text else 0)
    except Exception as exc:  # noqa: BLE001
        logger.error("[JDExtract] Tier 2 failed: %s", exc)

    # Both tiers failed
    raise AppError(
        message=(
            "Could not extract job description from this URL. "
            "The site may use bot protection or require login. "
            "Please copy and paste the job description manually."
        ),
        code="JD_EXTRACTION_FAILED",
        status_code=422,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — httpx + trafilatura
# ─────────────────────────────────────────────────────────────────────────────

async def _tier1_httpx(url: str) -> str | None:
    """Fetch the page with httpx and extract main content via trafilatura."""
    try:
        import httpx
        import trafilatura
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency for Tier 1: {exc}") from exc

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    async with httpx.AsyncClient(
        timeout=_HTTPX_TIMEOUT,
        headers=headers,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        html = response.text

    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,  # prioritize completeness over precision
        no_fallback=False,
    )
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — playwright headless browser
# ─────────────────────────────────────────────────────────────────────────────

async def _tier2_playwright(url: str) -> str | None:
    """Launch a headless Chromium browser, render the page, extract text."""
    try:
        from playwright.async_api import async_playwright
        import trafilatura
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency for Tier 2: {exc}") from exc

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
        )

        try:
            page = await context.new_page()

            # Hide webdriver property to reduce bot detection
            await page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            await page.goto(
                url,
                wait_until="networkidle",
                timeout=_PLAYWRIGHT_NAV_TIMEOUT_MS,
            )

            # Small wait to let lazy-loaded content settle
            await asyncio.sleep(1.5)

            html = await page.content()
        finally:
            await context.close()
            await browser.close()

    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,
        no_fallback=False,
    )
    return extracted
