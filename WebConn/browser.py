from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Error, Page, TimeoutError, async_playwright

from utils.logging_utils import setup_logger
from utils.sanitization import sanitize_text


@dataclass
class BrowserConfig:
    headless: bool = True
    nav_timeout_ms: int = 20000
    action_timeout_ms: int = 10000
    retries: int = 2


class BrowserSession:
    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._lock = asyncio.Lock()
        self.log = setup_logger("mcp_browser_server.session")

    async def start(self) -> None:
        if self._page:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.config.headless)
        self._context = await self._browser.new_context(ignore_https_errors=False)
        self._page = await self._context.new_page()
        self._page.set_default_navigation_timeout(self.config.nav_timeout_ms)
        self._page.set_default_timeout(self.config.action_timeout_ms)

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser session not initialized")
        return self._page

    async def _run(self, action_name: str, fn) -> dict[str, Any]:
        async with self._lock:
            await self.start()
            last_exc: Exception | None = None
            for attempt in range(self.config.retries + 1):
                try:
                    result = await fn()
                    return self._ok(result)
                except (TimeoutError, Error, RuntimeError) as exc:
                    last_exc = exc
                    self.log.warning("%s failed (attempt %s): %s", action_name, attempt + 1, exc)
                    if attempt >= self.config.retries:
                        break
                    await asyncio.sleep(0.35 * (attempt + 1))
            return self._err(action_name, str(last_exc) if last_exc else "unknown error")

    def _ok(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": True,
            **payload,
            "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }

    def _err(self, action: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "metadata": {"action": action, "timestamp": datetime.now(timezone.utc).isoformat()},
        }

    async def open_url(self, url: str) -> dict[str, Any]:
        async def _go():
            response = await self.page.goto(url, wait_until="domcontentloaded")
            await self.page.wait_for_load_state("networkidle", timeout=self.config.nav_timeout_ms)
            return {
                "url": self.page.url,
                "title": sanitize_text(await self.page.title(), max_len=500),
                "status": response.status if response else None,
            }

        return await self._run("open_url", _go)

    async def click(self, selector: str) -> dict[str, Any]:
        async def _click():
            await self.page.locator(selector).first.click(timeout=self.config.action_timeout_ms)
            return {"url": self.page.url}

        return await self._run("click", _click)

    async def type(self, selector: str, text: str) -> dict[str, Any]:
        async def _type():
            loc = self.page.locator(selector).first
            await loc.click(timeout=self.config.action_timeout_ms)
            await loc.fill(text, timeout=self.config.action_timeout_ms)
            return {"url": self.page.url}

        return await self._run("type", _type)

    async def press(self, key: str) -> dict[str, Any]:
        async def _press():
            await self.page.keyboard.press(key, timeout=self.config.action_timeout_ms)
            return {"url": self.page.url}

        return await self._run("press", _press)

    async def wait(self, ms: int) -> dict[str, Any]:
        async def _wait():
            await self.page.wait_for_timeout(ms)
            return {"url": self.page.url}

        return await self._run("wait", _wait)

    async def extract_text(self, selector: str | None = None) -> dict[str, Any]:
        async def _extract():
            if selector:
                text = await self.page.locator(selector).first.inner_text(timeout=self.config.action_timeout_ms)
            else:
                text = await self.page.inner_text("body", timeout=self.config.action_timeout_ms)
            return {"url": self.page.url, "content": sanitize_text(text)}

        return await self._run("extract_text", _extract)

    async def get_page_html(self) -> dict[str, Any]:
        async def _html():
            html = await self.page.content()
            return {"url": self.page.url, "content": sanitize_text(html, max_len=120000)}

        return await self._run("get_page_html", _html)

    async def screenshot(self) -> dict[str, Any]:
        async def _screenshot():
            png = await self.page.screenshot(full_page=True, type="png")
            import base64

            encoded = base64.b64encode(png).decode("ascii")
            return {"url": self.page.url, "image_base64": encoded}

        return await self._run("screenshot", _screenshot)
