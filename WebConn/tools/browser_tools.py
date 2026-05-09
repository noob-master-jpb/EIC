from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from browser import BrowserSession
from utils.validation import SearchInput, validate_public_url


class BrowserTools:
    def __init__(self, session: BrowserSession, allow_internal: bool = False):
        self.session = session
        self.allow_internal = allow_internal

    async def open_url(self, url: str) -> dict[str, Any]:
        validated = validate_public_url(url, allow_internal=self.allow_internal)
        return await self.session.open_url(validated.url)

    async def click(self, selector: str) -> dict[str, Any]:
        return await self.session.click(selector)

    async def type(self, selector: str, text: str) -> dict[str, Any]:
        return await self.session.type(selector, text)

    async def press(self, key: str) -> dict[str, Any]:
        return await self.session.press(key)

    async def wait(self, ms: int) -> dict[str, Any]:
        ms = max(0, min(ms, 30000))
        return await self.session.wait(ms)

    async def extract_text(self, selector: str | None = None) -> dict[str, Any]:
        return await self.session.extract_text(selector)

    async def get_page_html(self) -> dict[str, Any]:
        return await self.session.get_page_html()

    async def screenshot(self) -> dict[str, Any]:
        return await self.session.screenshot()

    async def search_web(self, query: str) -> dict[str, Any]:
        query_model = SearchInput(query=query)
        url = f"https://search.yahoo.com/search?p={quote_plus(query_model.query)}"
        return await self.open_url(url)
