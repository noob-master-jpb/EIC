from __future__ import annotations

import atexit
import asyncio
import os

from mcp.server.fastmcp import FastMCP

from browser import BrowserConfig, BrowserSession
from tools.browser_tools import BrowserTools
from utils.logging_utils import setup_logger

logger = setup_logger("mcp_browser_server")

HEADLESS = os.getenv("MCP_BROWSER_HEADLESS", "true").lower() != "false"
ALLOW_INTERNAL = os.getenv("MCP_BROWSER_ALLOW_INTERNAL", "false").lower() == "true"

session = BrowserSession(config=BrowserConfig(headless=HEADLESS))
tools = BrowserTools(session=session, allow_internal=ALLOW_INTERNAL)
mcp = FastMCP(name="browser-server")


@mcp.tool()
async def open_url(url: str):
    """Open URL in the persistent browser page."""
    logger.info("tool=open_url url=%s", url)
    return await tools.open_url(url)


@mcp.tool()
async def click(selector: str):
    """Click first element matching selector."""
    logger.info("tool=click selector=%s", selector)
    return await tools.click(selector)


@mcp.tool()
async def type(selector: str, text: str):
    """Fill text into first element matching selector."""
    logger.info("tool=type selector=%s", selector)
    return await tools.type(selector, text)


@mcp.tool()
async def press(key: str):
    """Send keyboard key press."""
    logger.info("tool=press key=%s", key)
    return await tools.press(key)


@mcp.tool()
async def wait(ms: int):
    """Wait for specified milliseconds."""
    logger.info("tool=wait ms=%s", ms)
    return await tools.wait(ms)


@mcp.tool()
async def extract_text(selector: str | None = None):
    """Extract text from page body or selector."""
    logger.info("tool=extract_text selector=%s", selector)
    return await tools.extract_text(selector)


@mcp.tool()
async def get_page_html():
    """Get current page HTML content."""
    logger.info("tool=get_page_html")
    return await tools.get_page_html()


@mcp.tool()
async def screenshot():
    """Capture full page screenshot and return base64 png."""
    logger.info("tool=screenshot")
    return await tools.screenshot()


@mcp.tool()
async def search_web(query: str):
    """Open web search results using browser flow."""
    logger.info("tool=search_web query=%s", query)
    return await tools.search_web(query)


def _shutdown() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return

    if loop.is_running():
        loop.create_task(session.close())
    else:
        loop.run_until_complete(session.close())


atexit.register(_shutdown)


if __name__ == "__main__":
    mcp.run()
