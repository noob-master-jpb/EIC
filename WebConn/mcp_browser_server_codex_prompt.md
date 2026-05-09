# Codex Prompt: Build a Python MCP Server with Headless Browser Access

## Goal
Build a production-ready **Model Context Protocol (MCP) server** that lets an LLM access the internet through a **headless browser** and retrieve page data reliably.

## Best-fit approach
Use **Playwright** as the core browser engine and build an **MCP server** around it.

Why this is the best fit:
- Handles JavaScript-heavy websites
- Works well headlessly
- Supports navigation, clicking, typing, screenshots, and DOM extraction
- Can be extended with safe, structured tools for retrieval

Use a **Fetch fallback** only for simple static pages or raw content retrieval.

## What to build
Create an MCP server in **Python** with these tools:
- `open_url(url)`
- `click(selector)`
- `type(selector, text)`
- `press(key)`
- `wait(ms)`
- `extract_text(selector?)`
- `get_page_html()`
- `screenshot()`
- `search_web(query)` if needed through a browser-based search flow

## Required behavior
- Run browser **headlessly by default**
- Support one persistent browser context per session
- Clean up sessions properly
- Return structured JSON outputs
- Be resilient to timeouts, redirects, popups, and dynamic content
- Do not expose local files or internal network resources
- Restrict outbound access with an allowlist if possible

## Security requirements
- Block access to `localhost`, private IP ranges, and internal domains unless explicitly enabled
- Add URL validation and protocol checks
- Add timeouts for all browser actions
- Sanitize extracted content
- Log tool calls and failures clearly

## Architecture
Implement this structure:
- `server.py` — MCP server entry
- `browser.py` — Playwright browser/session management
- `tools/*.py` — individual MCP tool handlers
- `utils/*.py` — validation, parsing, logging
- `requirements.txt`
- `README.md`

## Implementation guidance
1. Initialize an MCP server using the official Python MCP SDK.
2. Use Playwright for Python.
3. Launch Chromium in headless mode.
4. Maintain browser state per client session.
5. Expose each browser action as a separate MCP tool.
6. Add retries, timeout handling, and graceful cleanup.
7. Return concise structured JSON responses for the LLM.
8. Include setup and usage examples in the README.

## Tool design rules
Each tool should:
- Accept validated input only
- Fail safely with clear error messages
- Return machine-readable output
- Avoid leaking unnecessary page noise

Suggested tool output format:
```json
{
  "success": true,
  "url": "https://example.com",
  "title": "Example",
  "content": "...",
  "metadata": {
    "status": 200,
    "timestamp": "..."
  }
}
```

## Acceptance criteria
The build is complete when:
- The server runs locally
- The LLM can open a URL and extract content
- The browser can click and type on interactive pages
- The server works headlessly
- The output is stable and structured
- Basic security restrictions are in place

## Optional upgrades
- Add a `crawl_page` tool for recursive discovery
- Add OCR for screenshots
- Add support for authenticated sessions
- Add a sitemap-based fetcher
- Add caching for repeated retrieval

## Dependencies
Use:
- `playwright`
- `mcp`
- `pydantic`
- `asyncio`
- `uvicorn` only if HTTP transport is added

## Final instruction to Codex
Build this as a clean, modular Python MCP server using Playwright for Python. Use async patterns where appropriate. Prefer reliability and safety over feature count. Keep the tool surface small, stable, and easy to extend.

