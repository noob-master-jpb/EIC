# Python MCP Browser Server (Playwright)

This server lets an LLM browse real web pages through a headless browser and return stable JSON outputs.

## Why this works
- Uses **Playwright Chromium** (`browser.py`) so JS-heavy sites render correctly.
- Keeps one **persistent browser session** per server process for consistent state.
- Wraps every action with **timeouts + retries** (`browser.py`) for reliability.
- Blocks unsafe targets by default (`utils/validation.py`) so localhost/private networks are not exposed.
- Returns machine-friendly JSON from each tool (`server.py` + `tools/browser_tools.py`).

## How it works (short flow)
1. MCP starts in `server.py` and registers tools (`open_url`, `click`, `type`, etc.).
2. Each tool calls `tools/browser_tools.py` for input checks and orchestration.
3. Browser actions execute in `browser.py` (open/click/type/extract/screenshot).
4. Validation/safety happens in `utils/validation.py`.
5. Output cleanup happens in `utils/sanitization.py`.
6. Tool call logs are emitted by `utils/logging_utils.py`.

## File / folder reference
- `server.py`: MCP entrypoint and tool registration.
- `browser.py`: Playwright lifecycle + action execution + retry/timeout handling.
- `tools/browser_tools.py`: tool-level wrappers and `search_web` flow.
- `utils/validation.py`: URL/protocol/host safety checks.
- `utils/sanitization.py`: text/HTML sanitization and truncation.
- `utils/logging_utils.py`: log format and logger setup.
- `requirements.txt`: Python dependencies.

## Quick run
```bash
pip install -r requirements.txt
playwright install chromium
python server.py
```

## Defaults
- `MCP_BROWSER_HEADLESS=true`
- `MCP_BROWSER_ALLOW_INTERNAL=false`
