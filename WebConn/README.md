# Web Search Service

This service allows a any LLM like - (`qwen3.5-0.8b-local`) to answer user questions by dynamically searching the web and reading real-time context. It uses a primary headless browser approach (via MCP) and a robust BeautifulSoup fallback to ensure uninterrupted access even when search engines enforce rate limits or CAPTCHAs.

## How It Works

The service operates in a 3-step pipeline:
1. **Query Generation:** The user's question is passed to the LLM model using a few-shot prompt to generate a concise, optimized web search query (e.g., "nvidia market cap may 8 2026").
2. **Web Search & Extraction:** 
   - **Primary:** The service queries the local MCP Browser Server (`server.py`), which uses Playwright to open a headless browser, search Yahoo, and extract the page text.
   - **Fallback:** If the browser encounters a CAPTCHA or returns suspiciously short text, the service automatically falls back to `BeautifulSoup` to scrape search snippets directly via HTTP requests.
3. **Answer Generation:** The extracted text context is fed back into the local Qwen model alongside the original question. LLM then synthesizes a final, grounded answer based on the real-time web data.

## File & Folder Overview

* **`run_websrch.py`**: The parent entry point script. It handles command-line arguments and runs the core service as a subprocess.
* **`websrch.py`**: The core service script. It loads the LLM model, orchestrates the 3-step pipeline, communicates with the MCP server, and implements the BeautifulSoup fallback logic.
* **`server.py`**: The MCP Browser Server entry point. It registers browser tools (like `search_web` and `extract_text`) and manages the Playwright browser lifecycle.
* **`browser.py`**: Handles Playwright interactions (clicking, waiting, extracting) headlessly.
* **`tools/browser_tools.py`**: Tool-level wrappers for the browser actions, including the specific `search_web` orchestration.
* **`utils/`**: Contains helper modules for safety validations (`validation.py`), text sanitization (`sanitization.py`), and standard error logging (`logging_utils.py`).

## Usage

Run the parent script with your question. You can optionally specify the path to your LLM model.

```bash
# Ensure you are in the virtual environment
source ../.venv/bin/activate

# Run the search
python run_websrch.py --question "what is the latest market cap of nvidia in may 8 2026"
```
