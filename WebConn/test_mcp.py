import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP Server")
            res = await session.call_tool("search_web", {"query": "Nvidia market cap May 8 2026"})
            print("Search result:", res)
            text_res = await session.call_tool("extract_text", {})
            print("Extracted text length:", len(str(text_res)))

asyncio.run(main())
