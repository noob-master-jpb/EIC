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
            res = await session.call_tool("search_web", {"query": "Nvidia market cap"})
            # print("search result", res)
            await asyncio.sleep(2) # duckduckgo might need some time or click? 
            text_res = await session.call_tool("extract_text", {})
            print("Extracted text:")
            print(text_res.content[0].text[:500])

asyncio.run(main())
