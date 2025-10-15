import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("junie-context")


@mcp.tool()
async def ask_junie(question: str) -> str:
    print("GOT QUESTION: " + question)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "http://127.0.0.1:8000/ask_junie",
                json={"question": question}
            )
            response.raise_for_status()
            data = response.json()
            print(f" Response from FastAPI: {data}")
            return data.get("context", "No context found")
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print("STARTING MCP SERVER")
    mcp.run()
