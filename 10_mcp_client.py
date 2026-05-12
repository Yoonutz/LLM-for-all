"""
What it does: The client connects to the MCP server, loads tools, and launches a ReAct agent.
"""

import os
import asyncio
from contextlib import AsyncExitStack

from dotenv import load_dotenv

load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import create_model, Field

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def main():
    """Entry point for running the MCP client.

    Connects to the server, requests available tools, converts them to LangChain format,
    and starts a dialogue loop with the user via a ReAct agent.
    """
    stack = AsyncExitStack()

    # Connect to the MCP server via stdio
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "09_mcp_server.py"],
    )

    stdio_transport = await stack.enter_async_context(stdio_client(server_params))
    read_stream, write_stream = stdio_transport
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    # Retrieve the list of tools from the server
    mcp_tools = await session.list_tools()
    print(f"MCP Tools found: {len(mcp_tools.tools)}")
    for t in mcp_tools.tools:
        print(f"  - {t.name}: {t.description}")

    # Convert MCP tools to LangChain format
    langchain_tools = []
    for t in mcp_tools.tools:
        async def make_tool_fn(tool_name: str):
            """Creates an asynchronous wrapper for calling an MCP tool.

            Args:
                tool_name (str): Name of the tool provided by the MCP server.

            Returns:
                Callable: Function to invoke the corresponding tool.
            """
            async def call_tool(**kwargs) -> str:
                """Invokes the tool call on the MCP server side.

                Args:
                    **kwargs: Arguments passed to the tool.

                Returns:
                    str: Textual result of the tool execution.
                """
                result = await session.call_tool(tool_name, arguments=kwargs)
                from mcp.types import TextContent
                first = result.content[0] if result.content else None
                return first.text if isinstance(first, TextContent) else "No result"
            return call_tool

        fn = await make_tool_fn(t.name)

        properties = t.inputSchema.get("properties", {})
        fields: dict = {}
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            type_map = {"string": str, "number": float, "integer": int, "boolean": bool}
            fields[prop_name] = (type_map.get(prop_type, str), Field(description=prop_schema.get("description", "")))

        args_schema = create_model(f"{t.name}Schema", **fields)

        langchain_tools.append(StructuredTool.from_function(
            coroutine=fn,
            name=t.name,
            description=t.description or "",
            args_schema=args_schema,
        ))

    # ReAct Agent
    llm_with_tools = llm.bind_tools(langchain_tools)
    tools_map = {t.name: t for t in langchain_tools}

    print("\nMCP Agent is ready. Enter your question (or 'exit' to quit)\n")

    dialog: list = [
        {"role": "system", "content": "You are an AI assistant with MCP tools. Use them for accurate answers."},
    ]

    while True:
        user_input = input("❓ Question: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        dialog.append(HumanMessage(content=user_input))

        for _ in range(5):
            response = await llm_with_tools.ainvoke(dialog)
            dialog.append(response)

            if not response.tool_calls:
                print(f"\n🤖 {response.content}\n")
                break

            for tc in response.tool_calls:
                print(f"  🔧 MCP: {tc['name']}({tc['args']})")
                tool_fn = tools_map[tc["name"]]
                result = await tool_fn.ainvoke(tc["args"])
                print(f"  📋 → {result}")
                dialog.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    await stack.aclose()


asyncio.run(main())
