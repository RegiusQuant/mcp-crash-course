import asyncio
import json
from typing import Any, Dict, List

import litellm
from mcp import ClientSession
from mcp.client.sse import sse_client


session: ClientSession = None
model = "gpt-4o"


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Retrieve the MCP tools in Litellm/OpenAI function format."""
        
    global session
    
    tools_result = await session.list_tools()

    print("Connected to server with tools:")
    for tool in tools_result.tools:
        print(f"  • {tool.name}: {tool.description}")

    formatted = []
    for tool in tools_result.tools:
        formatted.append({"type": "function",
                          "function": {"name": tool.name,
                                       "description": tool.description,
                                       "parameters": tool.inputSchema}
                        })
    return formatted


async def process_query(query: str) -> str:

    global session, model
    tools = await get_mcp_tools()

    # First pass: let Litellm decide if it needs to call a tool
    first_response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )
    # Extract the assistant message
    assistant_message = first_response.choices[0].message
    messages: List[Dict[str, Any]] = [{"role": "user", "content": query}, assistant_message]

    # Check if Litellm wants to call any tools
    calls = assistant_message.tool_calls or assistant_message.function_call
    if calls:
        calls_list = calls if isinstance(calls, list) else [calls]

        for call in calls_list:
            if call.function:  # newer Litellm structure
                tool_name = call.function.name
                raw_args = call.function.arguments
            else:
                tool_name = call.name
                raw_args = call.arguments

            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = raw_args

            print(f"\nAssistant requests tool: {tool_name}({parsed_args})")
            permission = input("Allow execution? (y/n): ").strip().lower()

            if permission == "y":
                result = await session.call_tool(tool_name, arguments=parsed_args)
                tool_output = result.content[0].text
                print(f"→ {tool_name} returned: {tool_output}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id if hasattr(call, "id") else None,
                    "content": tool_output,
                })
            else:
                denied_msg = f"[Tool '{tool_name}' denied]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id if hasattr(call, "id") else None,
                    "content": denied_msg,
                })
                print(f"→ Skipped {tool_name}")

        # Second pass: ask Litellm for a reply now that tool output is available
        second_response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="none",
        )

        return second_response.choices[0].message.content.strip()
    else:
        return assistant_message.content.strip() # Litellm answered without needing any tools


async def main():
    # Replace this URL with wherever your MCP server is exposing SSE
    server_sse_url = "http://127.0.0.1:8000/sse"

    # 1) Open an SSE connection, which yields (read_stream, write_stream).
    # 2) Wrap those in a ClientSession so we can talk to MCP.
    async with sse_client(server_sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            global session
            session = mcp_session

            # Initialize the MCP connection
            await session.initialize()

            # Example query
            query = "What is 90 multiplied by 68.6? Use MCP tools if available."
            print(f"\nQuery: {query}")
            response = await process_query(query)
            print(f"\nFinal Response: {response}")

    print("Connection terminated successfully!")

if __name__ == "__main__":
    asyncio.run(main())