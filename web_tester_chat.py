import asyncio
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)

from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from utils import stringify_tool_call_results, stringify_tool_call_requests, format_assistant_responses

import json
import argparse
import os
import traceback

load_dotenv()


async def main(server_script_path: str):
    client = MCPClient()
    try:
        # await client.connect_to_python_server(server_script_path)
        await client.connect_to_server(
            command="npx", 
            args=["-y", "@playwright/mcp@latest", "--output-dir", "./"]
        )
        await client.chat_loop()
    finally:
        await client.cleanup()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()
        self.messages: list[ChatCompletionMessageParam] = []

    async def connect_to_server(self, command: str, args: list[str], env: dict = None):
        """Connect to an MCP server with custom command and arguments"""
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to server ({command} {' '.join(args)}) with tools:", [tool.name for tool in tools])

    async def connect_to_python_server(self, server_script_path: str):
        """Helper method to connect to a Python MCP server"""
        await self.connect_to_server("python", [server_script_path])

    async def connect_to_npx_server(self, package: str, additional_args: list[str] = None):
        """Helper method to connect to an NPX-based MCP server"""
        args = ["-y", package]
        if additional_args:
            args.extend(additional_args)
        await self.connect_to_server("npx", args)

    async def cleanup(self):
        await self.exit_stack.aclose()

    async def _available_tools(self) -> list[ChatCompletionToolParam]:
        response = await self.session.list_tools()
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema
                )
            )
            for tool in response.tools
        ]

    async def process_tool_call(self, tool_call) -> ChatCompletionToolMessageParam:
        assert tool_call['type'] == "function"

        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'] or "{}")

        max_try = 5
        error_message = ""
        for t in range(max_try):
            call_tool_result = await self.session.call_tool(tool_name, tool_args)
            if call_tool_result.isError:
                if t == max_try - 1:
                    error_message = f"[ERROR] Tool call failed: {call_tool_result}"
            else:
                break

        results = []
        if call_tool_result.isError:
            results.append(error_message)
        else:
            for result in call_tool_result.content:
                if result.type == "text":
                    results.append(result.text[:128000])
                else:
                    raise NotImplementedError(f"Unsupported result type: {result.type}")

        return ChatCompletionToolMessageParam(
            role="tool",
            content=json.dumps({
                **tool_args,
                tool_name: results
            }),
            tool_call_id=tool_call['id']
        )

    async def process_messages_streaming(self, messages: list[ChatCompletionMessageParam]):
        available_tools = await self._available_tools()

        stream = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            stream=True,
        )

        assistant_text_parts: list[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        print("\nAgent: ", end="", flush=True)

        for event in stream:
            choice = event.choices[0]
            delta = choice.delta

            if delta.content:
                print(delta.content, end="", flush=True)
                assistant_text_parts.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    slot = tool_calls_acc.setdefault(idx, {"id": None, "type": tc.type, "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        print("", flush=True)

        if finish_reason == "stop":
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="".join(assistant_text_parts)
                )
            )
            return messages

        if finish_reason == "tool_calls":
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                slot = tool_calls_acc[idx]
                assistant_tool_calls.append(
                    ChatCompletionMessageToolCallParam(
                        id=slot["id"] or f"tool_{idx}",
                        type=slot["type"] or "function",
                        function=Function(
                            name=slot["function"]["name"],
                            arguments=slot["function"]["arguments"]
                        )
                    )
                )

            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=assistant_tool_calls
                )
            )

            tasks = [asyncio.create_task(self.process_tool_call(tc)) for tc in assistant_tool_calls]
            tool_outputs = await asyncio.gather(*tasks)
            print(format_assistant_responses(tool_outputs))
            messages.extend(tool_outputs)

            return await self.process_messages_streaming(messages)

        if finish_reason == "length":
            raise ValueError("[ERROR] Length limit reached while streaming. Try a shorter query.")

        if finish_reason == "content_filter":
            raise ValueError("[ERROR] Content filter triggered while streaming.")

        raise ValueError(f"[ERROR] Unknown finish reason during streaming: {finish_reason}")

    async def chat_loop(self):
        """Run an interactive chat loop with streaming output"""
        print("Welcome to the MCP Client! Type 'exit' to quit.")
        self.messages = []

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break

            self.messages.append(ChatCompletionUserMessageParam(role="user", content=user_input))

            try:
                self.messages = await self.process_messages_streaming(self.messages)

            except Exception as e:
                print(f"Error processing user input: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client for connecting to a server.")
    parser.add_argument('server_script_path', help="Path to the server script (.py or .js)", type=str)
    args = parser.parse_args()

    asyncio.run(main(args.server_script_path))
