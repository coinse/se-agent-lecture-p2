import asyncio
from datetime import datetime
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

from utils import stringify_tool_call_results, stringify_tool_call_requests, format_assistant_responses, DualPrinter
from actor_utils import *

import json
import argparse
import os
import traceback
import chromadb

load_dotenv()

chroma_client = chromadb.Client()
result_path = 'agent_runs/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(result_path, exist_ok=True)
os.makedirs(os.path.join(result_path, 'memory'), exist_ok=True)
printer = DualPrinter(file_path=os.path.join(result_path, "output.log"))

async def main(target_website: str):
    client = MCPClient()
    try:
        await client.connect_to_server(
            command="npx", 
            args=["-y", "@playwright/mcp@latest", "--output-dir", "./"]
        )
        await client.testing_loop(target_website)
    finally:
        await client.cleanup()


class Actor:
    def __init__(self, target_website_url: str):
        self.target_website_url = target_website_url
        self.memory = chroma_client.create_collection(name="actor_memory")
        self.id_counter = 0

    def add_memory_entries(self, entries):
        ids = [str(self.id_counter + i) for i in range(len(entries))]
        self.id_counter += len(entries)
        self.memory.add(
            ids=ids,
            documents=entries
        )

    def retrieve_relevant_memory(self, query: str) -> list[str]:
        """Retrieve relevant memory entries based on a query"""
        results = self.memory.query(query_texts=[query], n_results=5)

        # semantically most similar memory entries to the current query
        return '\n'.join(results['documents'][0])

    def dump_memory(self):
        return json.dumps(self.memory.get())

    async def reflect_on_previous_attempt(self, messages, task, mcp_client):
        reflection_request = f'''The following are the trajectory of your attempt to perform the following task: 
{task}

on the website url {self.target_website_url}.

Message trajectory:
{str(messages)}

Generate brief one-line reflections and takeaways that you can refer to in future attempts. Your answer should only contain one reflection per a line, and no other text.'''
        messages = [{
            "role": "user",
            "content": reflection_request
        }]
        messages = await mcp_client.process_messages_streaming(messages) # TODO: disable tool calls
        response = messages[-1]['content']
        reflections = response.splitlines()

        self.add_memory_entries(reflections)


    async def attempt_perform_task(self, task_instruction, mcp_client, max_tries=3):
        for t in range(max_tries):
            messages = [{
                "role": "system",
                "content": f"You are a QA testing agent that can interact with web pages under the url {self.target_website_url} and perform an assigned task (testing scenario). Focus on the functionality testing, and try best to accomplish the task unless the task itself is turned to be impossible to perform on the target website."
            }]
            first_user_message = f'''Perform the following task:
{task_instruction}

You must perform the task inside the target webpage: {self.target_website_url}.
If you accidentally navigate away from the page, try to return to it.

Before and after calling any tools, briefly describe the context and your intentions.

After completing the task, please provide a brief testing report about the actions you took, report any issues you encountered.

If you were unable to complete the task but believe it is still possible, end your answer with the exact mark: <<TASK_INCOMPLETE>>
If the task instruction failed likely due to the webpage's possible bugs or issues, mark: <<TASK_FAILED>>
If successful, mark: <<TASK_SUCCESS>>'''
            if self.memory.count() > 0: # Add relevant memory
                retrieved_memory_entries = self.retrieve_relevant_memory(task_instruction)
                first_user_message += f'''
These are some takeaways from your previous attempts to accomplish the similar task:
{retrieved_memory_entries}'''

                printer.print(f"* * * [INFO] Relevant memory entries retrieved:")
                printer.print(retrieved_memory_entries)

            messages.append(ChatCompletionUserMessageParam(role="user", content=first_user_message))

            messages = await mcp_client.process_messages_streaming(messages)

            # Update memory
            _ = await self.reflect_on_previous_attempt(messages, task_instruction, mcp_client)
            with open(os.path.join(result_path, "memory", f"{self.id_counter}.json"), "w", encoding="utf-8") as f:
                f.write(self.dump_memory())

            response = messages[-1]['content']
            if '<<TASK_SUCCESS>>' in response:
                return "Success"

            if t == max_tries - 1:
                printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                if '<<TASK_FAILED>>' in response:
                    return "Failed"
                else:
                    return "CantPerform"


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()

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
        printer.print(f"\nConnected to server ({command} {' '.join(args)}) with tools:", [tool.name for tool in tools])

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
                    results.append(result.text[:256000])
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

        printer.print("\nAgent: ", end="", flush=True)

        for event in stream:
            choice = event.choices[0]
            delta = choice.delta

            if delta.content:
                printer.print(delta.content, end="", flush=True)
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

        printer.print("", flush=True)

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
            printer.print(format_assistant_responses(tool_outputs))
            messages.extend(tool_outputs)

            return await self.process_messages_streaming(messages)

        if finish_reason == "length":
            raise ValueError("[ERROR] Length limit reached while streaming. Try a shorter query.")

        if finish_reason == "content_filter":
            raise ValueError("[ERROR] Content filter triggered while streaming.")

        raise ValueError(f"[ERROR] Unknown finish reason during streaming: {finish_reason}")

    async def testing_loop(self, target_website):
        """Run an interactive (or autonomous) testing loop"""
        printer.print(f"Testing Agent for the websites: {target_website}")

        self.messages: list[ChatCompletionMessageParam] = []

        await self.process_tool_call({
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "arguments": json.dumps({
                    "url": target_website
                })
            },
            "id": "initial_navigation"
        })

        actor = Actor(target_website_url=target_website)
        while True:
            await self.process_tool_call({
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "arguments": json.dumps({
                        "url": target_website
                    })
                },
                "id": "initial_navigation"
            })
            user_input = input("\nProvide a task instruction for testing (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
        
            while True:
                try:
                    agent_identified_result = await actor.attempt_perform_task(user_input, self)
                    printer.print('----------------------- Task Completed by Agent -----------------------')
                    printer.print(f'* Agent identified that the task result is: {agent_identified_result}')

                except Exception as e:
                    printer.print(f"Error processing user input: {e}")
                    traceback.print_exc()

                choice = input("\nPress 'r' to retry, otherwise for trying new task: ")
                if choice.lower() != 'r':
                    break
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client for connecting to a server.")
    parser.add_argument('target_website', help="Path to the server script (.py or .js)", type=str)
    args = parser.parse_args()

    asyncio.run(main(args.target_website))
