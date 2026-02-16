import argparse
import json
import os
import subprocess
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Initialize the conversation with the user's prompt
    messages = [{"role": "user", "content": args.p}]
    
    # Use print statements for debugging, they'll be visible when running tests.
    print("Logs from the program will appear here!", file=sys.stderr)
    
    # Agent loop
    while True:
        # Send messages to the model
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read"
                                }
                            },
                            "required": ["file_path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the file to write to"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file"
                                }
                            },
                            "required": ["file_path", "content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute"
                                }
                            },
                            "required": ["command"]
                        }
                    }
                }
            ]
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # Get the response message
        response = chat.choices[0]
        message = response.message
        
        # Add the assistant's response to the message history
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else [])
            ] if hasattr(message, 'tool_calls') and message.tool_calls else None
        })
        
        # Check if there are tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the Read tool
                if function_name == "Read":
                    file_path = arguments.get("file_path")
                    if file_path:
                        with open(file_path, 'r') as f:
                            contents = f.read()
                        
                        # Add the tool result to the message history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": contents
                        })
                
                # Execute the Write tool
                elif function_name == "Write":
                    file_path = arguments.get("file_path")
                    content = arguments.get("content")
                    if file_path:
                        with open(file_path, 'w') as f:
                            f.write(content)
                        
                        # Add the tool result to the message history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "File written successfully"
                        })
                
                # Execute the Bash tool
                elif function_name == "Bash":
                    command = arguments.get("command")
                    if command:
                        try:
                            result = subprocess.run(
                                command,
                                shell=True,
                                capture_output=True,
                                text=True
                            )
                            # Combine stdout and stderr
                            output = result.stdout + result.stderr
                            if not output:
                                output = "Command executed successfully"
                            
                            # Add the tool result to the message history
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": output
                            })
                        except Exception as e:
                            # Add error message to the message history
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error executing command: {str(e)}"
                            })
        else:
            # No tool calls, we're done - print the final response
            print(message.content)
            break

if __name__ == "__main__":
    main()
