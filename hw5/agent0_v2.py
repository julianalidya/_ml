import os
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def read_file(path: str) -> str:
    full_path = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    full_path = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written: {full_path}"


def list_files(directory: str = ".") -> str:
    full_path = os.path.join(WORK_DIR, directory) if not os.path.isabs(directory) else directory
    entries = os.listdir(full_path)
    return "\n".join(entries)


TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Content to write into the file."}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path to list. Defaults to current directory."}
            },
            "required": []
        }
    }
]

SYSTEM_PROMPT = """
<agent>
  <role>You are a helpful file-management assistant.</role>
  <capabilities>
    <capability>Read files using the read_file tool</capability>
    <capability>Write files using the write_file tool</capability>
    <capability>List directory contents using the list_files tool</capability>
  </capabilities>
  <instructions>
    <instruction>Use tools when the user asks you to interact with files.</instruction>
    <instruction>Always confirm what action you took after using a tool.</instruction>
    <instruction>If unsure about a path, list the directory first.</instruction>
  </instructions>
</agent>
"""


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "read_file":
        return read_file(inputs["path"])
    elif name == "write_file":
        return write_file(inputs["path"], inputs["content"])
    elif name == "list_files":
        return list_files(inputs.get("directory", "."))
    else:
        return f"Unknown tool: {name}"


def run_agent(user_message: str):
    print(f"\n[User]: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        for block in response.content:
            if hasattr(block, "text"):
                print(f"[Agent]: {block.text}")

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  >> Tool call: {block.name}({block.input})")
                result = dispatch_tool(block.name, block.input)
                print(f"  << Result: {result[:200]}{'...' if len(result) > 200 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    print("=== v2-agent-xml ===")
    print(f"Working directory: {WORK_DIR}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit", "bye"):
                print("Goodbye!")
                break
            if user_input:
                run_agent(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
