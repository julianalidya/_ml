import os
import anthropic

client = anthropic.Anthropic()
MODEL    = "claude-sonnet-4-6"
REVIEWER = "claude-sonnet-4-6"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(WORK_DIR, path))
    return os.path.normpath(path)


def is_inside_workdir(path: str) -> bool:
    resolved = resolve_path(path)
    return resolved.startswith(WORK_DIR + os.sep) or resolved == WORK_DIR


def request_user_approval(tool_name: str, path: str) -> bool:
    print("\n" + "="*60)
    print("[SECURITY] ⚠️  Outside-directory access detected!")
    print(f"  Tool     : {tool_name}")
    print(f"  Path     : {path}")
    print(f"  Work dir : {WORK_DIR}")
    print("="*60)
    while True:
        answer = input("Allow this access? (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            print("[SECURITY] ✅ Access approved by user.\n")
            return True
        elif answer in ("no", "n"):
            print("[SECURITY] ❌ Access denied by user.\n")
            return False
        else:
            print("Please type 'yes' or 'no'.")


REVIEWER_SYSTEM = """
<reviewer>
  <role>You are a strict security reviewer for an AI agent.</role>
  <task>
    You will receive a description of a tool call the agent wants to make.
    Decide whether it is safe to execute.
  </task>
  <rules>
    <rule>Deny any action that could delete, overwrite, or corrupt critical system files.</rule>
    <rule>Deny any action that attempts to read sensitive files (e.g. /etc/passwd, ~/.ssh/*).</rule>
    <rule>Deny any action that tries to exfiltrate data to external services.</rule>
    <rule>Allow normal file read/write within the working directory.</rule>
  </rules>
  <response_format>
    Reply with ONLY one of:
      APPROVE - brief reason
      DENY    - brief reason
  </response_format>
</reviewer>
"""

def llm_review(tool_name: str, tool_inputs: dict) -> tuple[bool, str]:
    prompt = f"""
<tool_call>
  <name>{tool_name}</name>
  <inputs>{tool_inputs}</inputs>
  <workdir>{WORK_DIR}</workdir>
</tool_call>

Should this tool call be executed?
"""
    resp = client.messages.create(
        model=REVIEWER,
        max_tokens=256,
        system=REVIEWER_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = resp.content[0].text.strip()
    approved = verdict.upper().startswith("APPROVE")
    return approved, verdict


def read_file(path: str) -> str:
    full_path = resolve_path(path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    full_path = resolve_path(path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written: {full_path}"


def list_files(directory: str = ".") -> str:
    full_path = resolve_path(directory)
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
                "directory": {"type": "string", "description": "Directory path to list."}
            },
            "required": []
        }
    }
]

SYSTEM_PROMPT = """
<agent>
  <role>You are a helpful file-management assistant with security awareness.</role>
  <capabilities>
    <capability>Read files using the read_file tool</capability>
    <capability>Write files using the write_file tool</capability>
    <capability>List directory contents using the list_files tool</capability>
  </capabilities>
  <instructions>
    <instruction>Use tools when the user asks you to interact with files.</instruction>
    <instruction>Always confirm what action you took after using a tool.</instruction>
    <instruction>If unsure about a path, list the directory first.</instruction>
    <instruction>You operate under strict security controls. Some actions may be blocked or require approval.</instruction>
  </instructions>
</agent>
"""

def secure_dispatch(name: str, inputs: dict) -> str:
    print(f"\n  [Review] Checking tool call with reviewer LLM...")
    approved_by_llm, reason = llm_review(name, inputs)
    print(f"  [Review] Verdict: {reason}")

    if not approved_by_llm:
        return f"[BLOCKED by security reviewer] {reason}"

    path_tools = {"read_file": "path", "write_file": "path", "list_files": "directory"}
    if name in path_tools:
        path_key = path_tools[name]
        path_val = inputs.get(path_key, ".")
        if not is_inside_workdir(path_val):
            approved_by_user = request_user_approval(name, path_val)
            if not approved_by_user:
                return f"[BLOCKED] User denied access to outside path: {path_val}"

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
                result = secure_dispatch(block.name, block.input)
                print(f"  << Result: {result[:200]}{'...' if len(result) > 200 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    print("=== v3-agent-secure ===")
    print(f"Working directory: {WORK_DIR}")
    print("Security features: LLM review + outside-path user approval\n")

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
