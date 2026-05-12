"""
What it does: Progresses from a simple @tool to a full ReAct agent with multiple tools.
"""

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ══════════════════════════════════════════
# 1. DEFINING TOOLS
# ══════════════════════════════════════════
print("=" * 50)
print("1. Defining Tools")
print("=" * 50)


@tool
def calculator(expression: str) -> str:
    """Calculates a mathematical expression.

    Args:
        expression (str): Mathematical expression as a string (e.g., '2 + 2' or '17 ** 5').

    Returns:
        str: Result of the calculation or an error message.
    """
    try:
        # Note: Using eval is generally unsafe in production; use a safe math parser instead.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_time() -> str:
    """Returns the current date and time.

    Returns:
        str: Current date and time in 'YYYY-MM-DD HH:MM:SS' format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def reverse_string(text: str) -> str:
    """Reverses a string.

    Args:
        text (str): The original string to be reversed.

    Returns:
        str: The reversed string.
    """
    return text[::-1]


@tool
def file_writer(filename: str, content: str) -> str:
    """Writes text to a file.

    Args:
        filename (str): Name of the file to write to.
        content (str): Text content to be saved.

    Returns:
        str: Status message of the write operation.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {filename}"


tools = [calculator, get_time, reverse_string, file_writer]
print(f"Tools registered: {len(tools)}")
for t in tools:
    print(f"  - {t.name}: {t.description}")

# ══════════════════════════════════════════
# 2. SINGLE TOOL CALL
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("2. Single Call — llm.bind_tools()")
print("=" * 50)

llm_with_tools = llm.bind_tools(tools)

# Send a request — model decides which tool to call
response = llm_with_tools.invoke("What is 17 to the fifth power?")
print(f"Model response: {response.content}")
print(f"Tool calls: {response.tool_calls}")

if response.tool_calls:
    tc = response.tool_calls[0]
    print(f"\nModel wants to call: {tc['name']}")
    print(f"With arguments: {tc['args']}")

# ══════════════════════════════════════════
# 3. FULL REACT LOOP (MANUAL)
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("3. ReAct Cycle — Full Agent")
print("=" * 50)

tools_map = {t.name: t for t in tools}


def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """Executes the loop for a ReAct agent.

    ReAct agent: LLM thinks → calls tool → thinks again.
    The cycle continues until the model returns a final answer.

    Args:
        user_query (str): The user's text query.
        max_iterations (int, optional): Maximum number of steps (thought+action). Defaults to 5.

    Returns:
        str: The final text response from the agent or an iteration limit message.
    """
    messages = [
        {"role": "system", "content": """You are an AI assistant with tools. Use the ReAct (Reasoning and Acting) style.
Before every action, reason logically. Your cycle is:
1. Thought: Think about what needs to be done.
2. Action: Call the appropriate tool if required.
3. Observation: Receive and analyze the result.
Use tools for calculations, getting the time, and writing files."""},
        HumanMessage(content=user_query),
    ]

    for i in range(max_iterations):
        print(f"\n  [Iteration {i+1}]")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("  → Final Response")
            return str(response.content)

        # Execute all tool_calls
        for tc in response.tool_calls:
            print(f"  → Calling: {tc['name']}({tc['args']})")
            tool_fn = tools_map[tc["name"]]
            try:
                result = tool_fn.invoke(tc["args"])
                print(f"  ← Result: {result}")
            except Exception as e:
                result = f"Tool execution error: {e}"
                print(f"  ❌ {result}")

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
            ))

    return "Iteration limit exceeded"


# --- Test 1: Simple ---
print("\n--- Test 1: Simple ---")
answer = run_agent("What time is it now?")
print(f"\nResponse: {answer}")

# --- Test 2: Multi-step ---
print("\n--- Test 2: Multi-step ---")
answer = run_agent("Calculate 2^10 and reverse the result as a string")
print(f"\nResponse: {answer}")

# --- Test 3: No Tool Needed ---
print("\n--- Test 3: Question Without Tool ---")
answer = run_agent("Tell me a joke")
print(f"\nResponse: {answer}")

# ══════════════════════════════════════════
# 4. DIALOGUE AGENT
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("4. Dialogue Agent (type 'exit' to quit)")
print("=" * 50)

dialog_history = [
    {"role": "system", "content": """You are an AI assistant with tools. Use the ReAct (Reasoning and Acting) style.
Before every action, reason logically (Thought). Rules:
1. Use 'calculator' for math.
2. Use 'get_time' for the current time.
3. Use 'reverse_string' to reverse text.
4. Use 'file_writer' to write to files.
5. If no tool is needed — respond directly."""},
]

while True:
    user_input = input("\n❓ Question: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    dialog_history.append(HumanMessage(content=user_input))

    for i in range(5):
        response = llm_with_tools.invoke(dialog_history)
        dialog_history.append(response)

        if not response.tool_calls:
            print(f"\n🤖 {response.content}")
            break

        for tc in response.tool_calls:
            print(f"  🔧 {tc['name']}({tc['args']})")
            try:
                result = tools_map[tc["name"]].invoke(tc["args"])
                print(f"  📋 → {result}")
            except Exception as e:
                result = f"Error: {e}"
                print(f"  ❌ → {result}")
            dialog_history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
