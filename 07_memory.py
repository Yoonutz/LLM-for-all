"""
What it does: Demonstrates three memory strategies for LLM dialogues using LangGraph.
"""

import os

from dotenv import load_dotenv
load_dotenv()  # Must run before langchain imports so USER_AGENT is set in time

from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START

# ──────────────────────────────────────────
# LLM via OpenRouter
# ──────────────────────────────────────────
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_MSG = SystemMessage("You are a friendly AI assistant. Answer briefly.")

# ══════════════════════════════════════════
# 1. BUFFER MEMORY — Entire history in RAM
#    MemorySaver stores all messages per thread_id.
#    History is lost when the process exits.
# ══════════════════════════════════════════
print("=" * 50)
print("1. Buffer Memory — Full History")
print("=" * 50)

def buffer_node(state: MessagesState) -> dict:
    """Passes full conversation history to the LLM and returns the response.

    Args:
        state (MessagesState): Current graph state containing all messages.

    Returns:
        dict: New AI message to append to state.
    """
    response = llm.invoke([SYSTEM_MSG] + state["messages"])
    return {"messages": [response]}

buffer_graph = StateGraph(state_schema=MessagesState)
buffer_graph.add_edge(START, "model")
buffer_graph.add_node("model", buffer_node)
buffer_app = buffer_graph.compile(checkpointer=MemorySaver())

# thread_id is the LangGraph equivalent of session_id
cfg = {"configurable": {"thread_id": "triangle_session"}}

q1 = "Hi, my name is Triangle!"
print(f"Human: {q1}")
r1 = buffer_app.invoke({"messages": [HumanMessage(q1)]}, config=cfg)
print(f"AI: {r1['messages'][-1].content}\n")

q2 = "What is my name?"
print(f"Human: {q2}")
r2 = buffer_app.invoke({"messages": [HumanMessage(q2)]}, config=cfg)
print(f"AI: {r2['messages'][-1].content}\n")

# Inspecting memory contents via get_state()
snapshot = buffer_app.get_state(cfg)
msgs = snapshot.values["messages"]
print(f"Messages in memory: {len(msgs)}")
for msg in msgs:
    label = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{label}]: {msg.content[:60]}")


# ══════════════════════════════════════════
# 2. WINDOW MEMORY — Only the last K messages
#    MemorySaver stores the full history, but
#    trim_messages() limits what the LLM sees.
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("2. Window Memory — trim_messages(max_messages=2)")
print("=" * 50)

MAX_MESSAGES = 2  # Try 2, 4, 8 — at what value does the model remember your name?

def window_node(state: MessagesState) -> dict:
    """Trims history to the last MAX_MESSAGES before calling the LLM.

    Full history is preserved in the graph state; only the LLM context is trimmed.

    Args:
        state (MessagesState): Current graph state containing all messages.

    Returns:
        dict: New AI message to append to state.
    """
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=MAX_MESSAGES,
        token_counter=len,
        start_on="human",
    )
    response = llm.invoke([SYSTEM_MSG] + trimmed)
    return {"messages": [response]}

window_graph = StateGraph(state_schema=MessagesState)
window_graph.add_edge(START, "model")
window_graph.add_node("model", window_node)
window_app = window_graph.compile(checkpointer=MemorySaver())

w_cfg = {"configurable": {"thread_id": "w1"}}

print("Human: My name is Triangle. I'm learning LCEL.")
window_app.invoke({"messages": [HumanMessage("My name is Triangle. I'm learning LCEL.")]}, config=w_cfg)
print("Human: In the last lesson, I mastered RAG.")
window_app.invoke({"messages": [HumanMessage("In the last lesson, I mastered RAG.")]}, config=w_cfg)
print("Human: Now I'm studying memory in LangChain.")
window_app.invoke({"messages": [HumanMessage("Now I'm studying memory in LangChain.")]}, config=w_cfg)

print(f"\n(Only the last {MAX_MESSAGES} messages sent to LLM. Triangle's name is already forgotten.)")
print("Human: What is my name?")
r_window = window_app.invoke({"messages": [HumanMessage("What is my name?")]}, config=w_cfg)
print(f"AI (window={MAX_MESSAGES}): {r_window['messages'][-1].content}\n")


# ══════════════════════════════════════════
# 3. PERSISTENT MEMORY — SQLite on disk
#    SqliteSaver survives process restarts.
#    Replaces FileChatMessageHistory (JSON).
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("3. Persistent Memory — SQLite (07_chat_history.db)")
print("=" * 50)

DB_FILE = "07_chat_history.db"

def persistent_node(state: MessagesState) -> dict:
    """Passes full conversation history to the LLM and returns the response.

    Args:
        state (MessagesState): Current graph state containing all messages.

    Returns:
        dict: New AI message to append to state.
    """
    response = llm.invoke([SYSTEM_MSG] + state["messages"])
    return {"messages": [response]}

with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
    persistent_graph = StateGraph(state_schema=MessagesState)
    persistent_graph.add_edge(START, "model")
    persistent_graph.add_node("model", persistent_node)
    persistent_app = persistent_graph.compile(checkpointer=checkpointer)

    file_cfg = {"configurable": {"thread_id": "triangle_persistent"}}

    snapshot = persistent_app.get_state(file_cfg)
    existing_msgs = snapshot.values.get("messages", []) if snapshot.values else []

    if existing_msgs:
        print(f"Found old history: {len(existing_msgs)} messages — continuing dialogue.\n")
        q_file = "What notes or facts about me do you remember?"
    else:
        print("New history created.\n")
        q_file = "Remember: My name is Triangle. Note 1: LCEL is a pipeline via |. Note 2: RAG is search by documents."

    print(f"Human: {q_file}")
    r_file = persistent_app.invoke({"messages": [HumanMessage(q_file)]}, config=file_cfg)
    print(f"AI: {r_file['messages'][-1].content}\n")

print(f"History saved to: {DB_FILE}")
print("Run the script again — the AI will remember you from the SQLite database!")
