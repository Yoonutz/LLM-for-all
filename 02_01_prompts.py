"""
What it does: Compares Zero-shot, Role, Few-shot, and CoT on the same question.
"""
 
import os
from dotenv import load_dotenv
 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
 
 
load_dotenv()
 
# openai/gpt-4o-mini is excellent for basic tasks and supports SystemMessage.
# OpenRouter requires passing additional headers for identification.
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
 
task = "Explain how the attention mechanism works in transformers."
 
 
# ══════════════════════════════════════════
# 1. ZERO-SHOT — Naked question without context
# ══════════════════════════════════════════
print("=" * 50)
print("LEVEL 1: Zero-shot")
print("=" * 50)
# Zero-shot passes only a raw instruction. The LLM will respond at its discretion (length, format, detail).
response = llm.invoke([HumanMessage(content=task)])
print(response.content)
 
 
# ══════════════════════════════════════════
# 2. ROLE PROMPTING — Setting a role via SystemMessage
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("LEVEL 2: Role prompting")
print("=" * 50)
# Role Prompting sets the tone and style of the response. It doesn't improve facts, but helps match the format to the audience.
response = llm.invoke([
    SystemMessage(content="You are an experienced AI Engineer with 10 years of experience. You explain complex concepts simply and effectively."),
    HumanMessage(content=task)
])
print(response.content)
 
 
# ══════════════════════════════════════════
# 3. FEW-SHOT — Defining response format through examples
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("LEVEL 3: Few-shot + format")
print("=" * 50)
few_shot_prompt = """
Explain concepts using the following template:
1. One line — the essence
2. A real-life analogy
3. How it's used in code
 
Concept: What is gradient descent?
1. An optimization algorithm that minimizes model error.
2. Like descending a mountain in the fog — every step is in the direction of the steepest downward slope.
3. optimizer.step() in PyTorch performs one step of gradient descent.
 
Concept: """ + task
# Few-shot limits hallucinations and forces a rigid structure. Much more reliable than just asking in text.
response = llm.invoke([HumanMessage(content=few_shot_prompt)])
print(response.content)
 
 
# ══════════════════════════════════════════
# 4. CHAIN-OF-THOUGHT — Step-by-step reasoning
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("LEVEL 4: Chain-of-Thought")
print("=" * 50)
# Adding the phrase "Think step-by-step" (Chain-of-Thought) forces the LLM to output intermediate logical steps,
# which significantly increases the accuracy of complex reasoning.
cot_prompt = task + "\n\nThink step-by-step. First, describe the problem that attention solves, then the mechanism, then the math, and finally the output."
response = llm.invoke([
    SystemMessage(content="You are an ML researcher. Your reasoning is deep and structured."),
    HumanMessage(content=cot_prompt)
])
print(response.content)
