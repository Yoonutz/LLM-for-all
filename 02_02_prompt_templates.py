"""
What it does: Demonstrates PromptTemplate, ChatPromptTemplate, and FewShotPromptTemplate.
"""
 
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# Initialize the LLM to demonstrate templates in real-world conditions
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
 
 
# ══════════════════════════════════════════
# 1. SIMPLE TEMPLATE — PromptTemplate
# ══════════════════════════════════════════
template = """
Write a story about {character} in the style of {style}.
The action takes place in {setting}.
"""
 
prompt = PromptTemplate.from_template(template)
 
# Formatting — substituting variables
story = prompt.format(
    character="a cyberpunk detective",
    style="noir",
    setting="Neo-Tokyo 2077"
)
 
print("\n\n=== 01. Simple Template ===\n", story)
 
# The template substituted the values, now we can pass the string to the LLM
# PromptTemplate is useful when you need one long text without roles
response1 = llm.invoke([HumanMessage(content=story)])
print("🤖 LLM Response:\n", response1.content)
 
 
# ══════════════════════════════════════════
# 2. CHAT TEMPLATE — ChatPromptTemplate
# ══════════════════════════════════════════
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
 
prompt_messages = chat_template.format_messages(question="Hi, how are you?")
print("\n\n=== 02. Chat Template ===\n", prompt_messages)
 
# Chat templates are preferred for modern LLMs as they operate in a dialogue format.
# We explicitly break the prompt into system and human messages.
response2 = llm.invoke(prompt_messages)
print("\n🤖 LLM Response:\n", response2.content)
 
 
# ══════════════════════════════════════════
# 3. FEW-SHOT TEMPLATE — Learning by Example
# ══════════════════════════════════════════
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "fast", "output": "slow"},
]
 
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Word: {input}\nAntonym: {output}"
)
 
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
)
 
formatted_few_shot = few_shot_prompt.format(input="brave")
print("\n\n=== 03. Few-shot Template ===\n", formatted_few_shot)
 
# The LLM will see the examples and understand that it is required to provide only a single antonym word
response3 = llm.invoke([HumanMessage(content=formatted_few_shot)])
print("🤖 LLM Response:\n", response3.content)
