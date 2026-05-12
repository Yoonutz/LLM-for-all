"""
What it does: Demonstrates the progression from manual calls to LCEL with streaming.
"""
import os
 
from dotenv import load_dotenv
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
 
load_dotenv()
 
# gpt-4o-mini is used as the default model.
# OpenRouter requires additional headers for proper routing.
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
parser = StrOutputParser()
 
# --- Level 1: Just the model ---
print("=" * 50)
print("LEVEL 1: Without LCEL — Manual")
print("=" * 50)
# We call the model directly, passing a raw list of messages.
# The model returns an AIMessage object, from which we must manually extract .content.
response = llm.invoke([
    SystemMessage(content="You are an AI Engineer."),
    HumanMessage(content="What is a chain in LangChain? One sentence.")
])
print(response.content)
 
# --- Level 2: LCEL — prompt | llm | parser ---
print("\n" + "=" * 50)
print("LEVEL 2: LCEL pipe")
print("=" * 50)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineer. Answer briefly and to the point."),
    ("human", "{question}")
])
 
# LCEL (LangChain Expression Language) allows you to connect components via |.
# Data is passed from left to right: prompt(dictionary) -> llm(string) -> parser(string).
chain = prompt | llm | parser
 
response = chain.invoke({"question": "What is LCEL in LangChain?"})
print(response)
 
# --- Level 3: Variables in the prompt ---
print("\n" + "=" * 50)
print("LEVEL 3: Dynamic Prompt")
print("=" * 50)
prompt_dynamic = ChatPromptTemplate.from_messages([
    ("system", "You are {role}. Style: {style}."),
    ("human", "{question}")
])
 
# Dynamic templates allow you to create reusable chains.
# We pass multiple variables at once, and the template substitutes them into the correct places.
chain_dynamic = prompt_dynamic | llm | parser
 
response = chain_dynamic.invoke({
    "role": "an experienced ML instructor",
    "style": "explain it as if to a first-year student",
    "question": "Why are chains needed in AI applications?"
})
print(response)
 
# --- Level 4: Streaming ---
print("\n" + "=" * 50)
print("LEVEL 4: Token Streaming")
print("=" * 50)
# Streaming is useful for long responses (e.g., in a UI).
# The chain starts delivering the result in parts (chunks) as the model generates tokens.
for chunk in chain.stream({"question": "Explain the pipe | operator in LCEL in one paragraph."}):
    print(chunk, end="", flush=True)
print()
