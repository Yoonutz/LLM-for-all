"""
What it does: ChatPromptTemplate with history, multi-language support, FewShotChatMessagePromptTemplate, and intent classifier.
"""
 
import os
 
from dotenv import load_dotenv
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
 
load_dotenv()
 
# Use gpt-4o-mini for fast and high-quality responses
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
 
parser = StrOutputParser()
 
 
# ══════════════════════════════════════════
# 1. BASIC ChatPromptTemplate + LCEL
# ══════════════════════════════════════════
print("=" * 50)
print("1: ChatPromptTemplate — Basic")
print("=" * 50)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role}."),
    ("human", "{question}")
])
 
# First LCEL chain: prompt | llm | parser
# Variables (role, question) are passed to the template, it forms messages,
# passes them to the LLM, and the parser strips the response of metadata
chain = prompt | llm | parser
print(chain.invoke({
    "role": "an AI Engineer explaining LangChain to students",
    "question": "Why use ChatPromptTemplate instead of a regular string?"
}))
 
 
# ══════════════════════════════════════════
# 2. ChatPromptTemplate WITH HISTORY
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("2: ChatPromptTemplate — With History")
print("=" * 50)
prompt_history = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineering mentor. Answer concisely."),
    ("human", "What is LCEL?"),
    ("ai", "LCEL is a way to build chains using the | operator. prompt | llm | parser."),
    ("human", "{follow_up}")
])
 
# A chain with history allows you to set context so the LLM understands what the follow-up question refers to
chain_history = prompt_history | llm | parser
print(chain_history.invoke({
    "follow_up": "And why is the parser needed at the end of the chain?"
}))
 
 
# ══════════════════════════════════════════
# 3. MULTI-LANGUAGE TEMPLATE
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("3: ChatPromptTemplate — Multi-language")
print("=" * 50)
prompt_lang = ChatPromptTemplate.from_messages([
    ("system", "Respond strictly in language: {language}. You are an expert on {topic}."),
    ("human", "{question}")
])
 
# A template with many variables is useful for creating universal bots (e.g., multi-language ones)
chain_lang = prompt_lang | llm | parser
print(chain_lang.invoke({
    "language": "English",
    "topic": "transformers in Deep Learning",
    "question": "Explain the self-attention mechanism in two sentences"
}))
 
 
# ══════════════════════════════════════════
# 4. FEW-SHOT CHAT TEMPLATE
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("4: FewShotChatMessagePromptTemplate")
print("=" * 50)
examples = [
    {
        "input": "What is a token?",
        "output": "🔹 Essence: A unit of text for the model.\n🔹 Analogy: Like a syllable in a word.\n🔹 Example: 'LangChain' = 2 tokens."
    },
    {
        "input": "What is an embedding?",
        "output": "🔹 Essence: A numerical vector of a word's meaning.\n🔹 Analogy: Coordinates on a map of concepts.\n🔹 Example: 'King' and 'Czar' are close to each other in this space."
    },
]
 
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])
 
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
 
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI mentor. Explain using the template from the examples."),
    few_shot_prompt,
    ("human", "{question}")
])
 
# FewShotChatMessagePromptTemplate trains the LLM on dialogue examples,
# forcing it to generate responses in the specified style and format.
chain_few_shot = final_prompt | llm | parser
print(chain_few_shot.invoke({
    "question": "What is attention?"
}))
 
 
# ══════════════════════════════════════════
# 5. FEW-SHOT INTENT CLASSIFIER
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("5: Few-shot Intent Classifier")
print("=" * 50)
intent_examples = [
    {"input": "Install langchain", "output": "COMMAND"},
    {"input": "What is a transformer?", "output": "QUESTION"},
    {"input": "Write a script to call the model", "output": "TASK"},
    {"input": "Explain LCEL in simple terms", "output": "QUESTION"},
]
 
intent_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])
 
intent_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=intent_example_prompt,
    examples=intent_examples,
)
 
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user's request. Answer with one word: COMMAND, QUESTION, or TASK."),
    intent_few_shot,
    ("human", "{input}")
])
 
# This approach is often used in agent routers — when you need to redirect a request
# to a specialist based on the intent.
chain_intent = intent_prompt | llm | parser
 
test_inputs = [
    "How does gradient descent work?",
    "Create an AI agent with memory",
    "pip install openai",
]
for user_input in test_inputs:
    result = chain_intent.invoke({"input": user_input})
    print(f"'{user_input}' → {result.strip()}")
