"""
What it does: sends a HumanMessage via OpenRouter and gets a response from the model.
"""

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


load_dotenv()

# OpenRouter is compatible with the OpenAI API — we use ChatOpenAI with a base_url swap
llm = ChatOpenAI(
    model="openrouter/auto",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

user_input = (
    "Morpheus offers me two options: the blue pill — stay a Python developer and forget "
    "about AGI, the red pill — find out how deep the AI Engineering rabbit hole goes. "
    "Which pill should I choose and what happens next?\n"
)
user_input = user_input + input("> ")

response = llm.invoke([HumanMessage(content=user_input)])

print("\nResponse:\n", response.content)