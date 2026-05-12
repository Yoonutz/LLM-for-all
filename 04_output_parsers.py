"""
What it does: Demonstrates the progression from unstructured output to nested Pydantic models.
"""
 
import os
from dotenv import load_dotenv
 
from pydantic import BaseModel, Field
from typing import List
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
 
 
load_dotenv()
# Initialize gpt-4o-mini (it handles structured output exceptionally well)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
 
# --- Level 1: No Parser — Chaos ---
print("=" * 50)
print("LEVEL 1: Without Structure")
print("=" * 50)
# StrOutputParser simply returns the raw string.
# If we need to pass this data to other systems (like JSON), we'd have to write fragile regex.
chain_raw = ChatPromptTemplate.from_messages([
    ("human", "Write a review of the movie The Matrix")
]) | llm | StrOutputParser()
print(chain_raw.invoke({}))
 
# --- Level 2: Pydantic Model — Structure ---
print("\n" + "=" * 50)
print("LEVEL 2: PydanticOutputParser")
print("=" * 50)
 
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="Rating from 1 to 10")
    summary: str = Field(description="A brief one-sentence summary")
    themes: List[str] = Field(description="A list of the main themes of the movie")
 
# PydanticOutputParser generates formatting instructions (format_instructions),
# which tell the LLM exactly how to structure the JSON response.
parser = PydanticOutputParser(pydantic_object=MovieReview)
 
prompt = PromptTemplate(
    template="Write a movie review.\n{format_instructions}\nMovie: {movie}",
    input_variables=["movie"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
 
chain = prompt | llm | parser
review = chain.invoke({"movie": "Fight Club"})
print(f"Movie: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
print(f"Themes: {', '.join(review.themes)}")
 
# --- Level 3: with_structured_output — The Modern Way ---
print("\n" + "=" * 50)
print("LEVEL 3: with_structured_output (Modern)")
print("=" * 50)
 
# with_structured_output is the recommended approach in modern frameworks.
# We don't explicitly pass the format into the prompt; the LLM uses its native 
# Structured Output API (Function Calling). This is faster and more reliable.
llm_structured = llm.with_structured_output(MovieReview)
 
chain_modern = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. Analyze movies deeply."),
    ("human", "Write a review of the movie: {movie}")
]) | llm_structured
 
review2 = chain_modern.invoke({"movie": "Interstellar"})
print(f"Movie: {review2.title}")
print(f"Rating: {review2.rating}/10")
print(f"Summary: {review2.summary}")
print(f"Themes: {', '.join(review2.themes)}")
 
# --- Level 4: Nested Models ---
print("\n" + "=" * 50)
print("LEVEL 4: Nested Pydantic Models")
print("=" * 50)
 
class Director(BaseModel):
    name: str = Field(description="The director's name")
    style: str = Field(description="The director's style in one word")
 
class DetailedReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="Rating from 1 to 10")
    director: Director = Field(description="Information about the director")
    pros: List[str] = Field(description="List of pros")
    cons: List[str] = Field(description="List of cons")
 
# You can build arbitrarily complex nested structures; Pydantic validates each layer.
# The model is guaranteed to return a JSON matching this nested structure.
llm_detailed = llm.with_structured_output(DetailedReview)
 
chain_detailed = ChatPromptTemplate.from_messages([
    ("system", "You are a professional movie critic."),
    ("human", "Provide a detailed review of the movie: {movie}")
]) | llm_detailed
 
result = chain_detailed.invoke({"movie": "Inception"})
print(f"Movie: {result.title}")
print(f"Rating: {result.rating}/10")
print(f"Director: {result.director.name} — Style: {result.director.style}")
print(f"Pros: {', '.join(result.pros)}")
print(f"Cons: {', '.join(result.cons)}")
