from dotenv import load_dotenv
load_dotenv()  # Must run before langchain_community imports so USER_AGENT is set in time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# ─────────────────────────────────────────────
# 1. LOCAL EMBEDDINGS  (must match the model used in 05_rag_loaders.py)
# ─────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# 2. LOAD EXISTING VECTORSTORE
# ─────────────────────────────────────────────
vectorstore = Chroma(
    persist_directory="05_vectorstore",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ─────────────────────────────────────────────
# 3. LLM  (OpenRouter — same as all other lessons)
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ─────────────────────────────────────────────
# 4. RAG PROMPT
# ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use only the following retrieved context to answer the question.
If you don't know the answer, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)

# ─────────────────────────────────────────────
# 5. RAG CHAIN  (LCEL)
# ─────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────
# 6. QUERIES
# ─────────────────────────────────────────────
questions = [
    "What is RAG and how does it work?",
    "What is GraphRAG and how is it different from classic RAG?",
    "What are embeddings in the context of RAG?",
    "What is multimodal RAG?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print("-" * 60)
