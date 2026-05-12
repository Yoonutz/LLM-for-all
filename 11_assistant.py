"""
What it does: Mini-Perplexity — Internet search via DuckDuckGo, result
summarization with quality validation, and saving notes indexed in Chroma
for semantic search. When querying notes, it parses sources for a full
answer, with a fallback internet search if data is insufficient.
"""

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx
from ddgs import DDGS
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma


# ══════════════════════════════════════════
# 1. MODEL + EMBEDDINGS
# ══════════════════════════════════════════
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = Chroma(
    persist_directory="./notes_db",
    embedding_function=embeddings,
)

# Directory for notes in Markdown format
NOTES_DIR = Path("./notes")
NOTES_DIR.mkdir(exist_ok=True)

# Page content character limit (~1,500 tokens for gpt-4o-mini)
MAX_PAGE_CHARS = 4000


# ══════════════════════════════════════════
# 2. PYDANTIC MODELS FOR VALIDATION
# ══════════════════════════════════════════

class ContentValidation(BaseModel):
    """Result of validating extracted page content."""

    is_valid: bool = Field(description="True — content contains useful information. False — junk.")
    summary: str = Field(description="Summary (3-5 sentences in English) or reason for rejection.")


class AnswerQuality(BaseModel):
    """Evaluation of context sufficiency for answering a question."""

    has_answer: bool = Field(description="Is there enough data for an answer?")
    answer: str = Field(description="The answer itself or an empty string.")


# ══════════════════════════════════════════
# 3. CHAINS
# ══════════════════════════════════════════

# --- Content Validator with few-shot examples ---
validation_examples = [
    {
        "input": "Title: Best AI Frameworks 2025\n"
                 "Text: Please enable JavaScript to view this page. Subscribe to unlock...",
        "output": '{"is_valid": false, "summary": "Paywall / JavaScript required"}',
    },
    {
        "input": "Title: LangGraph vs CrewAI\n"
                 "Text: LangGraph is a stateful orchestration framework for multi-agent systems...",
        "output": '{"is_valid": true, "summary": "LangGraph is a graph-based orchestration framework..."}',
    },
    {
        "input": "Title: Page Not Found\n"
                 "Text: 404 Error. The page you are looking for does not exist.",
        "output": '{"is_valid": false, "summary": "Page not found (404)"}',
    },
]

validation_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

validation_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=validation_example_prompt,
    examples=validation_examples,
)

validation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content validator. Determine if the content is useful or junk.\n"
               "If useful — write a summary (3-5 sentences in English).\n"
               "If junk — reject with a reason."),
    validation_few_shot,
    ("human", "Title: {title}\nText: {body}"),
])

validation_chain = validation_prompt | llm.with_structured_output(ContentValidation)


# --- Answer chain based on context ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Answer STRICTLY based on the provided context.\n"
               "If there is enough data — provide a detailed answer with facts.\n"
               "If not — say so honestly.\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])

answer_chain = answer_prompt | llm.with_structured_output(AnswerQuality)


# ══════════════════════════════════════════
# 4. FUNCTIONS
# ══════════════════════════════════════════

def fetch_page_content(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Downloads a page and extracts text, removing HTML tags.

    Args:
        url (str): Page address.
        max_chars (int, optional): Maximum characters to extract. Defaults to MAX_PAGE_CHARS.

    Returns:
        str: Cleaned page text.
    """
    try:
        with httpx.Client(timeout=10.0, verify=False, follow_redirects=True) as client:  # noqa: S501
            resp = client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; MiniPerplexity/1.0)"
            })
            resp.raise_for_status()
            html = resp.text

            # Remove non-content blocks
            for tag in ("script", "style", "nav", "header", "footer",
                        "aside", "noscript", "svg", "iframe"):
                html = re.sub(
                    rf"<{tag}[^>]*>.*?</{tag}>", " ",
                    html, flags=re.IGNORECASE | re.DOTALL
                )

            html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", html)

            # Decode HTML entities
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&#\d+;", " ", text)
            text = re.sub(r"&\w+;", " ", text)

            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
    except Exception:
        return ""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Performs a search using DuckDuckGo.

    Args:
        query (str): Search query.
        max_results (int, optional): Maximum results. Defaults to 5.

    Returns:
        list[dict]: List of results (title, url, body).
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, region="wt-wt", max_results=max_results)
    return results


def validate_and_summarize(title: str, body: str) -> ContentValidation:
    """Validates content and generates a summary (structured output + few-shot).

    Args:
        title (str): Page title.
        body (str): Cleaned text.

    Returns:
        ContentValidation: Result with is_valid and summary.
    """
    if not body or len(body) < 50:
        return ContentValidation(is_valid=False, summary="Content type empty or too short")
    return validation_chain.invoke({"title": title, "body": body[:MAX_PAGE_CHARS]})


def search_and_validate(query: str, max_results: int = 5) -> list[dict]:
    """Search → Load Pages → Validate → Return Results.

    Args:
        query (str): Search query.
        max_results (int, optional): Maximum results. Defaults to 5.

    Returns:
        list[dict]: Results with fields for index, title, url, summary, and is_valid.
    """
    print(f"\n  🌐 Searching DuckDuckGo for: '{query}'\n")
    try:
        raw_results = web_search(query, max_results)
    except Exception as e:
        print(f"  ❌ Search error: {e}\n")
        return []

    if not raw_results:
        print("  📭 Nothing found.\n")
        return []

    results = []
    for i, r in enumerate(raw_results):
        title = r.get("title", "")
        url = r.get("href", "")
        print(f"  ⏳ [{i+1}/{len(raw_results)}] {title[:60]}...")

        page_text = fetch_page_content(url)
        validation = validate_and_summarize(title, page_text)

        status = "✅" if validation.is_valid else "⛔"
        print(f"       {status} {validation.summary[:80]}")

        results.append({
            "index": i + 1,
            "title": title,
            "url": url,
            "summary": validation.summary,
            "is_valid": validation.is_valid,
        })
    return results


def save_note(query: str, items: list[dict]) -> str:
    """Saves valid summaries as Markdown and indexes them in Chroma.

    Args:
        query (str): Search query (used as the note title).
        items (list[dict]): Results (only entries with is_valid=True are saved).

    Returns:
        str: File path or an empty string.
    """
    valid_items = [item for item in items if item.get("is_valid", False)]
    if not valid_items:
        return ""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = "".join(
        c if c.isalnum() or c in " -_" else "_" for c in query
    )[:50].strip()
    filepath = NOTES_DIR / f"{timestamp}_{safe_name}.md"

    lines = [f"# {query}", f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
    for item in valid_items:
        lines.append(f"## {item['title']}")
        lines.append(f"> Source: {item['url']}\n")
        lines.append(item["summary"])
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")

    docs = [
        Document(
            page_content=f"{query}\n\n{item['title']}\n\n{item['summary']}",
            metadata={
                "source": "personal_note",
                "query": query,
                "title": item["title"],
                "url": item["url"],
                "file": str(filepath),
                "created_at": datetime.now().isoformat(),
            },
        )
        for item in valid_items
    ]
    vectorstore.add_documents(docs)
    return str(filepath)


def search_notes(query: str, k: int = 3) -> list[dict]:
    """Semantic search through notes in Chroma.

    Args:
        query (str): Search query.
        k (int, optional): Number of results. Defaults to 3.

    Returns:
        list[dict]: Found fragments with metadata.
    """
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("url", ""),
            "file": doc.metadata.get("file", ""),
        }
        for doc in results
    ]


def answer_from_context(question: str, contexts: list[str]) -> AnswerQuality:
    """Generates an answer based on context with a sufficiency check.

    Args:
        question (str): The question.
        contexts (list[str]): Contextual text fragments.

    Returns:
        AnswerQuality: Result containing has_answer and answer.
    """
    context_str = "\n\n---\n\n".join(contexts)
    return answer_chain.invoke({"context": context_str[:12000], "question": question})


# ══════════════════════════════════════════
# 5. MAIN LOOP
# ══════════════════════════════════════════
print(f"\n{'='*50}")
print("🔍 Mini Perplexity — Search, Summarize, Notes")
print(f"📚 Notes in database: {vectorstore._collection.count()}")
print(f"{'='*50}")
print("\nCommands:")
print("  /search <query>   — Search the internet + summarize")
print("  /notes <question> — Search internal notes")
print("  /list             — List note files")
print("  exit              — Quit\n")

while True:
    user_input = input("❓ ").strip()

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        break

    # ── /list ──
    if user_input.lower() == "/list":
        files = sorted(NOTES_DIR.glob("*.md"))
        if not files:
            print("  📭 No notes found.\n")
        else:
            print(f"  📁 Notes ({len(files)}):")
            for f in files:
                print(f"    - {f.name}")
            print()
        continue

    # ── /notes — Search notes with fallback ──
    if user_input.lower().startswith("/notes"):
        query = user_input[6:].strip()
        if not query:
            print("  Usage: /notes <question>\n")
            continue

        print(f"  🔎 Searching notes for: '{query}'\n")
        results = search_notes(query)

        if not results:
            # No notes → Fallback immediately
            print("  📭 No notes found — searching the internet...\n")
            search_results = search_and_validate(query)
            valid = [r for r in search_results if r["is_valid"]]
            if valid:
                contexts = []
                for r in valid[:3]:
                    page = fetch_page_content(r["url"])
                    if page:
                        contexts.append(f"Source: {r['title']}\n{page}")
                if contexts:
                    result = answer_from_context(query, contexts)
                    if result.has_answer:
                        print(f"\n  🤖 {result.answer}\n")
                    else:
                        print("  🤷 Could not find an answer.\n")
                filepath = save_note(query, search_results)
                if filepath:
                    print(f"  💾 Note saved → {filepath}")
                    print(f"  📚 DB Count: {vectorstore._collection.count()}\n")
            else:
                print("  🤷 Could not find an answer.\n")
            continue

        # Show notes
        for i, r in enumerate(results):
            print(f"  ── Note {i+1} ──")
            print(f"  📌 {r['title']}")
            print(f"  🔗 {r['url']}")
            print()

        # Parse sources
        urls = list({r["url"] for r in results if r["url"]})
        contexts = []
        for url in urls[:3]:
            print(f"  🌐 Reading source: {url}")
            page = fetch_page_content(url)
            if page:
                contexts.append(page)
        for r in results:
            contexts.append(f"Summary: {r['content']}")

        print("  ⏳ Analyzing...\n")
        result = answer_from_context(query, contexts)

        if result.has_answer:
            print(f"  🤖 {result.answer}\n")
        else:
            # Fallback
            print("  📭 Context is insufficient — searching the internet...\n")
            search_results = search_and_validate(query)
            valid = [r for r in search_results if r["is_valid"]]
            if valid:
                fallback_ctx = []
                for r in valid[:3]:
                    page = fetch_page_content(r["url"])
                    if page:
                        fallback_ctx.append(f"Source: {r['title']}\n{page}")
                if fallback_ctx:
                    fb = answer_from_context(query, fallback_ctx)
                    if fb.has_answer:
                        print(f"\n  🤖 (from web) {fb.answer}\n")
                    else:
                        print("  🤷 Could not find an answer.\n")
                        continue
                filepath = save_note(query, search_results)
                if filepath:
                    print(f"  💾 Note saved → {filepath}")
                    print(f"  📚 DB Count: {vectorstore._collection.count()}\n")
            else:
                print("  🤷 Could not find an answer.\n")
        continue

    # ── /search — Search + Validation + Save ──
    if user_input.lower().startswith("/search"):
        query = user_input[7:].strip()
        if not query:
            print("  Usage: /search <query>\n")
            continue

        results = search_and_validate(query)
        if not results:
            continue

        valid = [r for r in results if r["is_valid"]]

        print(f"\n  {'─'*40}")
        for s in results:
            status = "✅" if s["is_valid"] else "⛔"
            print(f"\n  {status} [{s['index']}] {s['title']}")
            print(f"      🔗 {s['url']}")
            print(f"      📝 {s['summary'][:200]}")
        print(f"\n  {'─'*40}")
        print(f"  Valid: {len(valid)} / {len(results)}")

        if not valid:
            print("  ⚠️ No valid results found.\n")
            continue

        choice = input("\n  💾 Save valid results? (numbers, 'all', or Enter to skip): ").strip()
        if not choice:
            print("  ⏭ Skipped.\n")
            continue

        if choice.lower() == "all":
            selected = valid
        else:
            try:
                indices = {int(x.strip()) for x in choice.split(",")}
                selected = [s for s in valid if s["index"] in indices]
            except ValueError:
                print("  ⚠️ Invalid input.\n")
                continue

        if not selected:
            print("  ⚠️ Nothing selected.\n")
            continue

        filepath = save_note(query, selected)
        if filepath:
            print(f"\n  ✅ Saved {len(selected)} notes → {filepath}")
            print(f"  📚 DB: {vectorstore._collection.count()} documents\n")
        continue

    print("  💡 Commands: /search <query> | /notes <question> | /list | exit\n")
