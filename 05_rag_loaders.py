from dotenv import load_dotenv
load_dotenv()  # Must run before langchain_community imports so USER_AGENT is set in time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# 1. LOCAL EMBEDDINGS  (free, no API key needed)
# ─────────────────────────────────────────────
# Using sentence-transformers/all-MiniLM-L6-v2
# First run will download the model (~90 MB) from HuggingFace.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# 2. TEXT SPLITTER
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ─────────────────────────────────────────────
# 3. LOAD DOCUMENTS
# ─────────────────────────────────────────────
documents = []

# .txt files
try:
    loader = TextLoader("05_docs/notes/01_rag_multimodal.txt", encoding="utf-8")
    documents.extend(loader.load())
    print("Loaded: 01_rag_multimodal.txt")
except Exception as e:
    print(f"Skipped 01_rag_multimodal.txt: {e}")

try:
    loader = TextLoader("05_docs/notes/02_graphdb_graphrag.txt", encoding="utf-8")
    documents.extend(loader.load())
    print("Loaded: 02_graphdb_graphrag.txt")
except Exception as e:
    print(f"Skipped 02_graphdb_graphrag.txt: {e}")

# .txt references file
try:
    loader = TextLoader("05_docs/references/references.txt", encoding="utf-8")
    documents.extend(loader.load())
    print("Loaded: references.txt")
except Exception as e:
    print(f"Skipped references.txt: {e}")

# .csv file
try:
    loader = CSVLoader("05_docs/csv/01_rag_eval.csv")
    documents.extend(loader.load())
    print("Loaded: 01_rag_eval.csv")
except Exception as e:
    print(f"Skipped 01_rag_eval.csv: {e}")

# .pdf files
for pdf_name in [
    "01_rag_beyond.pdf",
    "02_graphrag_microsoft.pdf",
    "03_ea_graphrag.pdf",
]:
    try:
        loader = PyPDFLoader(f"05_docs/pdf/{pdf_name}")
        documents.extend(loader.load())
        print(f"Loaded: {pdf_name}")
    except Exception as e:
        print(f"Skipped {pdf_name}: {e}")

# .docx files
for docx_name in ["01_qdrant.docx", "02_milvus.docx"]:
    try:
        loader = Docx2txtLoader(f"05_docs/docx/{docx_name}")
        documents.extend(loader.load())
        print(f"Loaded: {docx_name}")
    except Exception as e:
        print(f"Skipped {docx_name}: {e}")

# web pages
urls = [
    "https://docs.langchain.com/langsmith/evaluation-approaches#retrieval-augmented-generation-rag",
    "https://docs.langchain.com/oss/python/langchain/retrieval#2-step-rag",
    "https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain",
    "https://docs.langchain.com/langsmith/evaluate-rag-tutorial#evaluate-a-rag-application",
]
try:
    loader = WebBaseLoader(urls)
    documents.extend(loader.load())
    print(f"Loaded {len(urls)} web pages")
except Exception as e:
    print(f"Skipped web pages: {e}")

# ─────────────────────────────────────────────
# 4. SPLIT AND STORE
# ─────────────────────────────────────────────
print(f"\nTotal raw documents loaded: {len(documents)}")

splits = text_splitter.split_documents(documents)
print(f"Total chunks after splitting: {len(splits)}")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory="05_vectorstore",
)

print(f"\nVectorstore created at 05_vectorstore/")
print(f"Total vectors stored: {vectorstore._collection.count()}")
