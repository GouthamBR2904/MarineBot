import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess

# ===============================
# PATH CONFIG
# ===============================
DATA_PATH = Path("../data/merged_marine_info.json")
DB_PATH = Path("../vector_store/chroma")

# ===============================
# EMBEDDINGS (MiniLM)
# ===============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return embedding_model.encode([text]).tolist()[0]

# ===============================
# LOAD DATASET
# ===============================
def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ===============================
# CREATE VECTOR DB
# ===============================
def create_vector_db():
    data = load_dataset()
    documents = [item["content"] for item in data if "content" in item]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    embeddings = LocalEmbeddings()
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=str(DB_PATH))
    print(f"✅ Database created with {len(chunks)} chunks.")

# ===============================
# LOAD VECTOR DB
# ===============================
def get_vector_db():
    embeddings = LocalEmbeddings()
    db = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
    return db

# ===============================
# LLaMA QUERY
# ===============================
def query_llama(prompt):
    """Send prompt to LLaMA 3 via Ollama"""
    process = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    output, error = process.communicate(input=prompt)
    if error:
        print("⚠️ LLaMA Error:", error)
    return output

# ===============================
# MARINE FILTER
# ===============================
def is_marine_question(question: str) -> bool:
    """Allow all ocean-related topics, block unrelated ones."""
    q_lower = question.lower()

    ocean_terms = [
        "ocean", "marine", "sea", "coast", "shore", "beach", "reef", "aquatic",
        "plastic", "pollution", "fishing", "whale", "dolphin", "shark",
        "turtle", "seal", "krill", "plankton", "habitat", "ecosystem", "biodiversity",
        "maritime", "ship", "navigation", "currents", "waves", "tsunami", "coral"
    ]

    blocked_terms = [
        "president", "prime minister", "election", "football", "cricket",
        "movie", "actor", "politics", "stock market", "AI chatbot"
    ]

    if any(term in q_lower for term in blocked_terms):
        return False

    return any(term in q_lower for term in ocean_terms)
