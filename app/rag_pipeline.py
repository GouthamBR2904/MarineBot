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

    lines = [line.strip() for line in output.split("\n") if line.strip()]
    return lines[-1] if lines else output.strip()

# ===============================
# MARINE FILTER
# ===============================
def is_marine_question(question: str) -> bool:
    q_lower = question.lower()

    ocean_terms = [
        "ocean", "marine", "sea", "coast", "shore", "beach", "reef", "aquatic",
        "plastic", "pollution", "fishing", "whale", "dolphin", "shark",
        "turtle", "seal", "krill", "plankton", "habitat", "ecosystem", "biodiversity",
        "maritime", "ship", "navigation", "currents", "waves", "tsunami", "coral",
        "kelp", "mangrove", "algae", "seaweed", "seagrass", "ocean floor", "deep sea",
        "underwater", "marine biology", "marine life", "climate change ocean"
    ]

    non_marine_context = [
        "movie", "film", "actor", "actress", "director",
        "song", "music", "album", "band",
        "company", "corporation", "startup", "business",
        "software", "game", "app",
        "football", "cricket", "basketball", "team", "match", "tournament",
        "whale company", "big fish movie", "orca energy"
    ]

    blocked_terms = [
        "president", "prime minister", "election", "politics", "government",
        "stock", "market", "share", "investment", "bitcoin", "crypto",
        "ai chatbot", "chatgpt", "artificial intelligence"
    ]

    if any(term in q_lower for term in non_marine_context + blocked_terms):
        return False

    return any(term in q_lower for term in ocean_terms)

# ===============================
# RAG + LLaMA RESPONSE (Grounded in Dataset)
# ===============================
def get_rag_llama_response(prompt: str) -> str:
    try:
        db = get_vector_db()
        docs = db.similarity_search(prompt, k=3)
        context = " ".join([doc.page_content for doc in docs])

        final_prompt = (
            f"You are a marine biology expert. Use the following marine context to answer:\n\n"
            f"{context}\n\n"
            f"Question: {prompt}\n"
            f"Answer in 1–2 short sentences, max 30 words. "
            f"Only marine facts, ignore movies, shows, sports, or companies."
        )

        answer = query_llama(final_prompt)
        words = answer.split()
        if len(words) > 50:
            answer = " ".join(words[:50]) + "..."
        return answer.strip()

    except Exception:
        return "Sorry, I couldn't retrieve the answer."

# ===============================
# QUICK LOOKUP FOR UNITY OBJECTS
# ===============================
quick_lookup = {
    "shark": "Sharks are apex predators with cartilaginous skeletons, vital to marine ecosystems.",
    "coral": "Corals form reefs that support thousands of marine species.",
    "turtle": "Sea turtles are migratory reptiles essential to ocean health."
}

def get_quick_answer(object_name: str):
    return quick_lookup.get(object_name.lower())
