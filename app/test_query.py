from rag_pipeline import get_vector_db, query_llama

# Load database
db = get_vector_db()

# Ask a sample marine question
query = "What is microplastic pollution?"
docs = db.similarity_search(query, k=3)

# Prepare context for LLaMA
context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"Answer the question based only on the following marine data:\n\n{context}\n\nQuestion: {query}\nAnswer:"

# Query LLaMA
response = query_llama(prompt)
print("LLaMA Response:\n", response)
