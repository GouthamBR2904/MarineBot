import speech_recognition as sr
import pyttsx3
from rag_pipeline import get_vector_db, query_llama, is_marine_question
from sentence_transformers import SentenceTransformer, util
import re

# ===============================
# Cleaner for TTS & Display
# ===============================
def clean_for_tts_and_display(text):
    text = re.sub(r'[*_`#>]', '', text)           # Remove markdown
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces/newlines
    return text

# ===============================
# Initialize DB & Models
# ===============================
db = get_vector_db()
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)   # Adjust speaking speed
tts_engine.setProperty('volume', 1.0) # Max volume

# ===============================
# Voice Test Logic
# ===============================
recognizer = sr.Recognizer()

print("🎤 Speak now (Marine-related question)...")
with sr.Microphone() as source:
    audio = recognizer.listen(source)

try:
    question = recognizer.recognize_google(audio)
    print(f"🗣 You said: {question}")
except sr.UnknownValueError:
    print("⚠️ Could not understand audio.")
    exit()
except sr.RequestError:
    print("⚠️ STT service unavailable.")
    exit()

# Marine filter
if not is_marine_question(question):
    response_text = "I can only answer marine-related questions."
else:
    # Search Chroma
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Similarity + phrase check
    query_embedding = similarity_model.encode(question, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, context_embedding).item()

    subject_phrase = question.lower()
    subject_in_context = subject_phrase in context.lower()

    # Decide dataset vs LLaMA fallback
    if similarity_score < 0.5 or not subject_in_context:
        response_text = query_llama(f"You are an expert marine biologist. Answer the following question in detail:\nQuestion: {question}")
    else:
        prompt = f"Answer based only on the following marine data:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        response_text = query_llama(prompt)

# Clean response
clean_response = clean_for_tts_and_display(response_text)

# Print + Speak immediately
print(f"\n🤖 Bot: {clean_response}")
tts_engine.say(clean_response)
tts_engine.runAndWait()
