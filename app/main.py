from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from rag_pipeline import get_vector_db, query_llama, is_marine_question
import speech_recognition as sr
import pyttsx3
import tempfile
from sentence_transformers import SentenceTransformer, util
import re

app = FastAPI()

# Load Chroma DB
db = get_vector_db()

# Initialize TTS (pyttsx3)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)   # Adjust speaking rate
tts_engine.setProperty('volume', 1.0) # Max volume
tts_engine.setProperty('voice', tts_engine.getProperty('voices')[0].id)  # Default voice

# Load similarity model
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Helper to Clean Output for TTS & Display
# ===============================
def clean_for_tts_and_display(text):
    # Remove markdown formatting symbols
    text = re.sub(r'[*_`#>]', '', text)
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================
# TEXT QUERY ENDPOINT
# ===============================
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_bot(request: QueryRequest):
    question = request.question

    # Marine-only filter
    if not is_marine_question(question):
        return {
            "status": "ignored",
            "question": question,
            "answer": "I can only answer marine-related questions.",
            "marine_related": False
        }

    # Search Chroma
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Similarity + exact phrase check
    query_embedding = similarity_model.encode(question, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, context_embedding).item()

    subject_phrase = question.lower()
    subject_in_context = subject_phrase in context.lower()

    # Decide dataset vs LLaMA fallback
    if similarity_score < 0.5 or not subject_in_context:
        answer = query_llama(f"You are an expert marine biologist. Answer the following question in detail:\nQuestion: {question}")
    else:
        prompt = f"Answer based only on the following marine data:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = query_llama(prompt)

    clean_answer = clean_for_tts_and_display(answer)

    return {
        "status": "success",
        "question": question,
        "answer": clean_answer,
        "marine_related": True
    }

# ===============================
# VOICE QUERY ENDPOINT
# ===============================
@app.post("/voice_query")
async def voice_query(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # STT
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)
        try:
            question = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return {"status": "error", "message": "Could not understand audio."}
        except sr.RequestError:
            return {"status": "error", "message": "STT service unavailable."}

    # Marine-only filter
    if not is_marine_question(question):
        response_text = "I can only answer marine-related questions."
    else:
        # Search Chroma
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Similarity + exact phrase check
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

    clean_response = clean_for_tts_and_display(response_text)

    # TTS
    audio_path = "response.wav"
    tts_engine.save_to_file(clean_response, audio_path)
    tts_engine.runAndWait()

    return {
        "status": "success",
        "question": question,
        "answer": clean_response,
        "marine_related": is_marine_question(question),
        "audio_file": audio_path
    }
