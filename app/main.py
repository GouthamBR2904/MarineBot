from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import tempfile
import speech_recognition as sr
import pyttsx3
import os

from fastapi.staticfiles import StaticFiles
from rag_pipeline import get_rag_llama_response, is_marine_question, get_quick_answer

app = FastAPI()

# ===============================
# STATIC FILE SETUP
# ===============================
if not os.path.exists("static/audio"):
    os.makedirs("static/audio")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===============================
# MODELS
# ===============================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    status: str
    question: str
    answer: str
    marine_related: bool
    reason: str = None

# ===============================
# BOT RESPONSE
# ===============================
def get_bot_response(question: str) -> str:
    object_name = question.replace("What is", "").replace("?", "").strip()

    quick_ans = get_quick_answer(object_name)
    if quick_ans:
        return quick_ans

    short_prompt = (
        f"You are a marine biology expert. "
        f"Answer ONLY about the marine animal, plant, or ecosystem named '{object_name}'. "
        f"Ignore movies, shows, sports teams, or companies. "
        f"Give a short factual marine-related answer in 1 sentence."
    )

    return get_rag_llama_response(short_prompt)

# ===============================
# TEXT QUERY
# ===============================
@app.post("/query", response_model=QueryResponse)
async def query_bot(request: QueryRequest):
    question = request.question.strip()

    if not is_marine_question(question):
        return QueryResponse(
            status="ignored",
            question=question,
            answer="I can only answer marine-related questions.",
            marine_related=False,
            reason="The question does not appear to be about the marine ecosystem."
        )

    answer = get_bot_response(question)
    return QueryResponse(
        status="success",
        question=question,
        answer=answer,
        marine_related=True
    )

# ===============================
# VOICE QUERY
# ===============================
@app.post("/voice_query")
async def voice_query(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        question = recognizer.recognize_google(audio_data)
    except Exception:
        question = "Could not understand audio."

    if not is_marine_question(question):
        return {
            "status": "ignored",
            "question": question,
            "answer": "I can only answer marine-related questions.",
            "audio_file": None,
            "marine_related": False,
            "reason": "The question does not appear to be about the marine ecosystem."
        }

    answer = get_bot_response(question)

    tts_filename = f"response_{os.path.basename(temp_audio_path)}.wav"
    tts_path = os.path.join("static/audio", tts_filename)

    engine = pyttsx3.init()
    engine.save_to_file(answer, tts_path)
    engine.runAndWait()

    audio_url = f"http://127.0.0.1:8000/static/audio/{tts_filename}"

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "audio_file": audio_url,
        "marine_related": True
    }

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
