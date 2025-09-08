# 🐠 MarineBot: Interactive AI Assistant for Ocean Education (VR Project)

## 📦 Project Structure
```
MarineBot/
├── app/                # FastAPI backend with RAG + LLaMA + voice support
├── data/               # Contains dataset (merged_marine_info.json)
├── vector_store/       # Chroma DB vector storage
├── MarineBotDemo1/     # Unity project (3D VR front-end)
├── requirements.txt    # Python dependencies
├── README.md           # This file
```

---

## 🔧 Prerequisites

### For Python Backend
- Python 3.10+
- `pip` package manager
- [Ollama](https://ollama.com/) with `llama3` model installed locally

### For Unity Frontend
- Unity 2022.3 LTS (Long Term Support)
- Windows Build Support (included in Unity Hub)
- Microphone permission enabled

---

## 🧠 How to Run the Backend

1. Open terminal in `marine_bot` folder.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
5. Server will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🕹️ How to Run the Unity Project

1. Open Unity Hub.
2. Click **Add Project** and select the `MarineBotDemo1/` folder.
3. Open the project.
4. Load the `MainScene.unity` inside `Scenes/` folder.
5. Click **Play** ▶️ to run the game.
6. Interact using:
   - Microphone input (ask questions)
   - Mouse click on marine creatures (triggers object-based query)

**Note:** Make sure the backend API is running before clicking Play.

---

## 🗣️ Voice Interaction

- Unity records audio input → sends it to `/voice_query` endpoint
- FastAPI transcribes → uses RAG (dataset) + LLaMA fallback
- Short answer returned and read aloud using Text-to-Speech

---

## ⚠️ Notes

- Only marine-related questions are allowed.
- If the bot doesn't find the answer in the dataset, it uses LLaMA to respond.
- Designed to be easily portable to Meta Quest later.

---
