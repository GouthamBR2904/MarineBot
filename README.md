# MarineBot – AI Voice Assistant for VR Marine Life Education

MarineBot is an AI-driven voice assistant built for a VR marine life education experience.  
It enables players to ask questions about marine species and ecosystems using voice, and receive short, game-friendly answers in real time.

## Features
- Real-time speech recognition for user queries  
- Retrieval-Augmented Generation (RAG) pipeline with LLaMA  
- Unity integration for object-based and voice-based interactions  
- Short, context-aware responses designed for VR gameplay  
- Focused knowledge base limited to marine-related topics  

## Tech Stack
- Python 3.10  
- LangChain, ChromaDB, LLaMA  
- Coqui TTS for offline speech synthesis  
- Unity for VR integration  

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/GouthamBR2904/MarineBot.git
cd MarineBot

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the assistant locally:
```bash
python app/main.py
```

Integrate with Unity to connect VR interactions to the voice bot.

## Example Outputs
- Ask: *"What is a clownfish?"*  
  → Response: *"A clownfish is a small orange-and-white reef fish that lives in sea anemones."*  
- Ask: *"How does coral bleaching happen?"*  
  → Response: *"It occurs when corals expel algae due to stress, often caused by rising sea temperatures."*  
