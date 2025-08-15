from fastapi import FastAPI, HTTPException, UploadFile, File, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import requests
import assemblyai as aai
from dotenv import load_dotenv
from typing import Dict, List
import re

# Load environment variables
load_dotenv()

# Initialize services
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
UPLOAD_DIR = os.path.join("app", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Models
class TextInput(BaseModel):
    text: str

class LLMQuery(BaseModel):
    text: str

# Application state
chat_histories: Dict[str, List[Dict[str, str]]] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Helper functions
def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to disk and return path"""
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = file.file.read()
        f.write(content)
    return file_location

def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using AssemblyAI"""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.status != aai.TranscriptStatus.completed:
        raise HTTPException(status_code=500, detail="Transcription failed")
    return transcript.text

def query_gemini(prompt: str) -> str:
    """Query Gemini LLM API"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)
    
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Unexpected Gemini response format")

def generate_murf_audio(text: str, voice_id: str = "en-US-natalie") -> str:
    """Generate TTS audio using Murf API"""
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {
        "accept": "application/json",
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "voice_id": voice_id,
        "text": text,
        "format": "mp3",
        "sampleRate": 44100
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)
    
    return response.json().get("audioFile")

# Routes
@app.get("/")
def read_index():
    return FileResponse(os.path.join("app/static", "index.html"))

@app.post("/generate-audio/")
def generate_audio(input: TextInput):
    try:
        audio_url = generate_murf_audio(input.text, "en-US-amara")
        return {"audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-echo/")
async def upload_echo(file: UploadFile = File(...)):
    try:
        file_location = save_uploaded_file(file)
        return JSONResponse({
            "filename": file.filename,
            "content_type": file.content_type,
            "size": os.path.getsize(file_location)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)
        transcript = transcribe_audio(file_path)
        return {"text": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/echo")
async def tts_echo(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)
        transcript = transcribe_audio(file_path)
        audio_url = generate_murf_audio(transcript)
        return {"audio_url": audio_url, "transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/query")
async def query_llm_audio(file: UploadFile = File(...)):
    try:
        # STT
        file_path = save_uploaded_file(file)
        user_text = transcribe_audio(file_path)
        
        # LLM
        llm_text = query_gemini(user_text)
        
        # TTS
        audio_url = generate_murf_audio(llm_text)
        
        return {
            "transcript": user_text,
            "llm_text": llm_text,
            "audio_url": audio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
chat_histories = {}


@app.post("/agent/chat/{session_id}")
async def agent_chat(session_id: str = Path(...), file: UploadFile = File(...)):
    try:
        # Initialize session if not exists
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        # Save file and verify
        file_path = save_uploaded_file(file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not saved properly")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Speech-to-text
        try:
            user_text = transcribe_audio(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Empty transcription")

        chat_histories[session_id].append({"role": "user", "text": user_text})

        # Prepare conversation context
        conversation = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}"
            for msg in chat_histories[session_id]
        )

        # LLM
        try:
            llm_text = query_gemini(conversation)

            # ✅ Clean unwanted text for voice output
            # Remove "Assistant:" at start
            if llm_text.lower().startswith("assistant:"):
                llm_text = llm_text.split(":", 1)[1].strip()

            # Remove any stage directions inside (), [], {}
            llm_text = re.sub(r'\([^)]*\)', '', llm_text)
            llm_text = re.sub(r'\[[^\]]*\]', '', llm_text)
            llm_text = re.sub(r'\{[^}]*\}', '', llm_text)

            # Collapse multiple spaces into one
            llm_text = re.sub(r'\s+', ' ', llm_text).strip()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

        chat_histories[session_id].append({"role": "assistant", "text": llm_text})

        # Text-to-speech
        try:
            audio_url = generate_murf_audio(llm_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

        return {
            "transcript": user_text,
            "llm_text": llm_text,
            "audio_url": audio_url,
            "chat_history": chat_histories[session_id]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))