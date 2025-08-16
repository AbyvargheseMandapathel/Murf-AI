# main.py
from fastapi import FastAPI, File, UploadFile, Path, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import schemas
from app.config import settings
from app.services import stt, tts, llm
from app.utils import save_uploaded_file
from app.logger import logger

import os

# Application state
chat_histories: dict = {}

app = FastAPI(title="Voice Agent API", version="1.0")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Health check
@app.get("/")
def read_index():
    return FileResponse("app/static/index.html")

# Endpoints

@app.post("/generate-audio/", response_model=BaseModel)
def generate_audio(input: schemas.TextInput):
    try:
        audio_url = tts.generate_murf_audio(input.text, voice_id="en-US-amara")
        return {"audio_url": audio_url}
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-echo/", response_model=schemas.UploadEchoResponse)
async def upload_echo(file: UploadFile = File(...)):
    try:
        file_location = save_uploaded_file(file)
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": os.path.getsize(file_location)
        }
    except Exception as e:
        logger.error(f"File upload echo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/file", response_model=schemas.TranscribeResponse)
async def transcribe_file(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)
        transcript = stt.transcribe_audio(file_path)
        return {"text": transcript}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/echo", response_model=schemas.TtsEchoResponse)
async def tts_echo(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)
        transcript = stt.transcribe_audio(file_path)
        audio_url = tts.generate_murf_audio(transcript)
        return {"transcript": transcript, "audio_url": audio_url}
    except Exception as e:
        logger.error(f"TTS echo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/query")
async def query_llm_audio(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)
        user_text = stt.transcribe_audio(file_path)
        llm_text = llm.query_gemini(user_text)
        audio_url = tts.generate_murf_audio(llm_text)
        return {
            "transcript": user_text,
            "llm_text": llm_text,
            "audio_url": audio_url
        }
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/chat/{session_id}", response_model=schemas.AgentChatResponse)
async def agent_chat(session_id: str = Path(...), file: UploadFile = File(...)):
    try:
        # Initialize session
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        # Save file
        file_path = save_uploaded_file(file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not saved.")
        if os.path.getsize(file_path) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        # STT
        user_text = stt.transcribe_audio(file_path)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Empty transcription.")
        chat_histories[session_id].append({"role": "user", "text": user_text})

        # Build conversation history
        conversation = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}"
            for msg in chat_histories[session_id]
        )

        # LLM
        llm_text = llm.query_gemini(conversation)
        chat_histories[session_id].append({"role": "assistant", "text": llm_text})

        # TTS
        audio_url = tts.generate_murf_audio(llm_text)

        return {
            "transcript": user_text,
            "llm_text": llm_text,
            "audio_url": audio_url,
            "chat_history": chat_histories[session_id]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")