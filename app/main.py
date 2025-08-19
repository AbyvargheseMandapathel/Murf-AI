# main.py - Day 18: Turn Detection with AssemblyAI
import asyncio
import time
from fastapi import FastAPI, File, UploadFile, Path, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import schemas
from app.config import settings
from app.services import stt, tts, llm
from app.utils import save_uploaded_file
from app.logger import logger

import os
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

# Set the AssemblyAI API key globally
aai.settings.api_key = settings.ASSEMBLYAI_API_KEY

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

# Your existing endpoints...
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

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# ----------------------------
# DAY 18: Turn Detection with AssemblyAI V3 - Final Transcripts Only
# ----------------------------

@app.websocket("/ws/transcribe/{session_id}")
async def websocket_transcribe_handler(websocket: WebSocket, session_id: str):
    """
    Day 18: WebSocket handler for real-time transcription with turn detection using AssemblyAI V3 SDK.
    Only sends final transcriptions when end_of_turn is detected.
    """
    await websocket.accept()
    print(f"\n🎤 === DAY 18: WebSocket Connected ===")
    print(f"📱 Session ID: {session_id}")
    
    client = None
    # Track WebSocket connection state
    websocket_closed = False
    is_connected = True
    
    # Get the main thread's event loop
    main_loop = asyncio.get_running_loop()

    try:
        # Event Handlers
        def on_begin(self, event: BeginEvent):
            print(f"✅ AssemblyAI Session started: {event.id}")

        def on_turn(self, event: TurnEvent):
            try:
                # Only process transcripts that have content
                if event.transcript.strip():
                    # Only send message when we detect the end of a turn
                    if event.end_of_turn:
                        print(f"\n🎯 FINAL TRANSCRIPT: {event.transcript}")
                        message = f"✅ {event.transcript}"
                        
                        # Only send if WebSocket is still open
                        if not websocket_closed:
                            main_loop.create_task(send_safe(websocket, message))
                        else:
                            print("ℹ️ WebSocket closed, not sending transcript.")
                    
                    # Ignore partial transcriptions - don't send anything
                    # This implements the "only show final" requirement
                    
            except Exception as e:
                print(f"❌ Error in on_turn: {e}")

        def on_terminated(self, event: TerminationEvent):
            print(f"ℹ️ Session terminated: {event.audio_duration_seconds} seconds processed")

        def on_error(self, error: StreamingError):
            error_msg = f"❌ Streaming Error: {error}"
            print(error_msg)
            # Only send if WebSocket is still open
            if not websocket_closed:
                main_loop.create_task(send_safe(websocket, error_msg))

        async def send_safe(ws, message):
            """Safely send message to websocket."""
            try:
                await ws.send_text(message)
            except Exception as e:
                # This exception is expected if the WebSocket is closed
                # We check 'websocket_closed' before sending, so we can ignore this
                if "after sending 'websocket.close'" not in str(e):
                    print(f"⚠️ Unexpected send error: {e}")
                pass

        # Initialize and Connect to AssemblyAI
        print("🔄 Initializing AssemblyAI V3 StreamingClient...")
        client = StreamingClient(
            StreamingClientOptions(
                api_key=settings.ASSEMBLYAI_API_KEY,
                api_host="streaming.assemblyai.com",
            )
        )
        
        # Register event handlers
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)
        
        print("🔗 Connecting to AssemblyAI...")
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                end_of_turn_confidence_threshold=0.7,
                min_end_of_turn_silence_when_confident=160,
                max_turn_silence=2400
            )
        )
        print("✅ Connected to AssemblyAI successfully!")
        
        await websocket.send_text("🎤 Ready to transcribe! Start speaking...")

        # Main Audio Processing Loop
        chunk_count = 0
        while is_connected:
            try:
                audio_bytes = await websocket.receive_bytes()
                chunk_count += 1
                
                if chunk_count % 50 == 0:
                    print(f"📊 Processed {chunk_count} audio chunks")
                
                # Stream audio to AssemblyAI
                client.stream(audio_bytes)
                
            except WebSocketDisconnect:
                print(f"🔌 Client disconnected: {session_id}")
                is_connected = False
                websocket_closed = True
                break
                
            except Exception as e:
                print(f"❌ Error in audio loop: {e}")
                is_connected = False
                websocket_closed = True
                break
                
    except Exception as e:
        error_msg = f"❌ Setup error: {e}"
        print(error_msg)
        is_connected = False
        websocket_closed = True
        try:
            await websocket.send_text(error_msg)
        except:
            pass
            
    finally:
        print("🧹 Cleaning up...")
        is_connected = False
        
        if client:
            try:
                client.disconnect(terminate=True)
                print("✅ AssemblyAI client disconnected")
            except Exception as e:
                print(f"ℹ️ Client disconnect: {e}")
        
        print("=== DAY 18: Session Complete ===\n")