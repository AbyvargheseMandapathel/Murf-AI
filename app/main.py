# main.py - Day 19: Real-Time Voice Agent with Turn Detection & LLM Streaming
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

# Set AssemblyAI API key
aai.settings.api_key = settings.ASSEMBLYAI_API_KEY

# Application state
chat_histories: dict = {}

app = FastAPI(title="Voice Agent API", version="1.0")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
def read_index():
    return FileResponse("app/static/index.html")


# === Existing Endpoints (Unchanged) ===
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
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        file_path = save_uploaded_file(file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not saved.")
        if os.path.getsize(file_path) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        user_text = stt.transcribe_audio(file_path)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Empty transcription.")

        chat_histories[session_id].append({"role": "user", "text": user_text})

        conversation = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}"
            for msg in chat_histories[session_id]
        )

        llm_text = llm.query_gemini(conversation)
        chat_histories[session_id].append({"role": "assistant", "text": llm_text})

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
# 🌐 WebSocket: Real-Time Transcription + LLM Streaming
# ----------------------------
@app.websocket("/ws/transcribe/{session_id}")
async def websocket_transcribe_handler(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time transcription with turn detection.
    Streams audio to AssemblyAI, detects full turns, and streams LLM response.
    """
    await websocket.accept()
    print(f"\n🎤 === DAY 19: WebSocket Connected ===")
    print(f"📱 Session ID: {session_id}")

    client = None
    websocket_closed = False
    is_connected = True
    main_loop = asyncio.get_running_loop()

    # Safe WebSocket sender
    async def send_safe(message: str):
        if websocket_closed:
            return
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"❌ Failed to send to WebSocket: {e}")
            # nonlocal websocket_closed
            # websocket_closed = True

    # ✅ Event Handlers: Now accept (client, event)
    def on_begin(client, event: BeginEvent):
        print(f"✅ AssemblyAI Session started: {event.id}")
        main_loop.create_task(send_safe("✅ Session started. Speak now..."))

    def on_turn(client, event: TurnEvent):
        if not event.transcript.strip():
            return
        if not event.end_of_turn:
            return  # Only process final turns

        # Prevent duplicate final transcripts
        if hasattr(client, 'last_final_transcript'):
            if client.last_final_transcript == event.transcript:
                return  # Skip if already processed
        client.last_final_transcript = event.transcript

        print(f"\n🎯 FINAL TRANSCRIPT: {event.transcript}")
        main_loop.create_task(send_safe(f"✅ You: {event.transcript}"))

        # --- STREAM LLM RESPONSE ---
        print("\n🚀 Streaming LLM response...\n" + "-" * 50)
        try:
            llm_stream = llm.query_gemini_stream(event.transcript)
            full_response = ""

            for chunk in llm_stream:
                text_chunk = chunk.text
                full_response += text_chunk
                print(f"💬 {text_chunk}", end="", flush=True)

            print("\n" + "-" * 50)
            print("✅ LLM Response Complete")
            main_loop.create_task(send_safe(f"🤖 {full_response}"))

            # --- 🔊 SEND TO MURF VIA WEBSOCKET ---
            async def send_to_murf():
                try:
                    from app.services.tts import stream_murf_tts_websocket
                    async for _ in stream_murf_tts_websocket(full_response, voice_id="en-US-amara"):
                        pass
                except Exception as e:
                    error_msg = f"❌ Murf TTS failed: {e}"
                    print(error_msg)
                    await send_safe(error_msg)

            asyncio.run_coroutine_threadsafe(send_to_murf(), main_loop)

        except Exception as e:
            error_msg = f"❌ LLM streaming error: {e}"
            print(error_msg)
            main_loop.create_task(send_safe(error_msg))

    def on_terminated(client, event: TerminationEvent):
        print(f"ℹ️ Session terminated: {event.audio_duration_seconds:.2f}s")
        if not websocket_closed:
            main_loop.create_task(send_safe("⏹️ Session ended."))

    def on_error(client, error: StreamingError):
        error_msg = f"❌ Streaming Error: {error}"
        print(error_msg)
        if not websocket_closed:
            main_loop.create_task(send_safe(error_msg))

    try:
        print("🔄 Initializing AssemblyAI V3 StreamingClient...")
        client = StreamingClient(
            StreamingClientOptions(
                api_key=settings.ASSEMBLYAI_API_KEY,
                api_host="streaming.assemblyai.com",
            )
        )

        # Attach event handlers
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
                max_turn_silence=2400,
            )
        )
        print("✅ Connected to AssemblyAI successfully!")

        await send_safe("🎤 Ready! Start speaking...")

        # Audio receiving loop
        while is_connected and not websocket_closed:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    if client:
                        client.stream(audio_bytes)
                elif "type" in data and data["type"] == "websocket.disconnect":
                    break
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"⚠️ Audio stream error: {e}")
                break

    except Exception as e:
        error_msg = f"❌ Setup error: {e}"
        print(error_msg)
        is_connected = False
        websocket_closed = True
        try:
            await send_safe(error_msg)
        except:
            pass

    finally:
        print("🧹 Cleaning up...")
        is_connected = False
        websocket_closed = True

        if client:
            try:
                client.disconnect(terminate=True)
                print("✅ AssemblyAI client disconnected")
            except Exception as e:
                print(f"ℹ️ Cleanup error: {e}")