import json
import os
import time
import uuid
import asyncio
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import websockets

# AI Services
import assemblyai as aai
from services.murf_service import MurfService
import google.generativeai as genai
from services.assembly_service import AssemblyAIStreamingClient

# Load environment variables
load_dotenv()

MURF_API_KEY = os.getenv('MURF_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not all([MURF_API_KEY, ASSEMBLYAI_API_KEY, GEMINI_API_KEY]):
    raise RuntimeError("❌ Missing one or more API keys in .env file")

# Configure APIs
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    aai_client = AssemblyAIStreamingClient(websocket, loop)
    murf_service = MurfService(websocket, MURF_API_KEY)

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg:
                aai_client.stream(msg["bytes"])
            elif "text" in msg:
                data = json.loads(msg["text"])
                if data.get("event") == "transcript" and data.get("status") == "final":
                    # Show thinking
                    await websocket.send_json({"event": "status", "status": "thinking"})

                    # Get LLM response
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = await model.generate_content_async(data["text"])
                    bot_text = response.text or "I'm not sure."

                    # Send transcript
                    await websocket.send_json({
                        "event": "transcript",
                        "text": bot_text,
                        "type": "bot"
                    })

                    # Start TTS
                    await murf_service.synthesize_speech(bot_text)

    except WebSocketDisconnect:
        pass
    finally:
        aai_client.close()
        await murf_service.close()