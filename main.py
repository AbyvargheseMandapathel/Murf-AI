import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# AI Services
import assemblyai as aai
import google.generativeai as genai
from services.murf_service import MurfService
from services.assembly_service import AssemblyAIStreamingClient

# Load environment variables
load_dotenv()

MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([MURF_API_KEY, ASSEMBLYAI_API_KEY, GEMINI_API_KEY]):
    raise RuntimeError("❌ Missing one or more API keys in .env file")

# Configure APIs
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Define the persona for Mayavi
persona = """
You are Mayavi, a mischievous and helpful imp from a magical forest.
You protect the forest and its inhabitants from villains and dark wizards.
Your language should be a bit whimsical and magical, reflecting your nature.
You often refer to yourself as "this imp" or "Mayavi."
Always maintain your friendly but sly demeanor, and answer questions as if you were speaking to a friend you are protecting.
"""

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the frontend HTML file."""
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    aai_client = AssemblyAIStreamingClient(websocket, loop)
    murf_service = MurfService(websocket, MURF_API_KEY)

    try:
        while True:
            msg = await websocket.receive()

            # Handle audio stream
            if "bytes" in msg:
                aai_client.stream(msg["bytes"])

            # Handle text-based messages
            elif "text" in msg:
                data = json.loads(msg["text"])

                if data.get("event") == "transcript" and data.get("status") == "final":
                    # Notify client that AI is thinking
                    await websocket.send_json({"event": "status", "status": "thinking"})

                    # Get Gemini response (run in thread if async not supported)
                    model = genai.GenerativeModel("gemini-1.5-flash",)
                    response = await asyncio.to_thread(model.generate_content, data["text"])
                    bot_text = response.text or "I'm not sure."

                    # Send AI transcript back to frontend
                    await websocket.send_json({
                        "event": "transcript",
                        "text": bot_text,
                        "type": "bot"
                    })

                    # Convert response to speech
                    await murf_service.synthesize_speech(bot_text)

    except WebSocketDisconnect:
        pass
    finally:
        aai_client.close()
        await murf_service.close()