# app/services/tts.py

import requests
import asyncio
import json
import websockets
from typing import AsyncGenerator, List
from app.config import settings
from app.logger import logger

# === Existing REST TTS (Keep for fallback) ===
def generate_murf_audio(text: str, voice_id: str = "en-US-natalie") -> str:
    """
    Legacy REST API call to generate audio and get URL.
    """
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {
        "accept": "application/json",
        "api-key": settings.MURF_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "voice_id": voice_id,
        "text": text,
        "format": "mp3",
        "sampleRate": 44100
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        audio_url = response.json().get("audioFile")
        if not audio_url:
            raise ValueError("No audioFile in response")
        logger.info("TTS audio generated via REST.")
        return audio_url
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"TTS failed: {str(e)}")


# === NEW: WebSocket TTS with Base64 Streaming ===

MURF_WS_URL = "wss://api.murf.ai/v1/tts/stream"
STATIC_CONTEXT_ID = "day20-static-context-001"  # Reuse context to avoid limits

async def stream_murf_tts_websocket(
    text: str,
    voice_id: str = "en-US-amara"
) -> AsyncGenerator[str, None]:
    """
    Streams text to Murf via WebSocket using query params for auth.
    Prints base64 audio to console.
    """
    # ✅ Correct URL with api-key in query params
    ws_url = (
        f"wss://api.murf.ai/v1/speech/stream-input"
        f"?api-key={settings.MURF_API_KEY}"
        f"&sample_rate=44100&channel_type=MONO&format=WAV"
    )

    try:
        # ✅ No extra_headers needed
        async with websockets.connect(ws_url) as ws:
            print("🟢 Connected to Murf WebSocket (TTS)")

            # 1. Send voice config with context_id
            await ws.send(json.dumps({
                "voice_config": {
                    "voiceId": voice_id,
                    "style": "Conversational",
                    "rate": 0,
                    "pitch": 0,
                    "variation": 1
                },
                "context_id": STATIC_CONTEXT_ID
            }))
            print("📨 Sent voice config to Murf")

            # 2. Send text
            await ws.send(json.dumps({
                "text": text,
                "end": True,
                "context_id": STATIC_CONTEXT_ID
            }))
            print(f"📨 Sent text to Murf: '{text[:50]}...'")

            # 3. Receive audio chunks
            while True:
                try:
                    response = await ws.recv()
                    data = json.loads(response)

                    if "audio" in data:
                        base64_audio = data["audio"]
                        # 🔥 PRINT BASE64 TO CONSOLE (REQUIRED)
                        print("🎧 Received base64 audio chunk:")
                        print(base64_audio)
                        yield base64_audio

                    if data.get("final") or data.get("status") == "complete":
                        print("✅ Murf TTS stream completed.")
                        break

                    if data.get("error"):
                        print(f"❌ Murf error: {data['error']}")
                        break

                except Exception as e:
                    print(f"⚠️ Error receiving from Murf: {e}")
                    break

    except Exception as e:
        print(f"❌ Failed to connect to Murf WebSocket: {e}")
        raise RuntimeError(f"Could not connect to Murf WebSocket: {str(e)}")