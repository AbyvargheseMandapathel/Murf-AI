import os
import json
import base64
import logging
import uuid
from typing import Optional

import websockets
from murf import Murf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MURF_API_KEY = os.getenv('MURF_API_KEY')

if not MURF_API_KEY:
    raise RuntimeError("MURF_API_KEY not found in environment")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Non-streaming TTS (for fallback or one-off use)
async def murf_tts(text: str) -> dict:
    """
    Generate an audio file from text using Murf's REST API.
    Use this if WebSocket is unavailable.
    """
    try:
        if not text or not text.strip():
            return {"error": "Missing or empty text"}

        client = Murf(api_key=MURF_API_KEY)
        res = client.text_to_speech.generate(
            text=text,
            voice_id="en-US-Ken"
        )

        if res and res.audio_file:
            return {"audio_file": res.audio_file}
        else:
            return {"error": "No audio file generated"}
    except Exception as e:
        logger.error(f"Non-streaming TTS failed: {e}")
        return {"error": "TTS generation failed"}


# Streaming WebSocket TTS Service
WS_URL = "wss://api.murf.ai/v1/speech/stream-input"


class MurfService:
    def __init__(self, websocket, api_key: str):
        """
        Initialize MurfService with FastAPI WebSocket and API key.
        
        Args:
            websocket: FastAPI WebSocket instance to send audio chunks to frontend
            api_key: Murf API key
        """
        self.websocket = websocket
        self.api_key = api_key
        self.ws_url = WS_URL
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.current_context_id: Optional[str] = None

    async def connect(self):
        """Establish WebSocket connection to Murf TTS API."""
        try:
            # 🔥 Correct URL: use `api-key`, not `api_key`
            connect_url = (
                f"{self.ws_url}?api-key={self.api_key}"
                f"&sample_rate=44100"
                f"&channel_type=MONO"
                f"&format=WAV"
            )

            self.connection = await websockets.connect(
                connect_url,
                open_timeout=10,
                close_timeout=10
            )
            logger.info("✅ Connected to Murf.ai TTS WebSocket")

            # Optional: Set voice config
            voice_config = {
                "voice_config": {
                    "voiceId": "en-US-Ken",
                    "style": "Conversational",
                    "rate": 0,
                    "pitch": 0,
                    "variation": 1
                }
            }
            await self.connection.send(json.dumps(voice_config))
            logger.info("🎤 Voice configuration sent")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Murf.ai: {e}")
            await self.websocket.send_json({
                "status": "error",
                "message": "Could not connect to TTS service"
            })
            raise

    async def synthesize_speech(self, text: str) -> str:
        """
        Stream speech synthesis from Murf and send audio chunks to frontend.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            str: Base64-encoded complete audio (optional)
        """
        if not text.strip():
            await self.websocket.send_json({
                "status": "error",
                "message": "Empty text provided"
            })
            return ""

        # Generate new Context ID for this turn
        self.current_context_id = f"ctx_{int(__import__('time').time())}_{uuid.uuid4().hex[:6]}"
        logger.info(f"🆕 New context_id: {self.current_context_id}")

        if not self.connection:
            await self.connect()

        try:
            # 🔥 Send TTS request with context_id and end=True
            tts_request = {
                "text": text,
                "context_id": self.current_context_id,
                "end": True  # End of turn
            }
            await self.connection.send(json.dumps(tts_request))
            logger.info(f"📤 Sent TTS request: {text[:50]}...")

            # Accumulate audio for return (optional)
            complete_audio_b64 = ""
            audio_chunks = []
            first_chunk = True

            while True:
                try:
                    response = await self.connection.recv()
                    data = json.loads(response)

                    # Match context
                    if data.get("context_id") != self.current_context_id:
                        continue

                    if "audio" in data:
                        audio_b64 = data["audio"]
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        stripped_b64 = base64.b64encode(audio_bytes).decode("utf-8")


                        # 🔧 Strip 44-byte WAV header only from first chunk
                        # if first_chunk and len(audio_bytes) > 44:
                        #     audio_bytes = audio_bytes[44:]
                        #     first_chunk = False

                        # Re-encode stripped bytes
                        stripped_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        audio_chunks.append(stripped_b64)
                        complete_audio_b64 += stripped_b64

                        # 🔊 Stream to frontend in real time
                        await self.websocket.send_json({
                            "event": "audio_chunk",
                            "audio_base64": stripped_b64,
                            "context_id": self.current_context_id,
                            "chunk_index": len(audio_chunks),
                            "is_final": False
                        })
                        logger.debug(f"🔊 Sent audio chunk #{len(audio_chunks)}")

                    if data.get("final"):
                        logger.info(f"✅ Final audio received for {self.current_context_id}")

                        # Send final signal
                        await self.websocket.send_json({
                            "event": "audio_chunk",
                            "context_id": self.current_context_id,
                            "is_final": True
                        })

                        break

                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from Murf")
                    continue
                except Exception as e:
                    logger.error(f"Error receiving audio: {e}")
                    break

            return complete_audio_b64

        except Exception as e:
            logger.error(f"❌ Error in speech synthesis: {e}")
            await self.websocket.send_json({
                "event": "error",
                "message": f"TTS Error: {str(e)}"
            })
            return ""

    async def close(self):
        """Close the Murf WebSocket connection."""
        if self.connection:
            try:
                await self.connection.close()
                logger.info("🛑 Murf.ai WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing Murf connection: {e}")
            finally:
                self.connection = None
        else:
            logger.info("📭 Murf connection was already closed")