# services/tts.py
import requests
from app.config import settings
from app.logger import logger

def generate_murf_audio(text: str, voice_id: str = "en-US-natalie") -> str:
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
        logger.info("TTS audio generated.")
        return audio_url
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"TTS failed: {str(e)}")