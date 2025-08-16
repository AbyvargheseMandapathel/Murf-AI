# services/stt.py
from app.config import settings
import assemblyai as aai
from app.logger import logger

def transcribe_audio(file_path: str) -> str:
    aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)

    if transcript.status != aai.TranscriptStatus.completed:
        error_msg = f"Transcription failed: {transcript.status}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("Transcription completed.")
    return transcript.text