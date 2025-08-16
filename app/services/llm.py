# services/llm.py
import requests
import re
from app.config import settings
from app.logger import logger

def clean_llm_response(text: str) -> str:
    """Clean response for TTS: remove prefixes and stage directions."""
    text = re.sub(r'(?i)^assistant\s*:\s*', '', text)
    text = re.sub(r'\([^)]*\)', '', text)  # Remove (...)
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [...]
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove {...}
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def query_gemini(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": settings.GEMINI_API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned_text = clean_llm_response(text)
        logger.info("LLM query successful.")
        return cleaned_text
    except (KeyError, IndexError) as e:
        logger.error(f"Malformed LLM response: {e}")
        raise RuntimeError("Invalid response format from Gemini")
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise RuntimeError(f"LLM error: {str(e)}")