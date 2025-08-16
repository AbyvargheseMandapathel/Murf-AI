# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class TextInput(BaseModel):
    text: str

class LLMQuery(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    text: str

class AgentChatResponse(BaseModel):
    transcript: str
    llm_text: str
    audio_url: str
    chat_history: List[ChatMessage]

class UploadEchoResponse(BaseModel):
    filename: str
    content_type: str
    size: int

class TranscribeResponse(BaseModel):
    text: str

class TtsEchoResponse(BaseModel):
    transcript: str
    audio_url: str