# config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str
    MURF_API_KEY: str
    GEMINI_API_KEY: str
    UPLOAD_DIR: str = "app/uploads"

    class Config:
        env_file = ".env"

settings = Settings()