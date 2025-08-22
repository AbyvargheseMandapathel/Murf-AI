from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str
    MURF_API_KEY: str
    GEMINI_API_KEY: str
    UPLOAD_DIR: str = "app/uploads"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()