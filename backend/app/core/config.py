import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App configs
    PROJECT_NAME: str = "AI Interview Platform"
    API_V1_STR: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./interview_platform.db"
    
    # LLM & Vector DB Keys
    OPENAI_API_KEY: str = ""
    JINA_API_KEY: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.getcwd(), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()