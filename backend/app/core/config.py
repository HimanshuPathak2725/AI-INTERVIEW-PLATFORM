from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Interview Platform"
    debug: bool = True
    database_url: str = "sqlite:///./interview_platform.db"
    chroma_persist_dir: str = "./chroma_db"
    GROQ_API_KEY: str = ""
    JINA_API_KEY: str = ""
    cors_origins: str = Field(default="http://localhost:5173")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def get_cors_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# settings = Settings()
