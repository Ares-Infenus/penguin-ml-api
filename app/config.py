"""Application configuration using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    model_path: str = "model/pipeline.joblib"
    metadata_path: str = "model/metadata.json"
    log_level: str = "INFO"

    @property
    def model_path_resolved(self) -> Path:
        return Path(self.model_path)

    @property
    def metadata_path_resolved(self) -> Path:
        return Path(self.metadata_path)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
