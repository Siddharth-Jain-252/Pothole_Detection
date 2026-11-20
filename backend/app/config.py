from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RoadSight Pothole API"
    database_url: str = "postgresql+psycopg://roadsight:roadsight_dev_password@localhost:5432/roadsight"
    model_path: str = str(ROOT_DIR / "models" / "yolov8_pothole_best.pt")
    model_device: str = "auto"  # auto, cpu, 0
    model_confidence: float = 0.25
    model_iou: float = 0.45
    max_upload_mb: int = 15
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    api_key: str = ""  # optional simple bearer token; leave empty for local development

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

settings = Settings()
