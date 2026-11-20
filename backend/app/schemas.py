from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class DetectionItem(BaseModel):
    label: str = "pothole"
    confidence: float = Field(ge=0, le=1)
    severity: str = "unclassified"
    bbox: list[float] | None = None
    class_id: int | None = None

class DetectionResponse(BaseModel):
    model: str
    inference_time_ms: float
    count: int
    annotated_image: str
    detections: list[DetectionItem]

class PotholeCreate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    severity: str = "unclassified"
    status: str = "reported"
    confidence: float | None = Field(default=None, ge=0, le=1)
    description: str = "AI-assisted road hazard report"
    detections: list[dict[str, Any]] = []
    source: str = "web-inspection"

class PotholeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    latitude: float
    longitude: float
    severity: str
    status: str
    confidence: float | None
    description: str
    detections: list[dict[str, Any]]
    source: str
    created_at: datetime
