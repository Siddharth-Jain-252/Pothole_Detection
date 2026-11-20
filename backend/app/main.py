import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from ultralytics import YOLO

from app.config import settings
from app.database import engine, init_db
from app.schemas import DetectionItem, DetectionResponse, PotholeCreate
from app.security import verify_api_key

MODEL: YOLO | None = None

def load_model() -> YOLO:
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Model weights not found: {model_path}. "
            "Copy yolov8_pothole_best.pt into the models/ directory or set MODEL_PATH."
        )
    return YOLO(str(model_path))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    init_db()
    MODEL = load_model()
    yield
    MODEL = None

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="YOLO pothole inference and PostGIS-powered nearby hazard reports.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "database": "configured",
    }

@app.post("/detect", response_model=DetectionResponse, dependencies=[Depends(verify_api_key)])
async def detect_image(file: UploadFile = File(...)):
    """Run the trained YOLO model on an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a valid image file")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB")
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Detection model is not loaded")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    started = time.perf_counter()
    try:
        def predict():
            device = settings.model_device
            kwargs = {
                "source": image,
                "conf": settings.model_confidence,
                "iou": settings.model_iou,
                "verbose": False,
            }
            if device != "auto":
                kwargs["device"] = device
            return MODEL.predict(**kwargs)[0]

        result = await run_in_threadpool(predict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    inference_ms = (time.perf_counter() - started) * 1000
    names = MODEL.names
    detections: list[DetectionItem] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else 0
            confidence = float(box.conf.item()) if box.conf is not None else 0.0
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append(DetectionItem(
                label=str(label),
                confidence=confidence,
                severity="unclassified",
                bbox=xyxy,
                class_id=class_id,
            ))

    # Ultralytics plot() returns a BGR image with model annotations.
    annotated_bgr = result.plot()
    ok, encoded = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")
    image_data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    return DetectionResponse(
        model=Path(settings.model_path).name,
        inference_time_ms=round(inference_ms, 2),
        count=len(detections),
        annotated_image=image_data_uri,
        detections=detections,
    )

@app.get("/potholes", dependencies=[Depends(verify_api_key)])
def get_potholes(
    lat: float | None = Query(default=None, ge=-90, le=90),
    lon: float | None = Query(default=None, ge=-180, le=180),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    radius_km: float = Query(default=10, gt=0, le=100),
    limit: int = Query(default=500, ge=1, le=2000),
):
    """Return reports, optionally within radius_km of a coordinate using PostGIS."""
    center_lat = latitude if latitude is not None else lat
    center_lon = longitude if longitude is not None else lon

    with engine.connect() as conn:
        if center_lat is not None and center_lon is not None:
            stmt = text("""
                SELECT id, latitude, longitude, severity, status, confidence,
                       description, detections, source, created_at,
                       ST_Distance(
                         geog,
                         ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                       ) AS distance_m
                FROM pothole_reports
                WHERE ST_DWithin(
                    geog,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                    :radius_m
                )
                ORDER BY distance_m ASC
                LIMIT :limit
            """)
            rows = conn.execute(stmt, {
                "lat": center_lat, "lon": center_lon,
                "radius_m": radius_km * 1000, "limit": limit
            }).mappings().all()
        else:
            rows = conn.execute(text("""
                SELECT id, latitude, longitude, severity, status, confidence,
                       description, detections, source, created_at
                FROM pothole_reports
                ORDER BY created_at DESC
                LIMIT :limit
            """), {"limit": limit}).mappings().all()

    return {"potholes": [dict(row) for row in rows]}

@app.post("/potholes", dependencies=[Depends(verify_api_key)])
def create_pothole(payload: PotholeCreate):
    """Persist a reviewed detection as a geospatial report."""
    stmt = text("""
        INSERT INTO pothole_reports
          (latitude, longitude, geog, severity, status, confidence,
           description, detections, source)
        VALUES
          (:lat, :lon, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
           :severity, :status, :confidence, :description,
           CAST(:detections AS jsonb), :source)
        RETURNING id, latitude, longitude, severity, status, confidence,
                  description, detections, source, created_at
    """)
    import json
    values = payload.model_dump()
    values["lat"] = values.pop("latitude")
    values["lon"] = values.pop("longitude")
    values["detections"] = json.dumps(values["detections"])
    with engine.begin() as conn:
        row = conn.execute(stmt, values).mappings().one()
    return dict(row)

@app.get("/reports/summary", dependencies=[Depends(verify_api_key)])
def report_summary():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT severity, status, COUNT(*) AS count
            FROM pothole_reports
            GROUP BY severity, status
            ORDER BY severity, status
        """)).mappings().all()
    return {"summary": [dict(row) for row in rows]}
