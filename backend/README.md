# RoadSight FastAPI + PostGIS backend

FastAPI backend for the React/Tailwind frontend delivered with this project. It loads the PotholeGuard-AI YOLOv8 checkpoint, performs image inference, returns an annotated image and detection metadata, and stores/searches reviewed pothole reports using PostGIS.

## 1. Add the trained model

Copy the repository's trained checkpoint into:

```text
models/yolov8_pothole_best.pt
```

If your checkpoint has another name, update `MODEL_PATH` in `.env`.

The backend does not download model weights automatically. Use the actual trained checkpoint from the PotholeGuard-AI repository.

## 2. Start PostGIS

Docker Desktop must be running.

```powershell
docker compose up -d postgis
```

## 3. Create Python environment

Python 3.10 or 3.11 recommended.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. On PowerShell:

```powershell
Copy-Item .env.example .env
```

Check `DATABASE_URL`, `MODEL_PATH`, and `CORS_ORIGINS`.

## 4. Run API

From the backend root:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

The first start loads the YOLO weights into memory and initializes PostGIS tables/indexes. Keep the model checkpoint out of source control if it is large or has redistribution restrictions.

## API endpoints

### POST `/detect`

- Multipart field: `file` (configurable on frontend with `VITE_UPLOAD_FIELD=file`)
- Returns `annotated_image` as a JPEG data URI, `detections`, `count`, `model`, and `inference_time_ms`.
- Each detection includes `label`, `confidence`, `severity="unclassified"`, `bbox` (`[x1,y1,x2,y2]`) and `class_id`.

This project intentionally does not invent physical pothole severity from a bounding box. Add a separately trained/validated severity model or a human review workflow if severity is needed.

### GET `/potholes`

Example:

```text
GET /potholes?lat=31.252&lon=75.705&radius_km=5
```

Uses PostGIS `geography` and `ST_DWithin` to return pothole reports inside the requested radius. The query uses metres internally and uses a GiST spatial index.

### POST `/potholes`

Example body:

```json
{
  "latitude": 31.2524,
  "longitude": 75.7052,
  "severity": "unclassified",
  "status": "reported",
  "confidence": 0.91,
  "description": "AI-assisted road hazard report",
  "detections": [],
  "source": "web-inspection"
}
```

### GET `/reports/summary`

Returns counts grouped by severity and status.

## Connect the React frontend

In the frontend `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_DETECT_PATH=/detect
VITE_POTHOLES_PATH=/potholes
VITE_SAVE_REPORT_PATH=/potholes
VITE_UPLOAD_FIELD=file
```

The FastAPI CORS configuration must include the frontend origin. If deploying the frontend, set `CORS_ORIGINS` to its exact origin and redeploy the backend.

## GPU notes

- `MODEL_DEVICE=auto` lets Ultralytics select the available device.
- Set `MODEL_DEVICE=cpu` to force CPU.
- Set `MODEL_DEVICE=0` to force CUDA GPU 0.
- CUDA requires a PyTorch build compatible with the installed NVIDIA driver/CUDA runtime. The default pip install may install CPU PyTorch; check `torch.cuda.is_available()` and install a matching PyTorch build if GPU inference is needed.

## Optional API key

Set `API_KEY` to a long random secret in backend `.env`. Protected routes then require:

```http
Authorization: Bearer YOUR_SECRET
```

Do not put a private API key in a public Vite `VITE_*` environment variable. Vite variables are exposed to browser users. For production, use user authentication/session tokens or a backend-for-frontend rather than a shared browser secret.

## Important production work still needed

- Add user authentication, authorization, rate limiting and audit logging.
- Add image retention/deletion policy and malware/content validation.
- Add moderation and duplicate-report merging.
- Add a verified field location workflow. The browser's current GPS is the inspector's location, not necessarily the pothole location.
- Configure HTTPS, managed PostGIS backups, monitoring and migrations.
- Verify model licensing and dataset terms before redistribution.
