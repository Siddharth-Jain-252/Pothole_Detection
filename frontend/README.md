# RoadSight — Pothole GIS Frontend

A React + Vite + Tailwind CSS frontend for the PotholeGuard-AI project. It removes the Streamlit UI and provides an operational dashboard around a separately deployed FastAPI inference service.

## Included screens

- **Overview:** KPI cards, map preview, recent activity and inspection workflow.
- **AI inspection:** upload a road image, call FastAPI, view annotated output and detection confidence, optionally attach the inspector's current GPS location, and submit a report.
- **GIS explorer:** OpenStreetMap + Leaflet markers, severity filters, nearby report list, radius selector and browser geolocation.
- **Reports:** searchable and filterable pothole report table.
- **Analytics:** severity mix and operational quality checks.
- **Proximity alerts:** browser notification permission and distance summaries for loaded reports.
- **Settings:** API endpoint configuration and connection test.

The map uses OpenStreetMap tiles. Sample markers around Phagwara are illustrative demo data only; they are not verified pothole reports.

## Requirements

- Node.js 18.18+ (Node 20 LTS recommended)
- npm
- Your FastAPI service deployed and reachable from the browser

## Run locally

```bash
npm install
```

Copy `.env.example` to `.env` and set your API URL and route names:

```env
VITE_API_BASE_URL=https://your-fastapi-domain
VITE_DETECT_PATH=/detect
VITE_POTHOLES_PATH=/potholes
VITE_SAVE_REPORT_PATH=/potholes
VITE_UPLOAD_FIELD=file
```

Then:

```bash
npm run dev
```

Open the URL printed by Vite (normally http://localhost:5173).

Production build:

```bash
npm run build
npm run preview
```

## FastAPI contract expected by this frontend

The endpoint names and response formats are configurable/adaptable. Update `src/lib/api.js` if your service differs.

### 1. Detection

`POST /detect` with `multipart/form-data`, image field named `file` by default. The frontend optionally also sends `latitude`, `longitude`, `lat`, and `lon`.

Example response:

```json
{
  "model": "PotholeGuard YOLOv8",
  "inference_time_ms": 142,
  "annotated_image": "BASE64_JPEG_OR_DATA_URI",
  "detections": [
    {
      "confidence": 0.91,
      "label": "pothole",
      "severity": "high",
      "bbox": [120, 80, 330, 260]
    }
  ],
  "count": 1
}
```

The normalizer also accepts `potholes`, `predictions`, `results`, `result_image`, `image`, `visualization`, `total_detections`, `processing_time_ms` and `latency_ms`.

### 2. Fetch GIS reports

`GET /potholes?lat=...&lon=...&radius_km=10`

Example response:

```json
{
  "potholes": [
    {
      "id": "p-101",
      "latitude": 31.2524,
      "longitude": 75.7052,
      "severity": "high",
      "status": "reported",
      "confidence": 0.91,
      "description": "GT Road service lane",
      "created_at": "2026-11-18T09:20:00Z"
    }
  ]
}
```

The frontend also accepts a top-level array, or `reports` / `items` wrapper.

### 3. Save a report

`POST /potholes` with JSON:

```json
{
  "latitude": 31.2524,
  "longitude": 75.7052,
  "severity": "high",
  "status": "reported",
  "confidence": 0.91,
  "description": "AI-assisted report from uploaded inspection",
  "detections": [],
  "source": "web-inspection"
}
```

For actual nearby pothole search, your backend must store GPS coordinates and query them spatially (ideally PostGIS `geography(Point, 4326)` with a radius query). A browser-only map cannot discover real potholes on its own.

## CORS

Your FastAPI app must allow the frontend origin. For local development, allow `http://localhost:5173`. For deployment, allow your actual frontend domain. Do not use wildcard origins with credentialed requests.

## Important GIS and product limitations

- Browser geolocation identifies the inspector's current position, not automatically the pothole's exact position. Dashcam GPS/EXIF or a field confirmation step is needed for precise geotagging.
- The nearby search radius control currently filters the request parameter; the backend must apply the radius query. The current report list is not distance-filtered client-side.
- Sample Phagwara map points are explicitly labelled sample data and must be replaced by real API records.
- Severity is only displayed if returned by your model/API. This frontend does not claim that bounding-box area estimates physical pothole depth.
- Proximity notifications work only while the page is open/active; they are not a background navigation service.
- For public deployment, add authentication, rate limiting, upload size enforcement, server-side image validation, report moderation and privacy controls.
- Do not expose secret API keys in `VITE_*` variables; Vite variables are shipped to the browser.

## File structure

```text
src/
  components/
    MapView.jsx
    Sidebar.jsx
    StatCard.jsx
    Topbar.jsx
  lib/
    api.js
    demoData.js
  App.jsx
  index.css
  main.jsx
```
