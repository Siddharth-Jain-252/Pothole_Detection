# Pothole Detection & Road Intelligence System

An AI-powered pothole detection and road monitoring platform designed to identify road damage from images using computer vision and deep learning. The system uses a custom-trained YOLOv8 model and a FastAPI backend, with geospatial data management planned around PostgreSQL and PostGIS.

## Overview

Road damage, particularly potholes, creates safety risks for commuters and increases vehicle maintenance and infrastructure repair costs. Manually identifying and documenting potholes across large road networks is time-consuming.

This project aims to automate pothole identification through deep learning and provide a foundation for geospatial road-condition monitoring.

Users can submit road images for AI-based pothole detection. The backend processes the image using a trained YOLOv8 model and returns detection results, including bounding boxes and confidence scores. Detected potholes can subsequently be associated with geographic coordinates and stored for mapping and reporting.

Deployed Link: https://pothole-detection-vert.vercel.app/
## Key Features

- **AI-based Pothole Detection:** Uses a custom-trained YOLOv8 object detection model.
- **Image Processing:** Processes uploaded road images and generates annotated detection results.
- **Confidence-based Detection:** Returns confidence scores for individual detections.
- **REST API:** Exposes model inference through FastAPI endpoints.
- **Geospatial Data Management:** Designed to store pothole reports with latitude and longitude using PostgreSQL and PostGIS.
- **Nearby Pothole Discovery:** Supports the planned retrieval of pothole reports within a geographic radius.
- **Scalable Architecture:** Separates model inference, API handling, and data management.
- **Cloud Deployment:** Backend is containerized using Docker and intended for deployment on Render.

## System Architecture

```text
             React Frontend
                   |
                   | HTTP / REST API
                   v
             FastAPI Backend
                   |
          +--------+---------+
          |                  |
          v                  v
     YOLOv8 Model       GIS / Report Service
          |                  |
          v                  v
     Image Inference    PostgreSQL + PostGIS
          |                  |
          +--------+---------+
                   |
                   v
             Detection Results
```

### Backend

FastAPI provides the REST API layer and handles image uploads, request validation, model inference, and report-related operations.

### Detection Engine

The detection engine uses Ultralytics YOLOv8 and the project's trained checkpoint to identify potholes in road images.

### Database and GIS

PostgreSQL with PostGIS is intended for storing geospatial pothole reports and performing location-based queries.


### Frontend

The frontend is designed around React and Tailwind CSS, providing image upload, detection visualization, geospatial exploration, and reporting interfaces. If maintained separately, configure its API base URL to point to the deployed backend.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8 |
| Deep Learning | PyTorch |
| Model Framework | Ultralytics |
| Image Processing | OpenCV, Pillow, NumPy |
| Backend | FastAPI |
| API Server | Uvicorn |
| Database | PostgreSQL |
| Geospatial Database | PostGIS |
| ORM | SQLAlchemy |
| Containerization | Docker |
| Deployment | Render |
| Frontend | React, Tailwind CSS |
| GIS Visualization | Leaflet / React Leaflet |

## Project Structure

The following is the intended organization. Adjust paths to match the actual repository.

```text
Pothole_Detection/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   ├── routers/
│   │   │   ├── detection.py
│   │   │   ├── potholes.py
│   │   │   └── analytics.py
│   │   └── services/
│   │       ├── detection_service.py
│   │       └── gis_service.py
│   │
│   ├── models/
│   │   └── yolov8_pothole_best.pt
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── dataset_yolo/
├── dataset-pothole-main/
├── evaluation_metrics/
│
├── training/
│   └── train_yolov8.py
│
├── inference_yolov8.py
├── pothole_dataset.yaml
├── README.md
├── TRAINING.md
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.11 recommended
- Git
- Docker (optional for local database setup)
- PostgreSQL with PostGIS (for database-dependent functionality)
- Trained YOLOv8 checkpoint

### 1. Clone the repository

```bash
git clone https://github.com/Siddharth-Jain-252/Pothole_Detection.git

cd Pothole_Detection
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv

.venv\Scripts\activate
```

Linux / macOS:

```bash
python3.11 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

pip install -r backend/requirements.txt
```

Run this from the repository root if your backend requirements file is located at `backend/requirements.txt`.

### 4. Configure environment variables

Create a `.env` file in the backend working directory.

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE

MODEL_PATH=./models/yolov8_pothole_best.pt
MODEL_DEVICE=auto

MODEL_CONFIDENCE=0.25
MODEL_IOU=0.45
MAX_UPLOAD_MB=15

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

API_KEY=
```

Configuration notes:

- `DATABASE_URL`: PostgreSQL connection string.
- `MODEL_PATH`: Location of the trained YOLOv8 checkpoint.
- `MODEL_DEVICE`: Inference device, depending on available hardware.
- `MODEL_CONFIDENCE`: Detection confidence threshold.
- `MODEL_IOU`: Non-maximum suppression IoU threshold.
- `MAX_UPLOAD_MB`: Maximum permitted image upload size.
- `CORS_ORIGINS`: Allowed frontend origins.
- `API_KEY`: Optional API authentication token.

Do not commit `.env` or database credentials to GitHub.

### 5. Start the backend

From the backend directory:

```bash
cd backend

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative API documentation:

```text
http://127.0.0.1:8000/redoc
```

## API Documentation

The FastAPI application exposes REST endpoints for image detection and other backend functionality.

The exact routes and request schemas should be taken from the implemented routers.

| Operation | Purpose |
|---|---|
| Health Check | Verify backend availability |
| Image Detection | Upload an image for YOLOv8 inference |
| Pothole Reports | Create and retrieve pothole reports |
| Nearby Potholes | Retrieve reports based on geographic radius |
| Analytics | Retrieve aggregated report statistics |

Refer to `/docs` for the exact available routes, request parameters, and response schemas.

## Detection Workflow

1. The user uploads a road image through the frontend.
2. The frontend sends the image to the FastAPI backend.
3. FastAPI validates the uploaded image.
4. The detection service loads or accesses the trained YOLOv8 model.
5. YOLOv8 performs object detection.
6. Bounding boxes and confidence scores are extracted.
7. The backend generates an annotated image.
8. Detection results are returned to the frontend.
9. Where location information is available, the user can associate the detection with geographic coordinates.
10. Reviewed reports can be stored for future mapping and analysis.

## GIS and Location Intelligence

The geospatial component is intended to support location-aware road monitoring.

Potential capabilities include:

- Storing pothole coordinates.
- Displaying reported potholes on an interactive map.
- Searching for nearby potholes.
- Filtering reports by location and status.
- Tracking reported and resolved road damage.

Location data should come from a valid GPS source, such as browser geolocation or image EXIF metadata. Image detections alone do not establish geographic coordinates.

PostGIS can support efficient radius-based geographic queries using spatial indexes.

## Model Information

The project uses a custom-trained YOLOv8 checkpoint.

Expected checkpoint:

```text
models/yolov8_pothole_best.pt
```

The inference pipeline uses the configured confidence and IoU thresholds.

For information about dataset preparation, training, validation, and model export, see [TRAINING.md](TRAINING.md).

## Docker

Build the backend image from the directory containing its Dockerfile:

```bash
docker build -t pothole-detection-backend .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env pothole-detection-backend
```

Make sure the model checkpoint is included in the Docker build context and copied into the expected model directory.

## Deployment on Render

The backend can be deployed as a Docker-based Web Service on Render.

Deployment overview:

1. Push the project to GitHub.
2. Create a new Render Web Service.
3. Connect the GitHub repository.
4. Select Docker as the runtime.
5. Set the correct Dockerfile path and build context.
6. Configure the required environment variables.
7. Configure the model checkpoint path.
8. Deploy the service.
9. Verify the health endpoint and API documentation.

For database-backed features, provision a PostgreSQL database with PostGIS support and configure its connection URL.

The application must listen on `0.0.0.0` and use Render's assigned `PORT` environment variable.

## Important Considerations

- YOLO confidence is not equivalent to physical pothole severity.
- Severity classification requires a separately validated method or human review.
- GPS coordinates must be obtained from a real location source.
- Persistent image storage should use external or persistent storage rather than relying on an ephemeral container filesystem.
- Model weights and datasets may require Git LFS or external storage if they exceed GitHub's file-size limits.
- Database credentials and API keys must be stored as deployment environment variables.

## Future Improvements

- Real-time pothole detection from video streams.
- GPS-based road damage reporting.
- Improved severity estimation using additional visual or geometric information.
- Duplicate report detection.
- Road-condition analytics and heatmaps.
- Model optimization using ONNX or TensorRT.
- Automated model monitoring and retraining.
- Role-based access control for report management.


## License

Add the project's actual license information here. If no license has been selected, do not assume the code is open-source licensed.
