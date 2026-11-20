from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.config import settings

engine: Engine = create_engine(settings.database_url, pool_pre_ping=True)

def init_db() -> None:
    """Create PostGIS-backed report storage and a spatial index."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pothole_reports (
                id BIGSERIAL PRIMARY KEY,
                latitude DOUBLE PRECISION NOT NULL,
                longitude DOUBLE PRECISION NOT NULL,
                geog geography(Point, 4326) NOT NULL,
                severity VARCHAR(30) NOT NULL DEFAULT 'unclassified',
                status VARCHAR(30) NOT NULL DEFAULT 'reported',
                confidence DOUBLE PRECISION,
                description TEXT NOT NULL DEFAULT 'Road hazard report',
                detections JSONB NOT NULL DEFAULT '[]'::jsonb,
                source VARCHAR(80) NOT NULL DEFAULT 'web-inspection',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_pothole_reports_geog
            ON pothole_reports USING GIST (geog)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_pothole_reports_created_at
            ON pothole_reports (created_at DESC)
        """))
