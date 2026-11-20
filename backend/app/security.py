from fastapi import Header, HTTPException
from app.config import settings

async def verify_api_key(authorization: str | None = Header(default=None)):
    """Optional bearer-token protection. Configure API_KEY in .env to enable."""
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")
