"""API route definitions."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Liveness probe used by orchestrators and uptime checks."""
    return {"status": "ok"}
