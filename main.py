"""ACRF application entrypoint.

Run with: uv run uvicorn main:app --reload
"""

from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
