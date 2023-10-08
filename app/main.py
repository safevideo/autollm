import logging

from fastapi import FastAPI

from .docs import title, description, version, openapi_url, terms_of_service, tags_metadata
from .api import ask_question, health_check

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=title,
    description=description,
    version=version,
    openapi_url=openapi_url,
    terms_of_service=terms_of_service,
    openapi_tags=tags_metadata,
)

# Include the API routes
app.include_router(ask_question.router, tags=["ask"])
app.include_router(health_check.router, tags=["health"])
