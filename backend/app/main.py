from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api.v1.interview import router as interview_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS for local development and microservice testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_routes=["*"],
    allow_headers=["*"],
)

# Include the newly mapped LangGraph interaction routes
app.include_router(interview_router, prefix=settings.API_V1_STR, tags=["interview"])

@app.get("/")
def root_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}
