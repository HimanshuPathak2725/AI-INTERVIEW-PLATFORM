from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.v1 import interview

app = FastAPI(title="AI Interview Platform API")

# Global CORS setup ensuring every incoming route header passes through
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Wildcard origins (*) ke sath allow_credentials False hona chahiye warna browser headers block kar deta hai
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core routing mounts
app.include_router(interview.router, prefix="/api/v1", tags=["Session"])

@app.get("/")
def read_root():
    return {"status": "healthy", "engine": "LangGraph Active"}