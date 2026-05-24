from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Continuum AI", description="Financial Memory Layer")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": "Continuum AI",
        "version": "1.0.0",
        "status": "operational",
        "message": "Memory layer is alive"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/status")
def api_status():
    return {
        "memory_engine": "ready",
        "vector_search": "pending",
        "llm_connection": "pending",
        "active_sessions": 0
    }