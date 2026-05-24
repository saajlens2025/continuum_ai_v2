import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io

# Import our modules
from memory_engine import MemoryEngine
from vector_store import VectorStore
from llm_client import LLMClient

# Initialize FastAPI
app = FastAPI(title="Continuum AI", description="Financial Memory Layer with Vector Search")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
memory_engine = MemoryEngine(vector_store)
llm_client = LLMClient()

# ========== MODELS ==========

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    session_id: str
    response: str
    memory_context: Optional[dict] = None

class SessionCreate(BaseModel):
    user_id: Optional[str] = "anonymous"

class CSVUploadResponse(BaseModel):
    session_id: str
    rows_processed: int
    columns: List[str]
    preview: List[dict]

# ========== HEALTH & ROOT ==========

@app.get("/")
def root():
    return {
        "name": "Continuum AI",
        "version": "2.0.0",
        "status": "operational",
        "features": ["vector_search", "csv_memory", "chat_memory", "multi_session"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ========== SESSION MANAGEMENT ==========

@app.post("/api/sessions")
def create_session(data: SessionCreate):
    session_id = str(uuid.uuid4())
    memory_engine.create_session(session_id, data.user_id)
    return {"session_id": session_id, "user_id": data.user_id}

@app.get("/api/sessions/{user_id}")
def get_sessions(user_id: str):
    sessions = memory_engine.get_user_sessions(user_id)
    return {"sessions": sessions}

# ========== CSV UPLOAD ==========

@app.post("/api/upload")
async def upload_csv(
    session_id: str = Form(...),
    user_id: str = Form("anonymous"),
    file: UploadFile = File(...)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are allowed")
    
    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Process rows
    rows_processed = memory_engine.process_csv_upload(
        session_id=session_id,
        user_id=user_id,
        df=df,
        filename=file.filename
    )
    
    return CSVUploadResponse(
        session_id=session_id,
        rows_processed=rows_processed,
        columns=df.columns.tolist(),
        preview=df.head(5).to_dict(orient='records')
    )

# ========== CHAT ==========

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Store user message
    memory_engine.store_message(
        session_id=request.session_id,
        user_id=request.user_id,
        role="user",
        content=request.message
    )
    
    # Retrieve relevant memory context
    context = memory_engine.retrieve_context(
        session_id=request.session_id,
        query=request.message
    )
    
    # Generate LLM response
    response = llm_client.generate(
        user_input=request.message,
        context=context,
        session_id=request.session_id
    )
    
    # Store assistant response
    memory_engine.store_message(
        session_id=request.session_id,
        user_id=request.user_id,
        role="assistant",
        content=response
    )
    
    return ChatResponse(
        session_id=request.session_id,
        response=response,
        memory_context=context
    )

# ========== MEMORY VIEWER ==========

@app.get("/api/memory/{session_id}")
def get_session_memory(session_id: str):
    memory = memory_engine.get_session_memory(session_id)
    return {"session_id": session_id, "memory": memory}

@app.delete("/api/memory/{session_id}/{message_id}")
def delete_message(session_id: str, message_id: int):
    memory_engine.soft_delete_message(session_id, message_id)
    return {"status": "deleted"}

# ========== ADMIN ==========

@app.get("/api/admin/stats")
def get_stats():
    stats = memory_engine.get_system_stats()
    return stats

@app.get("/api/admin/training-data")
def get_training_data(limit: int = 100):
    data = memory_engine.get_training_data(limit)
    return {"training_data": data}