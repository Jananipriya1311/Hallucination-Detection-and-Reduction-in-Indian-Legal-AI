"""
app.py
======
FastAPI server for Nyaya AI - Indian Legal Assistant
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from legal_engine import get_engine, Config

# ── Configuration ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDIANKANOON_TOKEN = os.getenv("INDIANKANOON_API_TOKEN")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyaya AI - Indian Legal Assistant",
    description="AI-powered legal assistant using Groq Llama 3.3 and IndianKanoon",
    version="2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    incident_date: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Dict
    disclaimer: str
    context_used: bool = False

class DocumentAnalysisResponse(BaseModel):
    analysis: str
    sources: Dict
    disclaimer: str

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Nyaya AI",
        "version": "2.0",
        "description": "Indian Legal Assistant using Groq Llama 3.3 and IndianKanoon",
        "endpoints": {
            "/chat": "POST - Ask legal questions",
            "/analyse/upload": "POST - Upload and analyze legal documents",
            "/health": "GET - System health check",
            "/docs": "GET - API documentation"
        },
        "model": Config.GROQ_MODEL,
        "provider": "Groq"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": Config.GROQ_MODEL,
        "provider": "Groq",
        "groq_configured": bool(GROQ_API_KEY),
        "indiankanoon_configured": bool(INDIANKANOON_TOKEN)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Answer legal questions using Groq Llama 3.3
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Groq API key not configured. Please set GROQ_API_KEY environment variable."
        )
    
    try:
        engine = get_engine(GROQ_API_KEY, INDIANKANOON_TOKEN)
        result = engine.answer_question(req.message, req.history)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse/upload", response_model=DocumentAnalysisResponse)
async def analyse_upload(file: UploadFile = File(...)):
    """
    Upload and analyze a legal document (PDF or TXT)
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Groq API key not configured. Please set GROQ_API_KEY environment variable."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(content))
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            except ImportError:
                raise HTTPException(status_code=400, detail="pypdf not installed for PDF processing")
        elif file.filename.endswith('.txt'):
            text = content.decode('utf-8', errors='ignore')
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or TXT files."
            )
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from file. File may be empty or corrupted."
            )
        
        # Analyze the document
        engine = get_engine(GROQ_API_KEY, INDIANKANOON_TOKEN)
        result = engine.analyse_document(text)
        return DocumentAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get information about the model being used"""
    return {
        "model": Config.GROQ_MODEL,
        "provider": "Groq",
        "temperature": Config.TEMPERATURE,
        "max_tokens": Config.MAX_TOKENS
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )