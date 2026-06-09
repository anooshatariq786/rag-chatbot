from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os, shutil, uuid
from pathlib import Path

from rag_engine import RAGEngine

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Schemas ---
class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": rag.get_document_count()}

@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"File type {suffix} not supported. Use: {allowed}")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunk_count = rag.add_document(str(save_path), file.filename, doc_id)
    return DocumentInfo(id=doc_id, filename=file.filename, chunk_count=chunk_count)

@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return rag.list_documents()

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    success = rag.delete_document(doc_id)
    if not success:
        raise HTTPException(404, "Document not found")
    return {"message": "Document deleted"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = rag.query(req.question, req.conversation_history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])

# Serve React frontend (for HF Spaces)
frontend_path = Path("frontend_build")
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")

    @app.get("/")
    def serve_root():
        return FileResponse(str(frontend_path / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file = frontend_path / full_path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(frontend_path / "index.html"))
