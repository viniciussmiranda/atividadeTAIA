"""
Entrypoint da aplicação (Vercel detecta `app` automaticamente neste arquivo).

Expõe:
  GET  /            -> interface web do chatbot (public/index.html)
  GET  /api/health  -> healthcheck simples
  POST /api/chat    -> recebe {message, history[]} e roda o grafo LangGraph
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import KNOWLEDGE_DOMAIN
from app.graph import run_rag
from app.llm import LLMError

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Chatbot RAG com LangGraph", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessageIn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/", include_in_schema=False)
def serve_index():
    index_path = BASE_DIR / "public" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface não encontrada.")
    return FileResponse(index_path)


@app.get("/api/health")
def health():
    return {"status": "ok", "domain": KNOWLEDGE_DOMAIN}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    try:
        result = run_rag(req.message, history)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        # ex.: índice vetorial ausente (esqueceram de rodar o ingest)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=result["answer"], sources=result["sources"])
