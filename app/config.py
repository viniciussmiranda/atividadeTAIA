"""Configurações da aplicação, lidas de variáveis de ambiente (.env)."""
from __future__ import annotations

import os
from pathlib import Path

# Carrega .env em desenvolvimento local (na Vercel, as env vars vêm do
# painel do projeto / `vercel env`, então python-dotenv é opcional).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# --- Domínio da base de conhecimento (usado no prompt do sistema) ----------
KNOWLEDGE_DOMAIN = os.getenv("KNOWLEDGE_DOMAIN", "jogos eletrônicos (games)")

# --- Provedor de LLM externo -------------------------------------------
# Suportados nativamente (API compatível com OpenAI Chat Completions):
#   groq, openai, deepseek, nvidia, huggingface
# Suportado com formato próprio:
#   gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

_DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "huggingface": "https://router.huggingface.co/v1",
}
_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "nvidia": "meta/llama-3.1-8b-instruct",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    "gemini": "gemini-1.5-flash",
}


def get_base_url() -> str:
    return LLM_BASE_URL or _DEFAULT_BASE_URLS.get(LLM_PROVIDER, "")


def get_model() -> str:
    return LLM_MODEL or _DEFAULT_MODELS.get(LLM_PROVIDER, "")


# --- Recuperação (retrieval) ------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
