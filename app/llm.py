"""
Cliente para a LLM externa usada exclusivamente na etapa de geração da
resposta final (conforme exigido pela atividade).

Suporta, via variáveis de ambiente (ver .env.example):
  - Groq          (LLM_PROVIDER=groq)         -> API compatível OpenAI
  - OpenAI        (LLM_PROVIDER=openai)       -> API compatível OpenAI
  - DeepSeek      (LLM_PROVIDER=deepseek)     -> API compatível OpenAI
  - NVIDIA NIM    (LLM_PROVIDER=nvidia)       -> API compatível OpenAI
  - Hugging Face  (LLM_PROVIDER=huggingface)  -> router compatível OpenAI
  - Gemini        (LLM_PROVIDER=gemini)       -> API própria do Google

Isolar isso em um único módulo faz o LangGraph não precisar saber qual
provedor está por trás — o nó `generate_answer` do grafo apenas chama
`call_llm(...)`.
"""
from __future__ import annotations

import requests

from app.config import (
    LLM_API_KEY,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    get_base_url,
    get_model,
)

TIMEOUT_SECONDS = 45


class LLMError(RuntimeError):
    pass


def _call_openai_compatible(system_prompt: str, messages: list[dict]) -> str:
    base_url = get_base_url()
    model = get_model()
    if not LLM_API_KEY:
        raise LLMError(
            f"LLM_API_KEY não configurada para o provedor '{LLM_PROVIDER}'. "
            "Defina as variáveis de ambiente (veja .env.example)."
        )

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{base_url}/chat/completions", json=payload, headers=headers, timeout=TIMEOUT_SECONDS
    )
    if resp.status_code >= 400:
        raise LLMError(f"Erro da API {LLM_PROVIDER} ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Resposta inesperada da API {LLM_PROVIDER}: {data}") from exc


def _call_gemini(system_prompt: str, messages: list[dict]) -> str:
    if not LLM_API_KEY:
        raise LLMError("LLM_API_KEY não configurada para o provedor 'gemini'.")
    model = get_model()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={LLM_API_KEY}"
    )

    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": LLM_TEMPERATURE,
            "maxOutputTokens": LLM_MAX_TOKENS,
        },
    }
    resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    if resp.status_code >= 400:
        raise LLMError(f"Erro da API Gemini ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Resposta inesperada da API Gemini: {data}") from exc


def call_llm(system_prompt: str, messages: list[dict]) -> str:
    """
    messages: lista de {"role": "user"|"assistant", "content": str}
    (o histórico de conversa + a pergunta atual já formatada com o contexto).
    """
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, messages)
    return _call_openai_compatible(system_prompt, messages)
