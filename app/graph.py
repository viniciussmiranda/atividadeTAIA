"""
Fluxo principal do RAG, orquestrado com LangGraph.

Nós do grafo (conforme pedido pela atividade):
  1. receive_question  -> entrada da pergunta do usuário
  2. retrieve_context   -> recuperação dos chunks mais relevantes no índice vetorial
  3. build_prompt        -> montagem do contexto (chunks + histórico) para a LLM
  4. generate_answer     -> chamada da LLM externa
  5. finalize             -> retorno da resposta (+ fontes usadas)

O estado do grafo é um TypedDict simples que vai sendo preenchido por cada nó.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import KNOWLEDGE_DOMAIN, MAX_HISTORY_MESSAGES
from app.llm import call_llm
from app.retrieval import RetrievedChunk, retrieve

SYSTEM_PROMPT_TEMPLATE = """Você é um assistente virtual especializado em {domain}.
Responda SEMPRE em português, de forma clara e objetiva.

Use apenas as informações fornecidas no CONTEXTO abaixo para responder. Se o
CONTEXTO não tiver informação suficiente para responder com segurança, diga
que não encontrou essa informação na base de conhecimento, em vez de inventar
uma resposta. Não mencione que está "consultando um contexto" — responda de
forma natural, como um especialista no assunto.

CONTEXTO:
{context}
"""


class ChatMessage(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class GraphState(TypedDict, total=False):
    question: str
    history: list[ChatMessage]
    retrieved: list[RetrievedChunk]
    context: str
    system_prompt: str
    llm_messages: list[ChatMessage]
    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# Nós
# ---------------------------------------------------------------------------

def receive_question(state: GraphState) -> GraphState:
    """Normaliza a entrada do usuário (nó de entrada da pergunta)."""
    question = (state.get("question") or "").strip()
    return {"question": question}


def retrieve_context(state: GraphState) -> GraphState:
    """Busca os chunks mais relevantes no índice vetorial local."""
    chunks = retrieve(state["question"])
    return {"retrieved": chunks}


def build_prompt(state: GraphState) -> GraphState:
    """Monta o contexto (chunks recuperados) e o prompt de sistema, e
    combina com o histórico recente de mensagens para dar memória de curto
    prazo ao chatbot."""
    retrieved = state.get("retrieved") or []
    if retrieved:
        context = "\n\n---\n\n".join(
            f"[Fonte: {c.title}]\n{c.text}" for c in retrieved
        )
    else:
        context = "(nenhum trecho relevante encontrado na base de conhecimento)"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(domain=KNOWLEDGE_DOMAIN, context=context)

    history = (state.get("history") or [])[-MAX_HISTORY_MESSAGES:]
    llm_messages: list[ChatMessage] = list(history)
    llm_messages.append({"role": "user", "content": state["question"]})

    sources = [
        {"title": c.title, "source": c.source, "score": round(c.score, 4)}
        for c in retrieved
    ]

    return {"context": context, "system_prompt": system_prompt, "llm_messages": llm_messages, "sources": sources}


def generate_answer(state: GraphState) -> GraphState:
    """Chama a LLM externa para gerar a resposta final."""
    if not state["question"]:
        return {"answer": "Pode reescrever sua pergunta? Não recebi nenhum texto."}
    answer = call_llm(state["system_prompt"], state["llm_messages"])
    return {"answer": answer}


def finalize(state: GraphState) -> GraphState:
    """Nó de saída: nada a transformar, apenas garante as chaves finais."""
    return {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
    }


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("receive_question", receive_question)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "retrieve_context")
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", "generate_answer")
    graph.add_edge("generate_answer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_rag(question: str, history: list[ChatMessage] | None = None) -> dict:
    graph = get_graph()
    result = graph.invoke({"question": question, "history": history or []})
    return {"answer": result["answer"], "sources": result["sources"]}
