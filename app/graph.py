"""
Fluxo principal do RAG, orquestrado com LangGraph.

Fluxo:
  1. receive_question
  2. classify_question
  3. decisão de classificação
  4. rewrite_question
  5. retrieve_context
  6. decisão de evidência
  7. generate_answer
  8. decisão: LLM respondeu "sem evidência"?
  9. verify_answer
  10. decisão da verificação
  11. finalize
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import FEW_SHOT, KNOWLEDGE_DOMAIN, MIN_RETRIEVAL_SCORE
from app.llm import call_llm
from app.retrieval import RetrievedChunk, retrieve
from app.prompts import (
    MSG_FORA_DO_DOMINIO,
    MSG_SAUDACAO,
    MSG_SEM_EVIDENCIA,
    MSG_NAO_SUSTENTADA,
    build_classify_messages,
    build_rewrite_messages,
    build_answer_messages,
    build_verify_messages,
    normalize_classificacao,
    normalize_verificacao,
    parse_json_object,
    clean_rewrite_output,
    is_sem_evidencia,
    needs_rewrite,
)


class ChatMessage(TypedDict):
    role: str
    content: str


class GraphState(TypedDict, total=False):
    question: str
    history: list[ChatMessage]

    classification: dict

    reformulated_question: str

    retrieved: list[RetrievedChunk]
    sources: list[dict]

    answer: str
    verification: dict


# ---------------------------------------------------------------------------
# Nós
# ---------------------------------------------------------------------------

def receive_question(state: GraphState) -> GraphState:
    """Normaliza a entrada do usuário."""
    question = (state.get("question") or "").strip()

    return {
        "question": question
    }


def classify_question(state: GraphState) -> GraphState:
    """Classifica a pergunta usando o prompt da Parte 1."""
    question = state.get("question", "")
    history = state.get("history") or []

    # Mensagem vazia: nada a classificar, não gasta chamada de LLM.
    if not question:
        return {
            "classification": {
                "categoria": "SAUDACAO",
                "confianca": "ALTA",
                "valido": True,
            }
        }

    system_prompt, messages = build_classify_messages(
        question=question,
        history=history,
        few_shot=FEW_SHOT,
        domain=KNOWLEDGE_DOMAIN,
    )

    raw_response = call_llm(system_prompt, messages)

    classification = normalize_classificacao(
        parse_json_object(raw_response)
    )

    return {
        "classification": classification
    }


def route_after_classification(state: GraphState) -> str:
    """Decide se a pergunta pode seguir para o RAG."""
    classification = state.get("classification") or {}
    category = classification.get("categoria")

    if category == "SAUDACAO":
        return "greeting"

    if category == "FORA_DO_DOMINIO":
        return "out_of_domain"

    return "games"


def greeting(state: GraphState) -> GraphState:
    """Resposta direta para uma saudação."""
    return {
        "answer": MSG_SAUDACAO,
        "sources": [],
    }


def out_of_domain(state: GraphState) -> GraphState:
    """Resposta direta para perguntas fora do domínio."""
    return {
        "answer": MSG_FORA_DO_DOMINIO,
        "sources": [],
    }


def rewrite_question(state: GraphState) -> GraphState:
    """Reformula a pergunta para melhorar a busca no RAG.

    Sem histórico não há referência a resolver (needs_rewrite), então a
    pergunta original é usada e a chamada de LLM é evitada."""
    question = state.get("question", "")
    history = state.get("history") or []

    if not needs_rewrite(history):
        return {"reformulated_question": question}

    system_prompt, messages = build_rewrite_messages(
        question=question,
        history=history,
        domain=KNOWLEDGE_DOMAIN,
    )

    raw_response = call_llm(system_prompt, messages)

    reformulated = clean_rewrite_output(
        raw_response,
        question,
    )

    return {
        "reformulated_question": reformulated
    }


def retrieve_context(state: GraphState) -> GraphState:
    """Busca os chunks mais relevantes usando a pergunta reformulada."""
    question = state.get("reformulated_question") or state.get("question", "")

    chunks = retrieve(question)

    sources = [
        {
            "title": chunk.title,
            "source": chunk.source,
            "score": round(chunk.score, 4),
        }
        for chunk in chunks
    ]

    return {
        "retrieved": chunks,
        "sources": sources,
    }


def route_after_retrieval(state: GraphState) -> str:
    """Verifica se a recuperação encontrou alguma evidência relevante.

    retrieve() sempre devolve os top-k, então além de "lista vazia" usamos
    um piso mínimo de similaridade (MIN_RETRIEVAL_SCORE)."""
    retrieved = state.get("retrieved") or []

    if not retrieved:
        return "no_evidence"

    if max(chunk.score for chunk in retrieved) < MIN_RETRIEVAL_SCORE:
        return "no_evidence"

    return "has_evidence"


def no_evidence(state: GraphState) -> GraphState:
    """Resposta direta quando o RAG não encontrou evidência."""
    return {
        "answer": MSG_SEM_EVIDENCIA,
        "sources": [],
    }


def generate_answer(state: GraphState) -> GraphState:
    """Gera a resposta usando exclusivamente os chunks recuperados."""
    question = (
        state.get("reformulated_question")
        or state.get("question", "")
    )

    retrieved = state.get("retrieved") or []

    system_prompt, messages = build_answer_messages(
        question=question,
        chunks=retrieved,
        domain=KNOWLEDGE_DOMAIN,
    )

    answer = call_llm(system_prompt, messages)

    return {
        "answer": answer
    }


def route_after_generation(state: GraphState) -> str:
    """Se o LLM respondeu 'não encontrei', não há o que verificar."""
    if is_sem_evidencia(state.get("answer", "")):
        return "no_evidence"

    return "verify"


def verify_answer(state: GraphState) -> GraphState:
    """Verifica se a resposta gerada é sustentada pelo contexto."""
    question = (
        state.get("reformulated_question")
        or state.get("question", "")
    )

    retrieved = state.get("retrieved") or []
    answer = state.get("answer", "")

    system_prompt, messages = build_verify_messages(
        question=question,
        chunks=retrieved,
        answer=answer,
        domain=KNOWLEDGE_DOMAIN,
    )

    raw_response = call_llm(system_prompt, messages)

    verification = normalize_verificacao(
        parse_json_object(raw_response)
    )

    return {
        "verification": verification
    }


def route_after_verification(state: GraphState) -> str:
    """Decide se a resposta foi validada."""
    verification = state.get("verification") or {}

    if verification.get("sustentada") is True:
        return "valid"

    return "invalid"


def invalid_answer(state: GraphState) -> GraphState:
    """Substitui uma resposta não sustentada por uma resposta segura."""
    return {
        "answer": MSG_NAO_SUSTENTADA,
        "sources": [],
    }


def finalize(state: GraphState) -> GraphState:
    """Nó de saída."""
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
    graph.add_node("classify_question", classify_question)
    graph.add_node("greeting", greeting)
    graph.add_node("out_of_domain", out_of_domain)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("no_evidence", no_evidence)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("invalid_answer", invalid_answer)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("receive_question")

    graph.add_edge(
        "receive_question",
        "classify_question",
    )

    graph.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {
            "greeting": "greeting",
            "out_of_domain": "out_of_domain",
            "games": "rewrite_question",
        },
    )

    graph.add_edge(
        "greeting",
        "finalize",
    )

    graph.add_edge(
        "out_of_domain",
        "finalize",
    )

    graph.add_edge(
        "rewrite_question",
        "retrieve_context",
    )

    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "no_evidence": "no_evidence",
            "has_evidence": "generate_answer",
        },
    )

    graph.add_edge(
        "no_evidence",
        "finalize",
    )

    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {
            "no_evidence": "no_evidence",
            "verify": "verify_answer",
        },
    )

    graph.add_conditional_edges(
        "verify_answer",
        route_after_verification,
        {
            "valid": "finalize",
            "invalid": "invalid_answer",
        },
    )

    graph.add_edge(
        "invalid_answer",
        "finalize",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph

    if _compiled_graph is None:
        _compiled_graph = build_graph()

    return _compiled_graph


def run_rag(
    question: str,
    history: list[ChatMessage] | None = None,
) -> dict:
    graph = get_graph()

    result = graph.invoke({
        "question": question,
        "history": history or [],
    })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }