"""
Testa os caminhos do grafo (Parte 2) SEM gastar tokens: a LLM é substituída
por uma versão falsa e determinística; o retrieval é o real.

    python scripts/test_graph.py            # LLM falsa (offline)
    python scripts/test_graph.py --live     # usa a LLM real (precisa de LLM_API_KEY)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import graph as G
from app import prompts as P

LIVE = "--live" in sys.argv


def fake_llm(system: str, messages: list[dict]) -> str:
    user = messages[-1]["content"]
    if system.startswith("# Papel\nVocê é um classificador"):
        msg = user.split("<mensagem>")[-1].lower()
        if "oi" in msg.split() or "olá" in msg:
            return '{"categoria": "SAUDACAO", "confianca": "ALTA"}'
        if any(w in msg for w in ("lasanha", "dor de cabeça", "presidente")):
            return '{"categoria": "FORA_DO_DOMINIO", "confianca": "ALTA"}'
        return '{"categoria": "JOGOS", "confianca": "ALTA"}'
    if system.startswith("# Papel\nVocê reescreve"):
        return "Quem publicou o GTA V?"
    if system.startswith("# Papel\nVocê é um verificador"):
        if "INVENTADO" in user:
            return '{"sustentada": false, "motivo": "fato ausente do contexto"}'
        return '{"sustentada": true, "motivo": "ok"}'
    # geração
    if "elden ring" in user.lower():
        return P.MSG_SEM_EVIDENCIA
    if "halo" in user.lower():
        return "Halo foi criado em 1492 pela Bungie INVENTADO [1]."
    return "Minecraft foi criado por Markus Persson [1].\n\nFontes: [1] Minecraft"


CASES = [
    # (descrição, pergunta, histórico, caminho esperado)
    ("saudação", "Oi", [], ["greeting"]),
    ("fora do domínio", "Qual a receita de lasanha?", [], ["out_of_domain"]),
    ("com evidência + sustentada", "Quem criou o Minecraft?", [],
     ["rewrite_question", "retrieve_context", "generate_answer", "verify_answer"]),
    ("sem evidência (piso de score)", "Como tratar dor de cabeça?", [],
     None),  # tratada abaixo
    ("sem evidência (LLM diz que não achou)", "Qual a missão principal de Elden Ring?", [],
     ["rewrite_question", "retrieve_context", "generate_answer", "no_evidence"]),
    ("resposta NÃO sustentada", "Quem criou Halo?", [],
     ["rewrite_question", "retrieve_context", "generate_answer", "verify_answer", "invalid_answer"]),
    ("continuação com histórico (reescrita)", "E quem publicou ele?",
     [{"role": "user", "content": "Quando saiu o GTA V?"},
      {"role": "assistant", "content": "Em 2013 [1]."}],
     ["rewrite_question", "retrieve_context", "generate_answer", "verify_answer"]),
]


def run(question, history):
    path, final = [], {}
    for chunk in G.get_graph().stream({"question": question, "history": history}, stream_mode="updates"):
        for node, upd in chunk.items():
            if node not in ("receive_question", "classify_question", "finalize"):
                path.append(node)
            if node == "finalize":
                final = upd
    return path, final


def main():
    if not LIVE:
        G.call_llm = fake_llm
    ok = True
    for desc, q, hist, expected in CASES:
        if expected is None:
            # força o caminho "sem evidência" por score: pergunta sem overlap de vocabulário,
            # classificada como JOGOS (o classificador falso mandaria para FORA_DO_DOMINIO)
            orig = G.retrieve
            G.retrieve = lambda query, k=None: []
            path, final = run("Como tratar dor de cabeça?" if LIVE else "zzz qqq", hist)
            G.retrieve = orig
            expected = ["rewrite_question", "retrieve_context", "no_evidence"]
        else:
            path, final = run(q, hist)
        passed = LIVE or path == expected
        ok &= passed
        print(f"[{'OK' if passed else 'FALHOU'}] {desc}")
        print(f"     pergunta : {q}")
        print(f"     caminho  : {' → '.join(path)}")
        print(f"     resposta : {final.get('answer', '')[:110]!r}")
        print(f"     fontes   : {len(final.get('sources', []))}")
    print("\nTODOS OS CAMINHOS OK" if ok else "\nHÁ FALHAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()