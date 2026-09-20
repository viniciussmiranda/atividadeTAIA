"""
Script de ingestão da base de conhecimento.

Lê os documentos brutos em data/raw/ (.txt, .md, .html, .pdf), limpa o texto,
quebra em chunks, gera vetores (TF-IDF + SVD, ou seja, embeddings densos por
Latent Semantic Analysis) e salva o índice vetorial em data/processed/.

Por que TF-IDF + SVD em vez de um modelo de embeddings tipo sentence-transformers?
--------------------------------------------------------------------------------
O backend deste projeto é pensado para rodar como uma Vercel Function em
Python, que tem um limite de bundle de 500MB e não deveria baixar modelos
pesados a cada cold start. TF-IDF + SVD (LSA) gera embeddings densos e um
índice vetorial de verdade (busca por similaridade de cosseno), mas usa
apenas scikit-learn/numpy — poucos MBs, sem downloads, sem GPU e com busca
praticamente instantânea. Além disso, a atividade pede que a LLM externa seja
usada "apenas na etapa de geração da resposta"; mantendo os embeddings 100%
locais isso fica garantido por construção.

Uso:
    python scripts/ingest.py
    python scripts/ingest.py --raw-dir data/raw --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# Stopwords em português (lista compacta, não depende de downloads externos)
# ---------------------------------------------------------------------------
PT_STOPWORDS = sorted({
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "fosse", "grande", "há", "isso", "isto",
    "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha",
    "minhas", "muito", "na", "nas", "nem", "no", "nos", "nossa", "nossas",
    "nosso", "nossos", "num", "numa", "não", "o", "os", "ou", "para", "pela",
    "pelas", "pelo", "pelos", "pode", "podem", "por", "qual", "quando", "que",
    "quem", "se", "seu", "seus", "sua", "suas", "são", "só", "também", "te",
    "tem", "têm", "tendo", "ter", "teu", "teus", "toda", "todas", "todo",
    "todos", "tu", "tua", "tuas", "um", "uma", "umas", "uns", "você", "vocês",
})

CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 120


@dataclass
class Chunk:
    id: int
    text: str
    title: str
    source: str
    doc_file: str


def clean_text(text: str) -> str:
    """Normaliza espaços em branco e remove linhas vazias em excesso."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title_and_source(text: str, fallback_title: str) -> tuple[str, str, str]:
    """Se o arquivo começar com um cabeçalho '[Título: X | Fonte: Y]', extrai."""
    header_match = re.match(r"^\[Título:\s*(.*?)\s*\|\s*Fonte:\s*(.*?)\]\s*\n+", text)
    if header_match:
        title, source = header_match.group(1), header_match.group(2)
        body = text[header_match.end():]
        return title, source, body
    return fallback_title, "", text


def read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Para ler PDFs instale as dependências de ingestão: "
            "pip install -r requirements-ingest.txt"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Para ler HTML instale as dependências de ingestão: "
            "pip install -r requirements-ingest.txt"
        ) from exc
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


READERS = {
    ".txt": read_txt_or_md,
    ".md": read_txt_or_md,
    ".pdf": read_pdf,
    ".html": read_html,
    ".htm": read_html,
}


def load_documents(raw_dir: Path) -> list[dict]:
    docs = []
    if not raw_dir.exists():
        return docs
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in READERS:
            continue
        raw_text = READERS[path.suffix.lower()](path)
        raw_text = clean_text(raw_text)
        if not raw_text:
            continue
        title, source, body = extract_title_and_source(raw_text, fallback_title=path.stem)
        docs.append({"title": title, "source": source, "text": body, "doc_file": path.name})
    return docs


def _format_game_text(game: dict) -> str:
    """Monta um texto legível (e indexável) a partir de um registro de jogo
    vindo da API, juntando os metadados estruturados com a descrição."""
    lines = []
    if game.get("released"):
        lines.append(f"Lançamento: {game['released']}")
    if game.get("genres"):
        lines.append(f"Gêneros: {', '.join(game['genres'])}")
    if game.get("platforms"):
        lines.append(f"Plataformas: {', '.join(game['platforms'])}")
    if game.get("developers"):
        lines.append(f"Desenvolvedora(s): {', '.join(game['developers'])}")
    if game.get("publishers"):
        lines.append(f"Publicadora(s): {', '.join(game['publishers'])}")
    if game.get("metacritic"):
        lines.append(f"Nota Metacritic: {game['metacritic']}")
    if game.get("esrb_rating"):
        lines.append(f"Classificação etária: {game['esrb_rating']}")
    header = "\n".join(lines)
    description = (game.get("description") or "").strip()
    return f"{header}\n\n{description}".strip()


def load_games_json(games_json: Path) -> list[dict]:
    """Carrega jogos coletados via scripts/fetch_games_api.py (lista de dicts
    com name/description/genres/platforms/... vindos da API) no mesmo formato
    de documento usado pelos arquivos de data/raw/."""
    if not games_json.exists():
        return []
    with open(games_json, "r", encoding="utf-8") as f:
        games = json.load(f)
    docs = []
    for game in games:
        name = game.get("name")
        text = _format_game_text(game)
        if not name or not text:
            continue
        docs.append({
            "title": name,
            "source": game.get("website", ""),
            "text": text,
            "doc_file": games_json.name,
        })
    return docs


def split_into_chunks(text: str) -> list[str]:
    """Quebra o texto em chunks por parágrafo, respeitando um tamanho alvo
    de caracteres, com sobreposição entre chunks consecutivos para preservar
    contexto nas bordas."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= CHUNK_SIZE_CHARS:
            current = f"{current}\n{para}".strip()
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
            # inicia o próximo chunk com sobreposição do final do anterior
            overlap = current[-CHUNK_OVERLAP_CHARS:] if current else ""
            current = f"{overlap}\n{para}".strip()
            # parágrafo sozinho maior que o tamanho alvo: corta em pedaços
            while len(current) > CHUNK_SIZE_CHARS * 1.5:
                chunks.append(current[:CHUNK_SIZE_CHARS])
                current = current[CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS:]
    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)
    return chunks


def build_chunks(docs: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = 0
    for doc in docs:
        for piece in split_into_chunks(doc["text"]):
            chunks.append(Chunk(
                id=next_id,
                text=piece,
                title=doc["title"],
                source=doc["source"],
                doc_file=doc["doc_file"],
            ))
            next_id += 1
    return chunks


def build_index(chunks: list[Chunk], n_components: int = 200):
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=PT_STOPWORDS,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = max(2, min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_embeddings = svd.fit_transform(tfidf_matrix)

    # normaliza L2 para que o produto escalar vire similaridade de cosseno
    norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense_embeddings = (dense_embeddings / norms).astype(np.float32)

    return vectorizer, svd, dense_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--games-json", default="data/games_api.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--n-components", type=int, default=200)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    games_json = Path(args.games_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo documentos de {raw_dir}...")
    docs = load_documents(raw_dir)
    print(f"  {len(docs)} documento(s) carregado(s) de {raw_dir}.")

    print(f"Lendo jogos de {games_json}...")
    game_docs = load_games_json(games_json)
    docs.extend(game_docs)
    print(f"  {len(game_docs)} jogo(s) carregado(s) de {games_json}.")

    if not docs:
        print(
            f"Nenhum documento encontrado. Rode 'python scripts/fetch_games_api.py' "
            f"para gerar {games_json}, ou adicione arquivos .txt/.md/.pdf/.html em {raw_dir}."
        )
        sys.exit(1)

    chunks = build_chunks(docs)
    print(f"  {len(chunks)} chunk(s) gerado(s) (tamanho alvo: {CHUNK_SIZE_CHARS} caracteres).")
    if len(chunks) < 2:
        print("Poucos chunks para indexar. Adicione mais conteúdo à base.")
        sys.exit(1)

    print("Gerando embeddings (TF-IDF + SVD)...")
    vectorizer, svd, embeddings = build_index(chunks, n_components=args.n_components)
    print(f"  Dimensão dos embeddings: {embeddings.shape[1]}")

    print(f"Salvando índice em {out_dir}...")
    import pickle
    with open(out_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(out_dir / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    meta = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "chunk_size_chars": CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Concluído! Arquivos gerados:")
    for name in ["vectorizer.pkl", "svd.pkl", "embeddings.npy", "chunks.json", "meta.json"]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
