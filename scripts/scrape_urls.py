"""
Coleta páginas web listadas em data/urls.txt (uma URL por linha, linhas
começando com # são ignoradas) e salva o texto extraído em data/raw/,
pronto para o script scripts/ingest.py processar.

Use isto para expandir a base de conhecimento com o domínio real escolhido
pela equipe (ex.: documentação técnica, páginas de turismo, notícias de um
tema específico etc.), rodando localmente na máquina de vocês (que tem
acesso normal à internet, diferente do ambiente usado para gerar a base de
exemplo deste repositório).

Uso:
    pip install -r requirements-ingest.txt
    python scripts/scrape_urls.py --urls-file data/urls.txt --out-dir data/raw
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RagChatbotBot/1.0; +https://example.com)"
}


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:80] or "pagina"


def fetch_and_extract(url: str, timeout: int = 20) -> tuple[str, str]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "aside", "form"]):
        tag.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path

    main = soup.find("main") or soup.find("article") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all(["p", "li", "h2", "h3"])]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    body = "\n\n".join(paragraphs)
    return title, body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", default="data/urls.txt")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--delay", type=float, default=1.0, help="segundos entre requisições")
    args = parser.parse_args()

    urls_file = Path(args.urls_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not urls_file.exists():
        print(f"Arquivo {urls_file} não encontrado. Crie um com uma URL por linha.")
        return

    urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"{len(urls)} URL(s) para coletar.")

    ok, failed = 0, 0
    for i, url in enumerate(urls, start=1):
        try:
            title, body = fetch_and_extract(url)
            if len(body) < 200:
                print(f"[{i}/{len(urls)}] aviso: pouco conteúdo extraído de {url}")
            filename = f"{i:03d}_{slugify(title)}.txt"
            (out_dir / filename).write_text(
                f"[Título: {title} | Fonte: {url}]\n\n{body}", encoding="utf-8"
            )
            print(f"[{i}/{len(urls)}] ok: {url} -> {filename}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(urls)}] erro em {url}: {exc}")
            failed += 1
        time.sleep(args.delay)

    print(f"\nConcluído. {ok} página(s) salva(s), {failed} falha(s).")
    print("Rode agora: python scripts/ingest.py")


if __name__ == "__main__":
    main()
