"""
Coleta dados de jogos pela API pública e gratuita da FreeToGame
(https://www.freetogame.com/api-doc) — não exige cadastro nem chave de API —
e salva um JSON estruturado em data/games_api.json, que o scripts/ingest.py
usa como fonte da base de conhecimento (no lugar dos artigos da Wikipédia).

A API cobre um catálogo de centenas de jogos free-to-play em vários gêneros
e plataformas (PC, navegador, PlayStation, etc.), com nome, descrição,
gênero, plataforma, desenvolvedora e publicadora.

Uso:
    python scripts/fetch_games_api.py
    python scripts/fetch_games_api.py --num-games 100 --platform pc --sort-by release-date
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://www.freetogame.com/api"


def fetch_game_list(platform: str, sort_by: str) -> list[dict]:
    params = {}
    if platform and platform != "all":
        params["platform"] = platform
    if sort_by:
        params["sort-by"] = sort_by
    resp = requests.get(f"{BASE_URL}/games", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_game_detail(game_id: int) -> dict | None:
    resp = requests.get(f"{BASE_URL}/game", params={"id": game_id}, timeout=20)
    if resp.status_code != 200:
        return None
    return resp.json()


def normalize(detail: dict) -> dict | None:
    description = (detail.get("description") or detail.get("short_description") or "").strip()
    name = detail.get("title")
    if not name or not description:
        return None
    return {
        "id": detail.get("id"),
        "name": name,
        "released": detail.get("release_date"),
        "description": description,
        "genres": [detail["genre"]] if detail.get("genre") else [],
        "platforms": [p.strip() for p in (detail.get("platform") or "").split(",") if p.strip()],
        "developers": [detail["developer"]] if detail.get("developer") else [],
        "publishers": [detail["publisher"]] if detail.get("publisher") else [],
        "metacritic": None,
        "esrb_rating": None,
        "website": detail.get("game_url") or detail.get("freetogame_profile_url"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-games", type=int, default=80, help="quantidade de jogos a buscar")
    parser.add_argument(
        "--platform",
        default="all",
        choices=["all", "pc", "browser"],
        help="filtro de plataforma da FreeToGame",
    )
    parser.add_argument(
        "--sort-by",
        default="popularity",
        choices=["popularity", "release-date", "alphabetical", "relevance"],
        help="critério de ordenação da lista",
    )
    parser.add_argument("--out", default="data/games_api.json")
    args = parser.parse_args()

    print(f"Buscando lista de jogos (plataforma={args.platform}, ordenação={args.sort_by})...")
    try:
        game_list = fetch_game_list(args.platform, args.sort_by)
    except requests.RequestException as exc:
        print(f"Falha ao acessar a API da FreeToGame: {exc}")
        sys.exit(1)

    game_ids = [g["id"] for g in game_list[: args.num_games]]
    print(f"  {len(game_ids)} jogo(s) selecionado(s). Buscando descrição completa de cada um...")

    games = []
    for i, game_id in enumerate(game_ids, 1):
        try:
            detail = fetch_game_detail(game_id)
        except requests.RequestException as exc:
            print(f"  [aviso] falhou id={game_id}: {exc}")
            continue
        if detail is None:
            continue
        normalized = normalize(detail)
        if normalized is None:
            continue
        games.append(normalized)
        print(f"  ({i}/{len(game_ids)}) {normalized['name']}")
        time.sleep(0.15)  # gentileza com a API

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)

    print(f"\nConcluído! {len(games)} jogo(s) salvos em {out_path}")
    print("Agora rode: python scripts/ingest.py   (para reconstruir o índice vetorial)")


if __name__ == "__main__":
    main()
