# Chatbot RAG com LangGraph — Jogos Eletrônicos

Chatbot web com RAG (Retrieval-Augmented Generation), back-end em Python
(FastAPI), orquestração do fluxo com LangGraph, e deploy na Vercel.

**Aplicação publicada:** https://atividade-taia.vercel.app/

## Base de conhecimento

- **Domínio:** jogos eletrônicos (games) — consoles, franquias, história dos
  videogames e esports.
- **Fontes:** 13 artigos da Wikipédia em português (cabeçalho `[Fonte: ...]`
  em cada arquivo de `data/raw/`), cobrindo Nintendo, PlayStation, Xbox,
  Nintendo Switch, Super Mario, The Legend of Zelda, Minecraft, Grand Theft
  Auto, League of Legends, esporte eletrônico e a história dos jogos
  eletrônicos.
- Os documentos brutos ficam em `data/raw/*.txt`. O índice vetorial
  pré-processado (chunks + embeddings, gerado por `scripts/ingest.py`) fica
  em `data/processed/` e é o que a aplicação usa em tempo de execução.

## Estrutura do repositório

```
app/
  main.py         # FastAPI (entrypoint: GET /, GET /api/health, POST /api/chat)
  graph.py        # grafo LangGraph: entrada → recuperação → prompt → LLM → resposta
  retrieval.py    # carrega o índice vetorial e faz a busca por similaridade
  llm.py          # cliente da LLM externa (Groq/OpenAI/DeepSeek/NVIDIA/HF/Gemini)
  config.py       # variáveis de ambiente / configuração
scripts/
  ingest.py       # constrói o índice vetorial a partir de data/raw/
  scrape_urls.py  # coleta páginas web (data/urls.txt) para data/raw/
data/
  raw/            # documentos brutos da base de conhecimento
  processed/      # índice vetorial pré-computado (usado em runtime)
public/
  index.html      # interface web do chatbot
vercel.json               # configuração da função Python na Vercel
requirements.txt          # dependências de runtime
requirements-ingest.txt   # dependências extras dos scripts de coleta/ingestão
.env.example              # variáveis de ambiente necessárias
```

## Dependências

- **Runtime** (`requirements.txt`): fastapi, pydantic, langgraph,
  scikit-learn, numpy, scipy, requests, python-dotenv.
- **Scripts de coleta/ingestão** (`requirements-ingest.txt`, uso local):
  as de runtime + beautifulsoup4, pypdf, uvicorn.

## Instruções de execução (local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-ingest.txt

cp .env.example .env
# edite o .env e coloque a LLM_API_KEY (ex.: chave gratuita da Groq)

uvicorn app.main:app --reload
# abra http://localhost:8000
```

O índice vetorial em `data/processed/` já vem pronto no repositório. Só é
necessário rodar `python scripts/ingest.py` de novo se o conteúdo de
`data/raw/` for alterado.

## Deploy na Vercel

1. Repositório conectado no GitHub.
2. Em [vercel.com](https://vercel.com) → **Add New → Project** → importar o
   repositório (a Vercel detecta o Python pelo `requirements.txt` e pelo
   entrypoint `app/main.py`).
3. Em **Environment Variables**, configurar:
   - `LLM_PROVIDER=groq`
   - `LLM_API_KEY`
   - `LLM_MODEL=qwen/qwen3.8-27b` (modelo usado na aplicação publicada; a
     lista de modelos disponíveis para a chave pode ser consultada em
     `https://api.groq.com/openai/v1/models`)
   - opcionalmente `KNOWLEDGE_DOMAIN`, `TOP_K` (ver `.env.example`)
4. **Deploy** → gera a URL pública com a interface do chat já funcionando.

Ou via CLI:
```bash
npm i -g vercel
vercel login
vercel --prod
```
