# Chatbot RAG com LangGraph — Jogos Eletrônicos

Chatbot web com **RAG (Retrieval-Augmented Generation)**, back-end em Python
(FastAPI) e orquestração do fluxo com **LangGraph**, pronto para deploy na
**Vercel**.

## Base de conhecimento

- **Domínio escolhido:** jogos eletrônicos (games) — consoles, franquias,
  história dos videogames e esports.
- **Fontes:** 13 artigos da Wikipédia em português (ver cabeçalho `[Fonte: ...]`
  em cada arquivo de `data/raw/`), cobrindo Nintendo, PlayStation, Xbox,
  Nintendo Switch, Super Mario, The Legend of Zelda, Minecraft, Grand Theft
  Auto, League of Legends, esporte eletrônico e a história dos jogos
  eletrônicos.
- Os arquivos brutos ficam em `data/raw/*.txt`. O índice vetorial já
  pré-processado (chunks + embeddings) fica em `data/processed/` e é o que a
  aplicação de fato usa em tempo de execução.
- **Importante:** esta é uma base de **demonstração** para o app já vir
  funcional. Para a entrega da atividade, troque/expanda o conteúdo de
  `data/raw/` pelo domínio que a sua equipe escolher (veja "Como trocar/
  expandir a base de conhecimento" abaixo) e rode o script de ingestão de
  novo.

## Arquitetura / pipeline

```
data/raw/*.{txt,md,pdf,html}
        │  scripts/ingest.py (limpeza → chunks → TF-IDF+SVD → índice)
        ▼
data/processed/ (vectorizer.pkl, svd.pkl, embeddings.npy, chunks.json)
        │
        ▼
app/retrieval.py  ──►  busca por similaridade de cosseno
        │
        ▼
app/graph.py (LangGraph)
  entrada da pergunta → recuperação de contexto → montagem do prompt
  → chamada da LLM externa → retorno da resposta
        │
        ▼
app/main.py (FastAPI: POST /api/chat)
        │
        ▼
public/index.html (interface web do chat)
```

### Por que TF-IDF + SVD em vez de um modelo de embeddings neural?

O back-end roda como uma **Vercel Function em Python**, que tem limite de
bundle de 500MB e não deve baixar modelos pesados a cada cold start. TF-IDF +
SVD (uma forma clássica de *Latent Semantic Analysis*) gera embeddings
densos de verdade e um índice vetorial pesquisável por similaridade de
cosseno, usando só `scikit-learn`/`numpy` — poucos MBs, sem downloads, sem
GPU, sem custo de rede e com busca praticamente instantânea. Isso também
garante, por construção, que a LLM externa é usada **apenas na etapa de
geração da resposta**, como pede o enunciado (os embeddings nunca passam por
uma LLM). O código de `scripts/ingest.py` e `app/retrieval.py` está isolado
o bastante para trocar por outra estratégia de embeddings se a equipe preferir.

### LLM externa suportada

Configurável via variáveis de ambiente (`LLM_PROVIDER`), sem trocar código:
`groq`, `openai`, `deepseek`, `nvidia`, `huggingface` (todas com API
compatível com OpenAI Chat Completions) ou `gemini` (API própria do Google).
Veja `app/llm.py` e `.env.example`.

## Estrutura do repositório

```
app/
  main.py         # FastAPI (entrypoint da Vercel: GET /, GET /api/health, POST /api/chat)
  graph.py         # grafo LangGraph (RAG)
  retrieval.py     # carrega o índice vetorial e faz a busca por similaridade
  llm.py           # cliente da LLM externa (multi-provedor)
  config.py        # variáveis de ambiente / configuração
scripts/
  ingest.py        # constrói o índice vetorial a partir de data/raw/
  scrape_urls.py   # coleta páginas web (data/urls.txt) para data/raw/
data/
  raw/             # documentos brutos da base de conhecimento (.txt/.md/.pdf/.html)
  processed/       # índice vetorial pré-computado (usado em runtime)
  urls.txt         # lista de URLs para o scraper (opcional)
public/
  index.html       # interface web do chatbot (servida pela Vercel)
vercel.json        # configuração da função Python na Vercel
requirements.txt   # dependências de runtime (bundle da função)
requirements-ingest.txt  # dependências extras só para rodar os scripts localmente
.env.example       # variáveis de ambiente necessárias
```

## Rodando localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-ingest.txt   # inclui as deps de runtime + uvicorn/scraping

cp .env.example .env
# edite o .env e coloque sua LLM_API_KEY (ex.: uma chave gratuita da Groq)

# (opcional) o índice em data/processed/ já vem pronto neste repositório;
# só rode o ingest de novo se você alterar data/raw/:
python scripts/ingest.py

uvicorn app.main:app --reload
# abra http://localhost:8000
```

Ou, com a Vercel CLI (recomendado, simula o ambiente serverless):

```bash
npm i -g vercel
vercel dev
```

## Deploy na Vercel

1. Suba este repositório para o GitHub (ou GitLab/Bitbucket).
2. Em [vercel.com](https://vercel.com), clique em **Add New → Project** e
   importe o repositório. A Vercel detecta o Python automaticamente pelo
   `requirements.txt` e pelo entrypoint `app/main.py` (variável `app`).
3. Em **Environment Variables**, adicione pelo menos:
   - `LLM_PROVIDER` (ex.: `groq`)
   - `LLM_API_KEY` (sua chave da LLM escolhida)
   - opcionalmente `LLM_MODEL`, `KNOWLEDGE_DOMAIN`, `TOP_K`, etc. (veja `.env.example`)
4. Clique em **Deploy**. Ao final você recebe uma URL pública
   (`https://seu-projeto.vercel.app`) já com a interface do chat funcionando.

Ou via CLI, direto do terminal:

```bash
npm i -g vercel
vercel login
vercel --prod
```

> A Vercel usa o índice vetorial já commitado em `data/processed/` — não é
> necessário rodar nenhum passo de build extra para isso funcionar.

## Como trocar/expandir a base de conhecimento

1. Escolha o domínio da sua equipe (ex.: turismo, saúde, documentação técnica).
2. Junte o conteúdo:
   - Coloque arquivos `.pdf`, `.txt`, `.md` ou `.html` em `data/raw/`; **ou**
   - Liste URLs em `data/urls.txt` e rode (na sua máquina, com internet normal):
     ```bash
     pip install -r requirements-ingest.txt
     python scripts/scrape_urls.py
     ```
3. Reconstrua o índice vetorial:
   ```bash
   python scripts/ingest.py
   ```
4. Atualize `KNOWLEDGE_DOMAIN` no `.env` (e nas env vars da Vercel) para
   refletir o novo domínio — isso ajusta o prompt de sistema da LLM.
5. Commit dos arquivos atualizados em `data/raw/` e `data/processed/`, e
   redeploy (`git push` se o projeto estiver conectado ao GitHub, ou
   `vercel --prod`).

## Como o fluxo do LangGraph atende ao enunciado

| Etapa pedida pelo enunciado          | Implementação                                         |
|---------------------------------------|--------------------------------------------------------|
| Entrada da pergunta                   | nó `receive_question` (`app/graph.py`)                 |
| Recuperação de contexto               | nó `retrieve_context` → `app/retrieval.py`             |
| Montagem do prompt (contexto + histórico) | nó `build_prompt`                                   |
| Chamada da LLM                        | nó `generate_answer` → `app/llm.py`                    |
| Retorno da resposta                   | nó `finalize`                                          |

## Dependências

Veja `requirements.txt` (runtime, o que vai para a Vercel) e
`requirements-ingest.txt` (extra para rodar os scripts de coleta/ingestão
localmente).
