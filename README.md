# Chatbot RAG com LangGraph — Jogos Eletrônicos

Chatbot web com RAG (Retrieval-Augmented Generation), back-end em Python
(FastAPI), orquestração do fluxo com LangGraph, e deploy na Vercel.

**Aplicação publicada:** https://atividade-taia.vercel.app/

## Base de conhecimento

- **Domínio:** jogos eletrônicos (games) — consoles, franquias, mecânicas de
  jogo, história dos videogames e esports.
- **Fontes combinadas** (94 documentos, 221 chunks no índice atual):
  - 14 artigos em português (cabeçalho `[Título: ... | Fonte: ...]` em cada
    arquivo de `data/raw/`), vindos da Wikipédia (Nintendo, PlayStation,
    Xbox, Nintendo Switch, esporte eletrônico, história dos jogos
    eletrônicos) e de wikis especializadas de jogos — Minecraft Wiki
    (`pt.minecraft.wiki`), GTA Wiki (`gta.fandom.com`) e Stardew Valley
    Wiki (`pt.stardewvalleywiki.com`) — com bem mais detalhe de mecânicas
    de jogo do que um artigo enciclopédico genérico.
  - 80 jogos coletados via API pública da FreeToGame
    (`scripts/fetch_games_api.py`, salvos em `data/games_api.json`), com
    nome, descrição, gênero, plataforma, desenvolvedora e publicadora de
    cada jogo.
- Os documentos brutos ficam em `data/raw/*.txt` e `data/games_api.json`.
  O índice vetorial pré-processado (chunks + embeddings, gerado por
  `scripts/ingest.py`) fica em `data/processed/` e é o que a aplicação usa
  em tempo de execução.

## Estrutura do repositório

```
app/
  main.py         # FastAPI (entrypoint: GET /, GET /api/health, POST /api/chat)
  graph.py        # grafo LangGraph com roteamento (classificação, reescrita, evidência, verificação)
  retrieval.py    # carrega o índice vetorial e faz a busca por similaridade
  llm.py          # cliente da LLM externa (Groq/OpenAI/DeepSeek/NVIDIA/HF/Gemini)
  prompts.py      # todos os prompts (v1 original + v2 refinados) e parsers das saídas JSON
  config.py       # variáveis de ambiente / configuração
scripts/
  ingest.py            # constrói o índice vetorial a partir de data/raw/ e data/games_api.json
  fetch_games_api.py   # coleta jogos da API pública da FreeToGame para data/games_api.json
  scrape_urls.py       # coleta páginas web (data/urls.txt) para data/raw/
  preview_prompts.py   # mostra cada prompt montado para uma pergunta (sem gastar tokens)
  test_graph.py        # testa todos os caminhos do grafo com LLM simulada (sem gastar tokens)
data/
  raw/            # documentos brutos da base de conhecimento (Wikipédia + wikis de jogos)
  games_api.json  # jogos coletados via API (gerado por scripts/fetch_games_api.py)
  processed/      # índice vetorial pré-computado (usado em runtime)
public/
  index.html      # interface web do chatbot
vercel.json               # configuração da função Python na Vercel
requirements.txt          # dependências de runtime
requirements-ingest.txt   # dependências extras dos scripts de coleta/ingestão
.env.example              # variáveis de ambiente necessárias
```

## Prompts (Engenharia de Prompt)

Todos os prompts ficam em `app/prompts.py`. Na versão original (v1) havia um
único prompt, que misturava regras e chunks no system prompt, sem
delimitadores. Na v2, cada etapa do grafo tem um prompt com uma única
responsabilidade, e a saída de uma etapa alimenta a próxima (prompt chaining).

| Prompt | Nó do grafo | Responsabilidade | Saída |
| --- | --- | --- | --- |
| `PROMPT_V1_SYSTEM` | — (não usado no grafo) | Prompt original, mantido sem alteração para a comparação | texto |
| `CLASSIFY_SYSTEM` | `classify_question` | Classificar a mensagem em `JOGOS`, `FORA_DO_DOMINIO` ou `SAUDACAO` | JSON `{categoria, confianca}` |
| `REWRITE_SYSTEM` | `rewrite_question` | Transformar pergunta de continuação em pergunta autônoma para a busca | 1 linha de texto |
| `ANSWER_SYSTEM_V2` | `generate_answer` | Responder só com base nos trechos, citando as fontes | texto com `[n]` + linha `Fontes:` |
| `VERIFY_SYSTEM` | `verify_answer` | Checar se a resposta está sustentada pelo contexto | JSON `{sustentada, motivo}` |

Técnicas aplicadas nos prompts v2:

- **Estrutura fixa** no system prompt: Papel, Tarefa, Regras e Formato de saída.
- **Separação entre instruções e dados**: as regras ficam no system prompt; o
  contexto e a pergunta vão no user prompt, dentro de `<contexto>`,
  `<trecho id fonte>`, `<pergunta>`, `<historico>` e `<mensagem>`.
- **Contexto tratado como dado**: os prompts dizem explicitamente que nada
  dentro das tags é instrução. Além disso, `escape_tags()` neutraliza essas
  tags no texto dos documentos e do usuário, para que um chunk não consiga
  "fechar" o `<contexto>` e abrir uma região falsa.
- **Sem evidência**: a resposta padrão é *"Não encontrei essa informação na
  base consultada."*, e `is_sem_evidencia()` permite ao grafo detectá-la.
- **Zero-shot x few-shot** no classificador (`FEW_SHOT=0/1`, padrão 1). O few-shot
  inclui 7 exemplos, entre eles casos de fronteira (esporte físico,
  continuação de conversa) e duas tentativas de injection.
- **Saídas JSON tratadas como dado**: `normalize_classificacao()` e
  `normalize_verificacao()` validam a saída contra o formato combinado e caem
  em um valor padrão se o JSON vier inválido.
- **Lembrete ao final** do user prompt da resposta, repetindo a regra de usar
  só o contexto (técnica *sandwich*).

Para ver os prompts montados sem chamar a LLM:

```bash
python scripts/preview_prompts.py "Quem criou o Minecraft?"
python scripts/preview_prompts.py "E quem publicou ele?" --historico "Quando saiu o GTA V?"
python scripts/preview_prompts.py "Oi, tudo bem?" --llm   # classificador zero-shot x few-shot (usa a API)
```

## Grafo (LangGraph)

O fluxo é controlado por um grafo com decisões (`app/graph.py`). Cada nó
reutiliza os prompts, normalizadores e mensagens padrão de `app/prompts.py`.

```mermaid
graph TD;
    start([início]) --> receive_question
    receive_question --> classify_question
    classify_question -. SAUDACAO .-> greeting
    classify_question -. FORA_DO_DOMINIO .-> out_of_domain
    classify_question -. JOGOS .-> rewrite_question
    rewrite_question --> retrieve_context
    retrieve_context -. sem evidência .-> no_evidence
    retrieve_context -. com evidência .-> generate_answer
    generate_answer -. "não encontrei" .-> no_evidence
    generate_answer -. resposta .-> verify_answer
    verify_answer -. sustentada .-> finalize
    verify_answer -. não sustentada .-> invalid_answer
    greeting --> finalize
    out_of_domain --> finalize
    no_evidence --> finalize
    invalid_answer --> finalize
    finalize --> fim([fim])
```

| Nó | Função |
| --- | --- |
| `receive_question` | Normaliza a entrada |
| `classify_question` | Classifica em `JOGOS`, `FORA_DO_DOMINIO` ou `SAUDACAO` (`build_classify_messages`) |
| `greeting` / `out_of_domain` | Respostas diretas (`MSG_SAUDACAO`, `MSG_FORA_DO_DOMINIO`) |
| `rewrite_question` | Torna a pergunta autônoma para a busca; sem histórico, não chama a LLM (`needs_rewrite`) |
| `retrieve_context` | Busca os chunks com a pergunta reformulada (`retrieve()`) |
| `no_evidence` | Resposta `MSG_SEM_EVIDENCIA`, sem fontes |
| `generate_answer` | Gera a resposta só com os chunks (`build_answer_messages`) |
| `verify_answer` | Verifica se a resposta é sustentada pelo contexto (`build_verify_messages`) |
| `invalid_answer` | Substitui a resposta por `MSG_NAO_SUSTENTADA`, sem fontes |
| `finalize` | Nó de saída (`answer` + `sources`) |

**Critério de evidência.** `retrieve()` sempre devolve os top-k chunks, e os
scores do índice (TF-IDF + SVD) não separam bem "está na base" de "não está":
uma pergunta sobre um jogo ausente da base pontua quase como uma presente.
Por isso há duas camadas:

1. **Piso de similaridade** (`MIN_RETRIEVAL_SCORE`, padrão `0.1`): descarta
   buscas sem nenhuma sobreposição de vocabulário (score `0.000` nos testes;
   as perguntas válidas testadas ficaram acima de `0.46`).
2. **Decisão da LLM de geração**: se ela responde `MSG_SEM_EVIDENCIA`
   (detectado por `is_sem_evidencia()`), o grafo vai para `no_evidence` sem
   gastar a etapa de verificação.

**Testes dos caminhos** (sem gastar tokens; LLM simulada, retrieval real):

```bash
python scripts/test_graph.py
```

Cobre: saudação, fora do domínio, pergunta com evidência e sustentada,
sem evidência (piso de score e resposta "não encontrei"), resposta não
sustentada e pergunta de continuação com histórico. Para o mesmo roteiro
com a LLM real (apenas mostra os caminhos): `python scripts/test_graph.py --live`.

## Dependências

- **Runtime** (`requirements.txt`): fastapi, pydantic, langgraph,
  scikit-learn, numpy, scipy, requests, python-dotenv.
- **Scripts de coleta/ingestão** (`requirements-ingest.txt`, uso local):
  as de runtime + beautifulsoup4, pypdf, uvicorn.

## Instruções de execução (local)

> Use **Python 3.12** (ou 3.11/3.13). Com Python 3.14 as versões fixadas de
> numpy/scipy/scikit-learn não têm pacote pronto e a instalação falha.
> Com `uv`: `pip install uv && uv venv --python 3.12 .venv`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-ingest.txt   # inclui o uvicorn

cp .env.example .env
# edite o .env e coloque a LLM_API_KEY (ex.: chave gratuita da Groq)

uvicorn app.main:app --reload
# abra http://localhost:8000
```

O índice vetorial em `data/processed/` já vem pronto no repositório. Só é
necessário rodar `python scripts/ingest.py` de novo se o conteúdo de
`data/raw/` ou `data/games_api.json` for alterado. Para atualizar o
catálogo de jogos coletado via API, rode `python scripts/fetch_games_api.py`
(não exige chave nem cadastro) e em seguida `python scripts/ingest.py`.

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