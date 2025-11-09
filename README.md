# Course TA Agent Studio

An end‑to‑end, multi‑tenant system for **creating, hosting and chatting with course‑specific Teaching Assistant agents**.
Agents are created via a simple UI, connect to a **Google Drive folder** (syllabus, slides, homeworks…), ingest & index
the content for Retrieval‑Augmented Generation (RAG), and **auto‑refresh daily**. Each user has isolated login, memory,
and conversation history. Responses stream token‑by‑token.

> This project is intended as a course project submission. It’s production‑grade enough to deploy on a single VM or Docker.
> It uses FastAPI + SQLite + Chroma + Google Drive API + OpenAI/Gemini providers.

## Highlights

- UI to **Create Agent** (name, persona, model, Google Drive folder). Generates a shareable link like `/a/{slug}`
- **Google Drive ingestion** (recursive) with MIME‑aware parsing; builds a per‑agent vector index
- **Daily refresher** using APScheduler (re‑ingests changed files; removes deleted ones)
- Multi‑tenant auth (email+password), **per‑user memory** & conversation stores
- **Streaming** chat via Server‑Sent Events (SSE)
- Switchable providers: **OpenAI** or **Gemini** (configure model names in UI)
- Edit agent settings after creation

## Quickstart

1) **Environment**

Create `.env` at repo root (or set env vars):
```
SECRET_KEY=change-me
OPENAI_API_KEY=sk-...            # if using OpenAI
GOOGLE_API_CLIENT_ID=...         # OAuth client for Installed app or Web
GOOGLE_API_CLIENT_SECRET=...
GEMINI_API_KEY=...               # if using Gemini
EMBED_MODEL=openai:text-embedding-3-small   # or gemini:text-embedding-004 or local:all-MiniLM-L6-v2
SCHED_CRON=0 3 * * *            # 03:00 daily refresh (server timezone)
BASE_URL=http://localhost:8000
```

2) **Python**

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) **Run**

```
uvicorn app.main:app --reload
```

Visit `http://localhost:8000`

## Google Drive Setup

- Create a Google Cloud project and enable **Drive API**.
- Create **OAuth Client ID** (Desktop or Web). Download `client_secret.json` if Desktop.
- First time you connect a Drive folder for an agent, you’ll be redirected to Google consent to authorize Drive read access.
- Put the **folder URL** or **folder ID** into the Agent form. We’ll index its files (recursively).

## Deploy

Use Docker with the included `docker-compose.yml` or deploy on Render/Fly.io. Set env vars and mount persistent volumes for `data/`.

## Architecture

- **FastAPI** for API/UI (Jinja templates)
- **SQLite** for users/agents/messages
- **Chroma** for embeddings per agent (`data/chroma/{agent_id}`)
- **APScheduler** background daily refresher
- **LangChain GoogleDriveLoader** to fetch Drive docs, with extra file‑type parsers

## Inspirations / Prior Art

- LangChain Google Drive loader & retriever docs (Drive connectors & loaders)  
- LlamaIndex Google readers and **live** ingestion patterns (incremental updates)  
- Flowise & Langflow (visual agent builders supporting Drive loaders)  
- FastAPI SSE streaming patterns  
- FastAPI + APScheduler background jobs  
See the main page footer for links.
