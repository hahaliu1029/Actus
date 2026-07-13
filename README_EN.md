<p align="center">
  <h1 align="center">Actus</h1>
  <p align="center">
    A self-hosted AI Agent platform for planning, reasoning, execution, and human takeover
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Next.js-16-black.svg" alt="Next.js">
  </p>
  <p align="center">
    English · <a href="README.md">中文</a>
  </p>
</p>

---

## Overview

Actus is organized around three runtimes:

- `api/`: FastAPI backend for sessions, agents, auth, files, settings, skills, and sandbox orchestration
- `ui/`: Next.js 16 frontend for chat, task progress, workbench, settings, and admin views
- `sandbox/`: per-session Docker sandbox with Shell, filesystem access, Chromium, and VNC/noVNC

The default execution model is a **LangGraph**-based `Planner + ReAct` flow: Actus uses a two-layer state machine (main_graph for planning/coordination + react_graph for tool execution loops) while streaming plan updates, tool calls, messages, and takeover events back to the client.

## Core Capabilities

- **LangGraph agent orchestration** — two-layer graph architecture (main_graph for planning + react_graph for tool execution loops), with a `FINISHING` intermediate state for asynchronous wrap-up (final summary, attachment archival, unread-count refresh)
- **LangChain tool system** — file, shell, browser, search tools registered via `@tool` decorators
- **MCP / A2A / Skill integrations** managed as first-class agent tools, with progressive MCP tool discovery and embedding-based skill selection
- **Skill v2 filesystem storage** under `/app/data/skills`, supporting GitHub, local directory, and SKILL.md format installation
- **Multimodal file understanding** — audio transcription (Whisper API / sandbox faster-whisper), PDF parsing (native / pymupdf4llm), image processing, video keyframe extraction + vision model analysis
- **Context overflow management** — two-level gradual compaction (85% LLM summarization / 95% hard truncation) + synchronous 3-phase trimming for automatic context window protection
- **Modular prompt system (B5)** — composable subpackage of sections / bundles / reminders / assembler / budget, with bilingual (en/zh) bundle dispatch and contextual reminders
- **Agent memory subsystem** — `memory_search` / `memory_get` tools backed by a retrieval pipeline (cosine similarity → temporal decay → MMR diversity rerank) and an embedding circuit breaker
- **Tool approval & confirmation** — user-scoped persistent allow/deny rules (`always_allow` / `always_deny`, matched by command + dir glob) + session-scoped allow cache (approve only) + LLM-assisted Smart Approve (independent risk evaluation by the summary LLM, returns approve/deny/escalate) + explicit confirmation (frontend approve supports once/session/always scopes, deny is one-shot), with risk assessment and a persistent audit log
- **Session event recovery** — Redis Stream backed SSE state resume; reconnecting clients pick up from the last event id instead of losing in-flight tool/message events
- **Execution health monitoring** — step-level watchdog, execution metrics, tool failure tracker, and a uniform JSON envelope for tool I/O
- **LLM call budgets** — independent connect-phase / read-phase timeouts wired through every adapter clone path, aligned with LangGraph RetryPolicy to avoid 9× HTTP retry amplification
- **Human takeover** for both `shell` and `browser` scopes
- **Workbench UI** with terminal preview, browser preview, VNC, timeline scrubbing, and file preview
- **Streaming transport** via SSE and WebSocket
- **Containerized sandbox** with Chromium, Xvfb, x11vnc, and websockify
- **Attachment storage** through MinIO / S3-compatible object storage with transfer progress tracking
- **JWT auth, admin tools, tool preferences, and runtime settings**
- **SSH tunnel** — optional autossh reverse tunnel to expose the local API to a cloud server
- **Language plumbing** — `Message.language` propagates end-to-end so the prompt assembler dispatches the correct (en/zh) bundle

## Architecture

![Actus Architecture](architecture.png)

<details>
<summary>ASCII version</summary>

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ UI (Next.js) │────▶│ API (FastAPI)│────▶│ PostgreSQL   │
└──────────────┘     │              │     └──────────────┘
                     │ Agent / Auth │────▶│ Redis        │
                     │ Files / Skill│     └──────────────┘
                     │ Settings     │────▶│ MinIO / S3   │
                     │ Sandbox Ctrl │     └──────────────┘
                     └──────┬───────┘
                            │ Docker
                            ▼
                     ┌──────────────┐
                     │ Sandbox      │
                     │ Shell/File   │
                     │ Chromium/VNC │
                     └──────────────┘

(Optional) Phone ──HTTP──▶ Cloud:18082 ──SSH Tunnel──▶ API:8000
```
</details>

For backend layering and runtime composition, see [项目架构.md](项目架构.md). Source: [architecture.excalidraw](architecture.excalidraw).

## Docker Compose Quick Start

### Requirements

- Docker Engine + Docker Compose v2
- At least 6 GB RAM available to Docker
- A valid LLM API key

### Start the stack

```bash
git clone https://github.com/hahaliu1029/Actus.git
cd Actus

cp .env.example .env
# Edit .env and provide at least:
# POSTGRES_PASSWORD
# JWT_SECRET_KEY
# MINIO_ACCESS_KEY
# MINIO_SECRET_KEY
# NEXT_PUBLIC_API_BASE_URL

docker compose --env-file .env up -d --build

# Optional: create a super admin
docker compose exec api python scripts/create_super_admin.py
```

After startup:

- UI: `http://localhost`
- API docs: `http://localhost:8000/docs`

The standard Docker Compose stack is a local-development topology. It starts a
loopback-only local MinIO and idempotently creates the `a2a-mcp` bucket before
the API starts. The S3 API is `http://127.0.0.1:9000`, the console is
`http://127.0.0.1:9001`, and credentials come from `MINIO_ACCESS_KEY` /
`MINIO_SECRET_KEY`. Changing `MINIO_API_PORT` automatically changes the public
endpoint to `localhost:<port>`; advanced deployments can override
`MINIO_PUBLIC_ENDPOINT` / `MINIO_PUBLIC_SECURE`. A root `.env` value for
`MINIO_ENDPOINT` does not control the standard Docker Compose stack, whose API
always uses `minio:9000` internally.

Switching from remote storage to this local MinIO creates a new empty data set;
there is no automatic migration, and existing attachments remain in the remote
S3 service. The pinned archived MinIO release image is not a production baseline.
Production deployments and remote URL consumers should use a deployer-managed
remote S3 service or a protected TLS endpoint. The existing `tunnel` profile
forwards only the API, not local MinIO; MCP and other direct remote URL fetchers
need a reachable remote/public endpoint.

### Runtime config notes

- In container mode, the backend uses `/app/data/config.yaml` inside the `api-data` volume
- If the file does not exist, the backend creates one from code defaults
- The recommended way to configure LLM, MCP, A2A, and Skill policies is through the frontend settings UI
- `api/config.yaml.example` is mainly for **local backend development** or manual pre-seeding

### Rebuild sandbox code correctly

The Compose service name is `sandbox-image`, not `sandbox`. After changing files under `sandbox/`, use:

```bash
docker compose --env-file .env build sandbox-image api
docker compose --env-file .env up -d --force-recreate api

# Optional: remove old temporary sandbox containers
docker ps --format '{{.Names}}' | grep '^actus-sb-' | xargs -r docker rm -f
```

## Local Development

### Frontend

```bash
cd ui
npm install
npm run dev
```

### Backend

Local backend development does **not** use the same variables as the root Compose `.env`. `api/core/config.py` reads runtime settings from `api/.env`, for example:

The remote S3 example below is for a host-run API or custom orchestration only;
it is not used by standard Docker Compose.

```bash
cd api
cp config.yaml.example config.yaml

cat > .env <<'EOF'
ENV=development
LOG_LEVEL=INFO
APP_CONFIG_FILEPATH=config.yaml
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/manus
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
MINIO_ENDPOINT=s3.example.com
MINIO_PUBLIC_ENDPOINT=s3.example.com
MINIO_ACCESS_KEY=replace-me
MINIO_SECRET_KEY=replace-me
MINIO_REGION=us-east-1
MINIO_SECURE=true
MINIO_PUBLIC_SECURE=true
MINIO_BUCKET_NAME=replace-me
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
EOF

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

You will also need PostgreSQL, Redis, a built `sandbox-image`, and an accessible
remote S3 / MinIO bucket for this host-run configuration.

## Tests

```bash
# Backend
cd api
pytest

# Frontend
cd ui
npm run test
```

## Repository Layout

```text
Actus/
├── api/                  # FastAPI backend
│   ├── app/
│   │   ├── application/  # Use case services (Skill, Memory Flush)
│   │   ├── domain/       # Models, flows, tools, prompts, context management
│   │   ├── infrastructure/ # DB/storage/external impls, file processors, embedding
│   │   └── interfaces/   # Routes, schemas, dependency injection
│   ├── core/             # Config, security
│   ├── scripts/          # Admin scripts
│   └── tests/            # Backend tests
├── ui/                   # Next.js frontend
├── sandbox/              # Docker sandbox source
├── tunnel/               # SSH reverse tunnel config (optional)
├── docker-compose.yml    # Compose stack
├── DEPLOY.md             # Deployment guide
├── api.md                # English API reference
├── api_zhcn.md           # Chinese API reference
└── 项目架构.md             # Architecture notes
```

## Documentation

- [Deployment Guide](DEPLOY.md)
- [English API Reference](api.md)
- [中文 API 文档](api_zhcn.md)
- [Architecture Notes](项目架构.md)
- [API README](api/README.md)
- [UI README](ui/README.md)
- [Sandbox README](sandbox/README.md)
- [SSH Tunnel](tunnel/README.md)
- [Contributing](CONTRIBUTING.md)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 17, SQLAlchemy 2.0 async, Alembic |
| Cache / Rate limit | Redis |
| Object storage | MinIO / S3-compatible |
| Agent | LangGraph StateGraph, LangChain BaseChatModel, PlannerReActFlow |
| Context management | TokenEstimator, ContextAssembler, GradualCompactor |
| File understanding | Whisper (OpenAI/sandbox), pymupdf4llm, vision model frame analysis |
| Embedding | OpenAI Embeddings, Redis cache, numpy vector index |
| Extension protocols | MCP (with progressive discovery), A2A, Skill (with SKILL.md format) |
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Zustand |
| Browser execution | Chromium, CDP, Playwright-style DOM operations |
| Sandbox | Docker, Supervisor, Xvfb, x11vnc, websockify |
| Testing | pytest, Vitest, Testing Library |

## License

Actus is released under the [Apache License 2.0](LICENSE).
