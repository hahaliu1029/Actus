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

Actus is organized around three application runtimes:

- `api/`: FastAPI backend for sessions, agents, auth, files, settings, extension governance, and sandbox orchestration
- `ui/`: Next.js 16 frontend for chat, task progress, workbench, settings, and admin views
- `sandbox/`: when enabled, a per-session Docker runtime with Shell, filesystem access, Chromium, and VNC/noVNC

The default execution model is a **LangGraph**-based `Planner + ReAct` flow: Actus uses a two-layer state machine (main_graph for planning/coordination + react_graph for tool execution loops) while streaming plan updates, tool calls, messages, and takeover events back to the client.

## Core Capabilities

- **LangGraph agent orchestration** — two-layer graph architecture (main_graph for planning + react_graph for tool execution loops), with a `FINISHING` intermediate state for asynchronous wrap-up (final summary, attachment archival, unread-count refresh)
- **LangChain tool system** — file, shell, browser, search tools registered via `@tool` decorators
- **Unified extension runtime and governance** — MCP / A2A / Skill / Plugin entries share the extension overview; MCP/A2A/Skill support global and user enablement while Plugin has a separate parent-level switch; health probes, call statistics, and optional `off` / `shadow` / `enforce` governance add install preflight, provenance and content pins, quarantine, reapproval, and audit
- **Plugin bundle installation** — `plugin.json` may declare Skill, MCP, and A2A members; installation supports a redacted dry-run preview and a compensating, startup-recoverable install/uninstall saga
- **Skill v2 filesystem storage** under `/app/data/skills`, supporting GitHub, local directory, and SKILL.md format installation
- **Multimodal file understanding** — audio transcription (Whisper API / sandbox faster-whisper), PDF parsing (native / pymupdf4llm), image processing, video keyframe extraction + vision model analysis
- **Context overflow management** — two-level gradual compaction (85% LLM summarization / 95% hard truncation) + synchronous 3-phase trimming for automatic context window protection
- **Modular prompt system (B5)** — composable subpackage of sections / bundles / reminders / assembler / budget, with bilingual (en/zh) bundle dispatch and contextual reminders
- **Agent memory subsystem (M1 Memory Redesign)** — three categories (`user` / `rule` / `fact`), `memory_search` / `memory_get` / `memory_save` tools, cosine similarity → temporal decay → MMR reranking, and an embedding circuit breaker; files are the source of truth, mounted read-only into sandboxes, with `FsReconciler`, an LLM quality gate, per-user daily caps, and system notifications
- **Permission Engine tool approval** — native / MCP / A2A / Skill calls share one decision pipeline; users can set per-tool `auto` / `ask` / `deny` preferences, explicit confirmation can create session or always grants, Smart Approve falls through to a human on timeout or infrastructure failure, and decisions retain a durable audit trail
- **Session event recovery** — Redis Stream backed SSE state resume; reconnecting clients pick up from the last event id instead of losing in-flight tool/message events
- **Execution health monitoring** — step-level watchdog, execution metrics, tool failure tracker, and a uniform JSON envelope for tool I/O
- **LLM call budgets** — independent connect-phase / read-phase timeouts wired through every adapter clone path, aligned with LangGraph RetryPolicy to avoid 9× HTTP retry amplification
- **Human takeover** for `shell` and `browser`, including request, lease renewal, completion, and recovery flows
- **Workbench UI** with terminal preview, browser preview, VNC, timeline scrubbing, and file preview
- **Streaming interaction** — session lists and conversations use SSE; takeover terminals and VNC use WebSocket
- **Three sandbox provision modes** — `always` provisions at task start, `on_demand` waits for the first sandbox access, and `off` removes the sandbox tools, takeover, and container-provisioning surface
- **Containerized sandbox** — when enabled, each session uses an isolated Docker container with Chromium, Xvfb, x11vnc, and websockify
- **Object storage and attachments** — the standard Compose stack starts a loopback-only MinIO by default and also supports remote S3-compatible storage; uploads are session-linked and transfers support progress tracking and resume
- **Users and administration** — JWT auth, super admins, user management, tool preferences, and runtime settings
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

For backend layering and runtime composition, see [项目架构.md](项目架构.md). Diagram sources:
[architecture.mmd](architecture.mmd) / [architecture.excalidraw](architecture.excalidraw).

## Docker Compose Quick Start

### Requirements

- Docker Engine + Docker Compose v2
  - `off` deployments require Docker Compose >= 2.24.4 for the `!override` YAML tag; see `docs/runbooks/sandbox-off-runbook.md`
- At least 6 GB RAM available to Docker
- A valid LLM API key, entered after startup in Settings or pre-seeded in runtime config

### Start the stack

```bash
git clone https://github.com/hahaliu1029/Actus.git
cd Actus

cp .env.example .env
cp api/config.yaml.example api/config.yaml
# Edit .env and provide at least:
# POSTGRES_PASSWORD
# JWT_SECRET_KEY
# MINIO_ACCESS_KEY
# MINIO_SECRET_KEY
# MEMORY_ROOT_HOST=/absolute/path/to/actus-memory
# NEXT_PUBLIC_API_BASE_URL
# Unless you are explicitly validating coordinator, set ACTUS_C2_COORDINATOR_ENABLED=false;
# .env.example currently keeps it true for CI/evaluation, while production rollout gates remain open
# Optional: set PYTHON_PACKAGE_INDEX_URL to override the Python package index

# First Memory deployment:
# Set MEMORY_ROOT_HOST in .env to an absolute host path, then create it.
mkdir -p /absolute/path/to/actus-memory
# If ACTUS_UID non-root mode is enabled, also chown that directory.

docker compose --env-file .env up -d --build

# Optional: create a super admin
docker compose exec api python scripts/create_super_admin.py

# Optional: manually reconcile Memory filesystem/DB consistency.
# docker compose exec api python -m app.cli.memory_reconcile
```

After startup:

- UI: `http://localhost` (default `UI_PORT=80`)
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

- Standard Compose bind-mounts host-side `api/config.yaml` at `/app/data/config.yaml` in the container
- Before the first start, initialize it from `api/config.yaml.example` as shown in the quick start
- The recommended way to configure models and extensions is through `Settings -> Model Providers / Extension Overview / MCP Servers / A2A Agents / Skill Ecosystem`
- `api/config.yaml.example` is mainly for **local backend development** or manual pre-seeding

### Sandbox provision modes

`SANDBOX_PROVISION_MODE` is an environment-only deployment switch that takes effect after the API
container is recreated (`docker compose restart api` does not apply changed environment values):

- `always` (default): acquire or create the session sandbox when a task starts
- `on_demand`: create it only on the first sandbox tool, VNC, takeover, or coordinator parent-sandbox I/O; pure-chat sessions create no container
- `off`: do not register sandbox or Skill-creation tools, reject takeover/VNC, and guarantee no container creation through application paths

An `off` deployment must use [docker-compose.sandbox-off.yml](docker-compose.sandbox-off.yml)
to remove the `sandbox-image` dependency and Docker socket, after disabling all three coordinator
flags. See the [sandbox off runbook](docs/runbooks/sandbox-off-runbook.md) for the required drain,
deployment, verification, and rollback commands. `SANDBOX_PROVISION_TIMEOUT_SECONDS` is the
combined create, ready, and post-provision hook budget for `on_demand` only.

### Extension governance modes

`EXTENSION_GOVERNANCE_MODE` is also an environment-only deployment switch that requires recreating
the API container (do not use `docker compose restart api`, which keeps old environment values):

- `off` (default): no governance registry or Plugin pipeline; existing MCP / A2A / Skill config paths retain their behavior
- `shadow`: record scans, observations, pins, and audit events; detection failures are fail-open, while quarantine, disable, delete, and parent-Plugin blocks still apply
- `enforce`: fail closed for unpinned extensions, pin mismatches, or unavailable governance storage

When governance is enabled, admins can refresh observations, approve pins, quarantine, reapprove,
governance-enable/disable, and install/uninstall Plugins from Extension Overview. The frontend install
flow starts with a redacted dry run; API callers explicitly choose dry-run or commit on the same
endpoint. In `shadow`, caution/dangerous detection results warn but do not require acknowledgement;
in `enforce`, caution requires `acknowledge` and dangerous requires `force`. An MCP/A2A member probe
failure requires `force` in either enabled governance mode.

Do not jump directly from `off` to `enforce`: recreate the API in `shadow`, refresh observations,
review audit output, approve the pins you intend to run, and only then enter `enforce`. Switching back
to `off` is not “keep current enforcement without recording”: the Plugin management surface and parent
`parent_blocked` projection disappear, while materialized members continue under their MCP/A2A/Skill
configuration. See the [API reference](api.md#plugin-v2plugins) for the full boundary.

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

The `api/.env` example below is for a host-run API or custom orchestration only;
standard Compose overrides the API's internal `MINIO_ENDPOINT` with `minio:9000`.

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
MINIO_ENDPOINT=localhost:9000
MINIO_PUBLIC_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_REGION=us-east-1
MINIO_SECURE=false
MINIO_PUBLIC_SECURE=false
MINIO_BUCKET_NAME=a2a-mcp
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
EOF

cd ..
uv sync
cd api
uv run bash dev.sh
```

You will also need PostgreSQL, Redis, a built `sandbox-image`, and either the local MinIO
from the standard Compose stack or an accessible remote S3-compatible bucket.

See [api/README.md](api/README.md) for details.

## Tests

```bash
# Backend
cd api
uv run pytest

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
| Extension protocols | MCP (with progressive discovery), A2A, Skill (with SKILL.md format), Plugin bundles and extension governance |
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Zustand |
| Browser execution | Chromium, CDP, Playwright-style DOM operations |
| Sandbox | Docker, Supervisor, Xvfb, x11vnc, websockify; `always` / `on_demand` / `off` provisioning |
| Testing | pytest, Vitest, Testing Library |

## License

Actus is released under the [Apache License 2.0](LICENSE).
