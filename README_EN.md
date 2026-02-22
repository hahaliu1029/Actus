<p align="center">
  <h1 align="center">Actus</h1>
  <p align="center">
    Open-source AI Agent Platform — Built on MCP Tool Protocol & A2A Agent-to-Agent Communication
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

## Features

- **ReAct Agent** — Reasoning + Acting pattern with multi-turn reasoning and tool calling
- **MCP Tool Protocol** — Dynamically connect external tool servers via [Model Context Protocol](https://modelcontextprotocol.io/)
- **A2A Protocol** — [Agent-to-Agent](https://google.github.io/A2A/) communication and collaboration
- **Planner + ReAct Flow** — Two-phase agent orchestration: plan first, then execute
- **Sandboxed Execution** — Docker-isolated code execution environment
- **Browser Automation** — Web interaction powered by Playwright + browser-use
- **Multi-model Support** — Compatible with OpenAI API format (DeepSeek, Kimi, etc.)
- **Streaming Output** — Real-time streaming responses with chain-of-thought display
- **File Management** — MinIO/S3-compatible object storage for file upload/download
- **User System** — JWT authentication with role-based access control
- **Modern Frontend** — Next.js 16 + React 19 + Tailwind CSS + Zustand

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   UI (Next.js)│────▶│  API (FastAPI)│────▶│  PostgreSQL  │
└──────────────┘     │              │     └──────────────┘
                     │   ┌────────┐ │     ┌──────────────┐
                     │   │ Agent  │ │────▶│    Redis      │
                     │   │ Engine │ │     └──────────────┘
                     │   └───┬────┘ │     ┌──────────────┐
                     │       │      │────▶│  MinIO / S3   │
                     │   ┌───▼────┐ │     └──────────────┘
                     │   │MCP/A2A │ │     ┌──────────────┐
                     │   │Tools   │ │────▶│  Sandbox      │
                     │   └────────┘ │     │  (Docker)     │
                     └──────────────┘     └──────────────┘
```

The backend follows **Clean Architecture + DDD** layered design.

## Quick Start

### Prerequisites

- Docker Engine + Docker Compose v2
- At least 6 GB memory available for Docker
- A MinIO / S3-compatible object storage service

### One-click Deployment

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Actus.git
cd Actus

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your database password, JWT secret, MinIO credentials, etc.

# 3. Configure Agent runtime parameters
cp api/config.yaml.example api/config.yaml
# Edit api/config.yaml with your LLM API key and MCP server configurations

# 4. Start all services
docker compose --env-file .env up -d --build

# 5. Create admin account (recommended)
docker compose exec api python scripts/create_super_admin.py
```

After startup, visit:

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Local Development

```bash
# Backend
cd api
pip install -r requirements.txt
bash dev.sh

# Frontend
cd ui
npm install
npm run dev
```

## Project Structure

```
Actus/
├── api/                 # Backend service (FastAPI)
│   ├── app/             # Application code (Clean Architecture layers)
│   │   ├── domain/      #   Domain layer (models, agents, tools, prompts)
│   │   ├── application/ #   Application layer (use case orchestration)
│   │   ├── infrastructure/ # Infrastructure layer (DB, storage, external services)
│   │   └── interfaces/  #   Interface layer (routes, schemas, DI)
│   ├── core/            # Core config & security
│   ├── alembic/         # Database migrations
│   ├── scripts/         # Admin scripts
│   └── tests/           # Tests
├── ui/                  # Frontend (Next.js)
├── sandbox/             # Sandbox service (code execution)
├── docker-compose.yml   # Container orchestration
└── DEPLOY.md            # Deployment guide
```

## Documentation

- [Deployment Guide](DEPLOY.md) — Docker Compose deployment steps
- [API Reference](api.md) — Complete REST API documentation
- [API 文档 (中文)](api_zhcn.md) — REST API 接口文档
- [Architecture](项目架构.md) — Backend architecture design (Chinese)
- [Contributing](CONTRIBUTING.md) — How to contribute

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 16 + React 19 |
| Database | PostgreSQL 17 + SQLAlchemy 2.0 (async) |
| Cache/Queue | Redis (Redis Streams) |
| Object Storage | MinIO / S3-compatible |
| LLM Integration | OpenAI SDK (DeepSeek, Kimi, etc.) |
| Agent Protocols | MCP SDK + A2A |
| Browser Automation | Playwright + browser-use |
| Sandbox | Docker container isolation |
| Auth | JWT + bcrypt |
| Testing | pytest + Vitest |

## Contributing

We welcome Issues and Pull Requests! Please read the [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
