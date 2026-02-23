# Deployment Guide (Docker Compose)

This guide deploys the full stack on one host:

- `ui` (Next.js)
- `api` (FastAPI)
- `postgres`
- `redis`
- sandbox image build for dynamic task containers

## 1. Prerequisites

- Docker Engine + Docker Compose v2
- At least 6 GB memory available for Docker

## 2. Configure environment

```bash
cd /path/to/Actus-opensource
cp .env.example .env
```

Edit `.env` at minimum:

- `JWT_SECRET_KEY`
- `POSTGRES_PASSWORD`
- `NEXT_PUBLIC_API_BASE_URL`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET_NAME`

`NEXT_PUBLIC_API_BASE_URL` must point to a browser-reachable API URL.
Remote MinIO bucket must already exist and be reachable from `api`.

## 3. Start all services

```bash
cd /path/to/Actus-opensource
docker compose --env-file .env up -d --build
```

## 4. Verify runtime

```bash
docker compose ps
docker compose logs -f api
```

Open:

- UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

## 5. Create super admin (optional but recommended)

```bash
cd /path/to/Actus-opensource
docker compose exec api python scripts/create_super_admin.py
```

## 6. Stop / restart

```bash
docker compose down
docker compose up -d
```

To remove all persistent data volumes as well:

```bash
docker compose down -v
```

## Notes

- API startup runs Alembic migrations automatically.
- App config is persisted in `api-data` volume (`/app/data/config.yaml`).
- **Skill data is stored on the filesystem** at `/app/data/skills` within the `api-data` volume (configurable via `SKILLS_ROOT_DIR` environment variable). Each Skill occupies a separate directory containing `meta.json`, `manifest.json`, `SKILL.md`, and an optional `bundle/` directory.
- Sandbox containers are created by API through Docker socket mount.
- MinIO is not started by compose. API connects to your remote MinIO using `.env` values.
- If your deployment domain changes, rebuild UI so new `NEXT_PUBLIC_API_BASE_URL` is baked in.
- Skills can be installed and managed through the frontend under **Settings → Skill Ecosystem**, supporting both GitHub and local directory sources.
