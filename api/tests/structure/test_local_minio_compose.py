"""Pin the standard Compose deployment to a local, initialized MinIO."""

from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE = (_REPO_ROOT / "docker-compose.yml").read_text()
_ENV_EXAMPLE = (_REPO_ROOT / ".env.example").read_text()
_CI = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
_DOCS = {
    path: (_REPO_ROOT / path).read_text()
    for path in ("README.md", "api/README.md", "CONTRIBUTING.md")
}
_DEPLOY_DOC = (_REPO_ROOT / "DEPLOY.md").read_text()
_README_EN = (_REPO_ROOT / "README_EN.md").read_text()


def _service_block(name: str) -> str:
    """Return exactly one two-space-indented Compose service block."""
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [\w-]+:\n|^[\w-]+:\n|\Z)",
        _COMPOSE,
    )
    assert match is not None, f"service {name!r} is missing"
    return match.group(0)


def test_minio_services_use_pinned_images():
    minio = _service_block("minio")
    init = _service_block("minio-init")

    assert "image: minio/minio:RELEASE.2025-09-07T16-13-09Z" in minio
    assert "image: minio/mc:RELEASE.2025-08-13T08-35-41Z" in init


def test_minio_api_and_console_are_loopback_only_and_data_is_persistent():
    minio = _service_block("minio")

    assert 'command: server /data --console-address ":9001"' in minio
    assert '"127.0.0.1:${MINIO_API_PORT:-9000}:9000"' in minio
    assert '"127.0.0.1:${MINIO_CONSOLE_PORT:-9001}:9001"' in minio
    assert "- minio-data:/data" in minio


def test_minio_has_readiness_healthcheck_restart_policy_and_network():
    minio = _service_block("minio")

    assert "restart: unless-stopped" in minio
    assert 'test: ["CMD-SHELL", "curl -fsS http://localhost:9000/minio/health/ready"]' in minio
    assert "interval: 5s" in minio
    assert "timeout: 3s" in minio
    assert "retries: 20" in minio
    assert "start_period: 5s" in minio
    assert re.search(r"(?m)^    networks:\n      - actus-net$", minio)


def test_minio_server_region_uses_the_shared_compose_region_source():
    minio = _service_block("minio")

    assert re.search(
        r"(?m)^      MINIO_SITE_REGION: \$\{MINIO_REGION:-us-east-1\}$", minio
    )


def test_minio_init_is_idempotent_and_preserves_container_variables():
    init = _service_block("minio-init")

    assert 'restart: "no"' in init
    assert re.search(
        r"(?m)^      minio:\n        condition: service_healthy$", init
    )
    assert 'entrypoint: ["/bin/sh", "-c"]' in init
    assert "set -eu" in init
    assert (
        'mc alias set local http://minio:9000 "$${MINIO_ACCESS_KEY}" '
        '"$${MINIO_SECRET_KEY}"'
    ) in init
    assert (
        'mc mb --ignore-existing --region "$${MINIO_REGION}" '
        '"local/$${MINIO_BUCKET_NAME}"'
    ) in init
    assert "|| true" not in init
    assert re.search(r"(?m)^    networks:\n      - actus-net$", init)


def test_api_waits_for_the_exact_minio_init_service():
    api = _service_block("api")

    assert re.search(
        r"(?m)^      minio-init:\n        condition: service_completed_successfully$",
        api,
    )


def test_api_uses_internal_minio_and_derives_local_public_endpoint():
    api = _service_block("api")

    assert re.search(r"(?m)^      MINIO_ENDPOINT: minio:9000$", api)
    assert re.search(
        r'^      MINIO_PUBLIC_ENDPOINT: "\$\{MINIO_PUBLIC_ENDPOINT:-localhost:'
        r'\$\{MINIO_API_PORT:-9000\}\}"$',
        api,
        re.MULTILINE,
    )
    assert re.search(r'(?m)^      MINIO_SECURE: "false"$', api)
    assert re.search(
        r"(?m)^      MINIO_PUBLIC_SECURE: \$\{MINIO_PUBLIC_SECURE:-false\}$", api
    )
    assert re.search(
        r"(?m)^      MINIO_REGION: \$\{MINIO_REGION:-us-east-1\}$", api
    )


def test_top_level_minio_data_volume_exists():
    volumes = re.search(
        r"(?ms)^volumes:\n(?P<body>.*?)(?=^[\w-]+:\n|\Z)", _COMPOSE
    )

    assert volumes is not None
    assert re.search(r"(?m)^  minio-data:$", volumes.group("body"))


def test_example_credentials_are_explicitly_local_only():
    minio_section = re.search(
        r"(?ms)^# ---------- MinIO.*?(?=^# ---------- 应用配置)", _ENV_EXAMPLE
    )

    assert minio_section is not None
    assert re.search(
        r"仅.*本地开发.*生产.*(?:不可用|必须自定义)", minio_section.group(0)
    )
    assert "MINIO_ACCESS_KEY=minioadmin" in minio_section.group(0)
    assert "MINIO_SECRET_KEY=minioadmin" in minio_section.group(0)


def test_ci_validates_rendered_local_minio_compose_without_starting_services():
    step = re.search(
        r"(?ms)^      - name: Validate standard Compose MinIO config\n"
        r"(?P<body>.*?)(?=^      - name: |^  [\w-]+:|\Z)",
        _CI,
    )

    assert step is not None
    body = step.group("body")
    assert "set -euo pipefail" in body
    assert (
        "MINIO_API_PORT=19000 docker compose --env-file .env.example config -q"
        in body
    )
    assert 'api_block="$(sed -n' in body
    assert 'minio_block="$(sed -n' in body
    assert "MINIO_PUBLIC_ENDPOINT: localhost:19000" in body
    assert "host_ip: 127.0.0.1" in body
    assert 'published: \"19000\"' in body
    assert "docker compose up" not in body


def test_ci_starts_pinned_minio_and_fails_closed_before_creating_bucket():
    step = re.search(
        r"(?ms)^      - name: Start MinIO \+ create bucket\n"
        r"(?P<body>.*?)(?=^      - name: |^  [\w-]+:|\Z)",
        _CI,
    )

    assert step is not None
    body = step.group("body")
    assert "minio/minio:RELEASE.2025-09-07T16-13-09Z" in body
    assert "minio/mc:RELEASE.2025-08-13T08-35-41Z" in body
    assert "minio/minio:latest" not in body
    assert "http://localhost:9000/minio/health/ready" in body
    assert "ready=0" in body
    assert "for i in $(seq 1 30)" in body
    assert re.search(r"if curl .*; then\s+ready=1\s+break\s+fi", body)
    assert "sleep 1" in body
    assert 'test "$ready" = 1' in body
    assert "mc alias set local http://localhost:9000 minioadmin minioadmin" in body
    assert "mc mb --ignore-existing --region us-east-1 local/a2a-mcp" in body


def test_ci_keeps_the_standard_compose_config_gate():
    assert "- name: Validate standard Compose MinIO config" in _CI
    assert (
        "MINIO_API_PORT=19000 docker compose --env-file .env.example config -q"
        in _CI
    )


def test_each_primary_doc_describes_the_local_minio_topology_and_boundary():
    for path, body in _DOCS.items():
        assert "标准 Docker Compose" in body, path
        assert "a2a-mcp" in body, path
        assert "http://127.0.0.1:9000" in body, path
        assert "http://127.0.0.1:9001" in body, path
        assert "MINIO_ACCESS_KEY" in body, path
        assert "MINIO_SECRET_KEY" in body, path
        assert "MINIO_API_PORT" in body, path
        assert "MINIO_PUBLIC_ENDPOINT" in body, path
        assert "MINIO_PUBLIC_SECURE" in body, path
        assert "生产基线" in body, path
        assert "远程 S3" in body, path
        assert "tunnel" in body, path


def test_api_readme_documents_direct_host_api_minio_environment():
    body = _DOCS["api/README.md"]

    assert "MINIO_ENDPOINT=localhost:9000" in body
    assert "MINIO_PUBLIC_ENDPOINT=localhost:9000" in body
    assert "MINIO_REGION=us-east-1" in body
    assert "MINIO_SECURE=false" in body
    assert "MINIO_PUBLIC_SECURE=false" in body


def test_deploy_doc_describes_local_minio_migration_and_remote_boundary():
    body = _DEPLOY_DOC

    assert "标准 Docker Compose" in body
    assert "a2a-mcp" in body
    assert "http://127.0.0.1:9000" in body
    assert "http://127.0.0.1:9001" in body
    assert "MINIO_PUBLIC_ENDPOINT" in body
    assert "MINIO_PUBLIC_SECURE" in body
    assert "MINIO_ENDPOINT" in body
    assert "不自动迁移" in body
    assert "远程 S3" in body
    assert "生产基线" in body
    assert "tunnel" in body
    assert "Compose **不会** 启动 MinIO" not in body
    assert "- 可访问的 MinIO / S3 兼容对象存储" not in body
    assert "- `MINIO_BUCKET_NAME` 对应的 bucket 需要事先存在" not in body


def test_english_readme_describes_local_minio_migration_and_remote_boundary():
    body = _README_EN

    assert "standard Docker Compose" in body
    assert "a2a-mcp" in body
    assert "http://127.0.0.1:9000" in body
    assert "http://127.0.0.1:9001" in body
    assert "MINIO_PUBLIC_ENDPOINT" in body
    assert "MINIO_PUBLIC_SECURE" in body
    assert "MINIO_ENDPOINT" in body
    assert "no automatic migration" in body
    assert "remote S3" in body
    assert "production baseline" in body
    assert "tunnel" in body
    assert "host-run API or custom orchestration only" in body
    assert "- A reachable MinIO / S3-compatible bucket that already exists" not in body
    assert "# MINIO_ENDPOINT" not in body
