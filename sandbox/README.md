# Actus Sandbox

`sandbox/` 定义了 Actus 的会话级 Docker 沙箱镜像。后端是否以及何时创建独立容器，
由 `SANDBOX_PROVISION_MODE` 控制；容器用于执行 Shell、访问文件、控制浏览器和提供
远程桌面能力。

## 沙箱内包含什么

- FastAPI 服务（默认端口 `8080`）
- Shell / PTY 会话执行能力
- 文件读写、搜索、替换、上传下载
- Chromium 浏览器
- CDP 转发端口 `9222`
- Xvfb 虚拟显示
- x11vnc（`5900`）
- websockify / noVNC WebSocket（`5901`）
- Supervisor 统一管理所有进程

## 构建与运行方式

在 Actus 主工程里先按根目录 `.env.example` 准备 `.env`（包括有效的
`MEMORY_ROOT_HOST`），Compose 只负责构建 `sandbox-image`：

```bash
docker compose --env-file .env build sandbox-image
```

真正的会话沙箱由后端 API 通过 Docker Socket 动态创建，并以 `actus-sb*` 前缀命名。

## 沙箱供给模式

| 模式 | 行为 | Compose 要求 |
|------|------|--------------|
| `always` | 默认值；每个 session 预置父沙箱，保持原有 eager 行为 | 标准 `docker-compose.yml` |
| `on_demand` | 首次沙箱工具、VNC、接管或 coordinator 父沙箱 I/O 触发时才创建；纯聊天 session 不创建容器 | 标准 Compose；可用 `SANDBOX_PROVISION_TIMEOUT_SECONDS` 调整 create / ready / hooks 总预算（默认 90 秒） |
| `off` | 不创建容器，不注册沙箱工具，VNC/接管/skill-create 面拒绝请求 | 必须叠加 `docker-compose.sandbox-off.yml` |

切换 `always` / `on_demand` 时修改根目录 `.env` 后重启 API：

```bash
SANDBOX_PROVISION_MODE=on_demand
docker compose --env-file .env up -d --build
```

`off` 不能只设置环境变量。canonical override 会把模式 literal-pin 为 `off`、从 API
移除 `/var/run/docker.sock`、删除 `sandbox-image` 依赖并停用该镜像服务；它要求 Docker
Compose 2.24.4 或更高版本。切换前还必须先 drain 存量沙箱，并将三个 coordinator flag
全部设为 `false`。不要自行拼装命令，完整流程和回退步骤见
[`../docs/runbooks/sandbox-off-runbook.md`](../docs/runbooks/sandbox-off-runbook.md)。

## 本地单独调试

如果你要单独验证沙箱镜像：

```bash
cd sandbox
docker build -t actus-sandbox:dev .
docker run --rm -it \
  -p 127.0.0.1:8080:8080 \
  -p 127.0.0.1:9222:9222 \
  -p 127.0.0.1:5900:5900 \
  -p 127.0.0.1:5901:5901 \
  actus-sandbox:dev
```

这些调试端口没有面向公网的认证边界，示例只绑定 `127.0.0.1`；不要改成全网卡监听。

启动后可访问：

- OpenAPI：`http://localhost:8080/docs`
- CDP：`http://localhost:9222`
- VNC：`localhost:5900`
- noVNC WebSocket：`ws://localhost:5901`

## 当前技术栈

- Ubuntu 22.04
- Python 3.10
- Node.js 24
- FastAPI
- Supervisor
- Chromium
- Xvfb
- x11vnc
- websockify

## 目录结构

```text
sandbox/
├── app/
│   ├── core/                   # 配置与中间件
│   ├── interfaces/            # FastAPI 路由、Schema、错误处理
│   ├── models/                # 数据模型
│   └── services/              # Shell / File / Supervisor 服务
├── Dockerfile
├── supervisord.conf
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## API 路由

沙箱 API 统一挂在 `/api` 下，分成三类：

- `/api/file/*`
- `/api/shell/*`
- `/api/supervisor/*`

其中：

- `shell/ws` 提供 PTY 双向 WebSocket
- `POST /api/shell/exec-command` 支持 `wait_seconds` 参数，允许调用方对短命令同步等待结果，避免"明知短命令但还要异步轮询"
- 文件接口支持读取、写入、替换、搜索、上传、下载、删除
- Supervisor 接口支持超时销毁、重启和状态查询

## 关键文件

- `Dockerfile`
- `supervisord.conf`
- `app/main.py`
- `app/interfaces/endpoints/shell.py`
- `app/interfaces/endpoints/file.py`
- `app/interfaces/endpoints/supervisor.py`

## 开发注意事项

- 修改沙箱代码后，需要在主工程中重建 `sandbox-image` 和 `api`
- `supervisord.conf` 里定义了 app、chrome、socat、xvfb、x11vnc、websockify 的启动顺序
- Chromium 实际监听 `8222`，通过 `socat` 转发到对外的 `9222`
- API lifecycle 当前只支持单 worker：`WEB_CONCURRENCY` 必须为 `1`（或不设置），不得
  scale `api` 或为 uvicorn 增加多个 workers
- `always` / `on_demand` 需要 API 的 Docker Socket 挂载；`off` 下任何新增代码都不得
  重新引入 Docker 访问或 lifecycle mutation

## 测试

`sandbox/` 是 uv workspace member，但拥有独立的 `app` package；它与 `api/app` 同名，
因此必须从本目录作为独立 pytest project 运行：

```bash
cd sandbox
uv run --locked pytest -q
```

供给模式的配置、装配与流程测试属于后端测试，从 `api/` 运行并使用项目 venv：

```bash
cd api
uv run pytest tests/core/test_sandbox_provision_mode_config.py -v
uv run pytest tests/app/application/services/test_sandbox_provisioner.py -v
```

`tests/sandbox/` 中标记为 `sandbox` 的测试需要 Docker；`sandbox_real_image` 子集还会启动
真实 `actus-sandbox` 镜像，默认 pytest 会排除它们，CI 分别在 adversarial 与
real-image smoke job 中运行。
