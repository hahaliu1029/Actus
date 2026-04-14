# Actus Sandbox

`sandbox/` 定义了 Actus 的会话级 Docker 沙箱镜像。每个任务会话在运行时都会按需创建独立容器，用于执行 Shell、访问文件、控制浏览器和提供远程桌面能力。

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

在 Actus 主工程里，Compose 只负责构建 `sandbox-image`：

```bash
docker compose build sandbox-image
```

真正的会话沙箱由后端 API 通过 Docker Socket 动态创建，并以 `actus-sb*` 前缀命名。

## 本地单独调试

如果你要单独验证沙箱镜像：

```bash
cd sandbox
docker build -t actus-sandbox:dev .
docker run --rm -it \
  -p 8080:8080 \
  -p 9222:9222 \
  -p 5900:5900 \
  -p 5901:5901 \
  actus-sandbox:dev
```

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
