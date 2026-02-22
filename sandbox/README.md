# Sandbox 沙箱服务

Actus 的隔离代码执行环境。每个用户会话通过独立的 Docker 容器提供安全的沙箱环境，用于执行 Agent 生成的代码。

## 功能

- Shell 命令执行
- 文件读写操作
- 进程管理（基于 Supervisor）
- Chrome 浏览器支持（用于网页自动化）

## 架构

```
sandbox/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── core/                # 配置与中间件
│   ├── interfaces/          # API 路由与 Schema
│   ├── models/              # 数据模型
│   └── services/            # 业务逻辑（Shell、文件、Supervisor）
├── Dockerfile               # 沙箱镜像定义
├── supervisord.conf          # Supervisor 进程管理配置
└── requirements.txt          # Python 依赖
```

## 说明

沙箱服务不需要独立启动，API 服务会通过 Docker Socket 动态创建和管理沙箱容器。镜像构建由 `docker-compose.yml` 中的 `sandbox-image` 服务完成。