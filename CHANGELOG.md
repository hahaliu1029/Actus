# Changelog

本文件记录项目的版本变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] - 2026-02-24

### 变更

- 沙箱 Shell 执行链路优化：长时间命令（如 `npm start`、`pip install`、`apt-get install`）不再阻塞主流程，`shell_execute` 会在短等待后返回 `running`，可通过读取输出/等待进程继续跟踪。
- Shell 输出读取任务增加会话级生命周期管理（启动前回收旧 reader，结束后清理 task），降低并发读输出与后台任务泄漏风险。
- Shell 输出统一增加最大长度截断策略，降低长命令持续输出导致的内存增长风险。
- 安装类命令自动非交互化增强：对 `apt/apt-get`、`yum/dnf`、`apk`、`pip`、`npm/yarn/pnpm/npx`、`conda`、`poetry` 注入常见非交互参数，减少卡在确认提示的概率。
- 设置页「模型提供商」补全全部 LLM 配置项的中文说明文案，便于理解参数语义。

### 文档

- 更新中英文 API 文档中的 `LLMConfig` 字段定义，补充上下文溢出治理相关配置项。
- 补充部署文档与 README：说明 `sandbox-image` 为 Compose 服务名，并提供沙箱代码变更后的正确重建命令。

## [0.1.0] - 2025-XX-XX

### 新增

- ReAct Agent 引擎（Reasoning + Acting 推理循环）
- MCP (Model Context Protocol) 工具协议集成，支持动态接入外部工具服务器
- A2A (Agent-to-Agent) 智能体间通信协议支持
- Planner + ReAct 两阶段 Agent 流程编排
- Docker 沙箱隔离代码执行环境
- Playwright + DOM 索引提取方案浏览器自动化，通过 CDP 连接沙箱 Chromium，实时提取可交互元素并通过选择器精确操作
- 多模型支持（兼容 OpenAI API 格式：DeepSeek、Kimi 等）
- 流式输出与思考过程展示
- MinIO/S3 兼容的文件上传下载
- JWT 认证与角色权限管理
- Next.js 16 + React 19 现代前端
- Docker Compose 一键部署
- PostgreSQL + Redis 数据存储
- Skill 生态系统：独立的 Skill 扩展层，基于文件系统存储（`/app/data/skills`），支持 GitHub / 本地目录双来源安装
- SKILL.md 驱动的安装规范，支持 frontmatter 自动解析，manifest 为可选兼容字段
- Skill 安全机制：命令黑名单、风险策略（off / enforce_confirmation）、路径穿越防护、bundle 大小限制
- Skill 选择器：基于关键词评分的渐进式候选 Skill 推荐
- Skill 索引服务：基于目录 mtime 的缓存失效机制
- Skill v2 API（`/api/v2/skills/*`），v1 API 返回 410 引导迁移
