# Changelog

本文件记录项目的版本变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

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
