# Actus UI

`ui/` 是 Actus 的 Next.js 16 前端，负责聊天交互、会话列表、任务摘要、工作台、设置页和管理员界面。

## 当前界面能力

- 登录 / 注册
- 首页快捷提问与会话入口
- 左侧会话列表、删除会话、主题切换
- 会话详情页：
  - 流式消息展示
  - Markdown 渲染（基于 `react-markdown`）+ `shiki` 代码语法高亮
  - 计划 / 步骤状态展示
  - 工具确认卡片（响应 `tool_confirmation` 事件，承载工具审批与确认系统的用户决策）
  - 任务摘要与文件面板
  - 终端预览
  - 浏览器预览
  - VNC 画面
  - 时间线回放
  - 子智能体树、合并时间线与成本汇总
  - 上下文压缩折叠标记与详情弹窗
  - `/help`、`/mcp`、`/skills`、`/cost`、`/permissions`、`/takeover`、
    `/compact` 斜杠命令及 Skill 命令补全
  - 图片、视频和 PDF/文档结构化工具结果预览
  - 按会话 `sandbox_mode` 收缩工作台、VNC 与沙箱文件操作
- 会话事件恢复（两条路径）：
  - **页面刷新 / 重新进入**：先 `GET /sessions/{id}` 拉完整快照，再 `POST /sessions/{id}/chat` 携带 `event_id` 续流
  - **运行中连接中断**：通过 `GET /sessions/{id}/events?since_seq=...&since=...` 增量补齐；优先使用单调 `seq`，同时保留 `event_id` 兼容旧事件。该 GET 不是实时订阅，补齐后会话仍在运行时需重新走 `/chat` 的 `event_id` 续流路径
- 设置弹窗：
  - Agent 通用配置
  - 模型提供商配置
  - MCP 服务器
  - A2A Agent 配置
  - Skill 生态
  - MCP / A2A / Skill 运行时扩展总览、探测与统一启停
  - Plugin 安装、启停与卸载
  - 记忆管理（创建、编辑、筛选、批量删除、pin/unpin）
  - 文件理解配置（视觉降级、音频转录、视频分析）— 视觉/音频面板新版样式
  - 用户管理
- 文件传输面板：上传/下载进度跟踪、EMA 测速、取消和重试
- 图片代理路由：`/api/image-proxy`

## 技术栈

- Next.js 16（App Router）
- React 19
- Tailwind CSS 4
- Zustand
- Radix UI
- `react-markdown` + `shiki`（代码语法高亮）
- noVNC
- Vitest + Testing Library

## 本地开发

```bash
cd ui
npm install
npm run dev
```

常用命令：

```bash
npm run lint
npm run test
npm run build
```

默认开发地址：`http://localhost:3000`

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEXT_PUBLIC_API_BASE_URL` | 浏览器可访问的后端 API 地址 | `http://localhost:8000/api` |

注意：

- 这是**构建时注入**变量
- 容器部署时由 `ui-app` 镜像构建参数传入
- 修改后端域名或端口后，需要重新执行前端构建

## 部署形态

容器模式下前端分成两层：

- `ui-app`：Next.js standalone 运行时
- `ui`：nginx 网关，对外暴露 80 端口

对应文件：

- `Dockerfile`
- `Dockerfile.nginx`
- `next.config.ts`
- `nginx.conf`

## 目录结构

```text
ui/
├── src/
│   ├── app/                   # App Router 页面与 API route
│   │   ├── page.tsx           # 首页
│   │   ├── login/             # 登录页
│   │   ├── register/          # 注册页
│   │   ├── sessions/[id]/     # 会话详情页
│   │   └── api/image-proxy/   # 图片代理
│   ├── components/            # UI 组件
│   ├── hooks/                 # 自定义 hooks
│   ├── lib/                   # API 客户端、store、状态文案、工具函数
│   └── test/                  # 测试初始化
├── public/
├── package.json
└── next.config.ts
```

## 关键组件

- `components/left-panel.tsx`
- `components/chat-input.tsx`
- `components/session-task-dock.tsx`
- `components/workbench-panel.tsx`
- `components/workbench-interactive-terminal.tsx`
- `components/workbench-browser-preview.tsx`
- `components/vnc-viewer.tsx`
- `components/manus-settings.tsx`
- `components/command-menu.tsx` — 斜杠命令筛选与键盘补全
- `components/session/agent-tree-panel.tsx` — 子智能体层级与运行状态
- `components/session/merged-timeline-panel.tsx` — 父子会话合并时间线
- `components/session/compaction-detail-modal.tsx` — 上下文压缩详情
- `components/settings/extensions-overview.tsx` — 运行时扩展清单与治理入口
- `components/settings/plugin-install-dialog.tsx` — Plugin 安装流程
- `components/settings/memory-management.tsx` — 用户记忆管理
- `components/markdown-renderer.tsx` — react-markdown + shiki 渲染管道
- `components/tool-confirmation-card.tsx` — 工具调用确认卡片，对接后端工具审批系统
- `components/transfer-panel.tsx` — 文件传输进度面板
- `components/transfer-progress.tsx` — 单个传输任务进度条

## 状态管理

Zustand store 位于：

- `src/lib/store/auth-store.ts`
- `src/lib/store/session-store.ts`
- `src/lib/store/settings-store.ts`
- `src/lib/store/lifecycle-store.ts` — C7 生命周期事件的灰度 typed reducer 状态
- `src/lib/store/transfer-store.ts` — 文件传输状态管理（进度、速度、取消）
- `src/lib/store/ui-store.ts`

斜杠命令定义与执行位于 `src/lib/commands/`；父子会话树归一化位于
`src/lib/agent-tree.ts`；生命周期事件协议与分发位于 `src/lib/lifecycle/`。

## HTTP 客户端

- `src/lib/api/fetch.ts` — 通用 API 请求（基于 fetch）
- `src/lib/api/axios-client.ts` — 文件传输专用（基于 axios，支持 401 自动刷新 token、进度回调）
- `src/lib/api/auth-utils.ts` — Token 管理与刷新逻辑

## 测试

测试文件与源码并列，例如：

- `src/components/*.test.tsx`
- `src/lib/**/*.test.ts`
- `src/app/sessions/[id]/page.test.tsx`

运行方式：

```bash
cd ui
npm run test
```
