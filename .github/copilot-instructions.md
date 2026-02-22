# Actus Copilot Instructions

## 项目概述

这是一个 **开源 AI Agent 平台**，基于 MCP (Model Context Protocol) 工具协议与 A2A (Agent-to-Agent) 智能体间通信协议，构建具有工具调用能力的 ReAct Agent。

## 架构模式

### 整体架构
项目由三个子服务组成：
- `api/` — 后端服务 (FastAPI)，采用 **整洁架构 (Clean Architecture) + DDD**
- `ui/` — 前端 (Next.js 16 + React 19)
- `sandbox/` — 沙箱服务（Docker 隔离代码执行）

### ReAct Agent 实现模式
项目采用 **ReAct (Reasoning + Acting)** 模式，核心循环：
1. 用户输入 → 2. LLM 推理 → 3. 工具调用（可选）→ 4. 结果整合 → 5. 递归/输出

### 工具定义约定
使用 **Pydantic BaseModel** 定义工具输入参数 schema：

```python
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    param: str = Field(..., description="参数描述")

# 转换为 OpenAI tools 格式
tools = [{
    "type": "function",
    "function": {
        "name": tool_func.__name__,
        "description": tool_func.__doc__,  # 使用函数 docstring
        "parameters": ToolInput.model_json_schema(),
    }
}]
```

## 环境配置

- 使用 `.env` 文件管理环境变量（参考 `.env.example`）
- 运行时配置使用 `api/config.yaml`（参考 `api/config.yaml.example`）
- 敏感信息（密钥、密码）通过环境变量注入，**不硬编码**

## 常用模型标识

| Provider | Model ID | 用途 |
|----------|----------|------|
| DeepSeek | `deepseek-chat` | 通用对话 |
| DeepSeek | `deepseek-reasoner` | 推理增强 |
| Kimi | `kimi-k2-0905-preview` | 通用对话 |

## 流式响应处理

处理流式 tool_calls 时需要按 `index` 聚合分片：
```python
tool_calls_obj: dict[int, ChoiceDeltaToolCall] = {}
for chunk in response:
    if chunk_tool_calls := chunk.choices[0].delta.tool_calls:
        for tc in chunk_tool_calls:
            if tc.index not in tool_calls_obj:
                tool_calls_obj[tc.index] = tc
            else:
                tool_calls_obj[tc.index].function.arguments += tc.function.arguments
```

## 开发命令

```bash
# 安装依赖（使用 uv）
uv sync

# 后端开发
cd api && bash dev.sh

# 前端开发
cd ui && npm run dev

# 运行测试
pytest api/tests/          # 后端测试
cd ui && npm run test      # 前端测试

# Docker 部署
cp .env.example .env       # 配置环境变量
cp api/config.yaml.example api/config.yaml  # 配置运行时参数
docker compose --env-file .env up -d --build
```

## 注意事项

- 工具调用后必须将结果以 `role: "tool"` 消息格式追加到 messages
- `tool_call_id` 必须与对应的 tool_call 匹配
- 递归调用 `process_query()` 时不要重复添加用户消息
