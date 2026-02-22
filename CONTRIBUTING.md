# 贡献指南

感谢你对 Actus 的关注！我们欢迎任何形式的贡献，包括但不限于：

- 报告 Bug
- 提交功能建议
- 提交代码修复或新功能
- 完善文档
- 分享使用经验

## 开始之前

1. 阅读本贡献指南
2. 阅读 [行为准则](CODE_OF_CONDUCT.md)
3. 查看 [已有 Issue](https://github.com/your-org/Actus/issues)，避免重复

## 开发环境搭建

### 后端 (API)

```bash
# 推荐使用 Python 3.12
cd api

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt

# 复制配置文件
cp config.yaml.example config.yaml
# 编辑 config.yaml 填入你的 LLM API Key

# 启动开发服务器
bash dev.sh
```

### 前端 (UI)

```bash
cd ui
npm install
npm run dev
```

### 运行测试

```bash
# 后端测试
cd api
pytest

# 前端测试
cd ui
npm run test
```

## 分支策略

- `main` — 稳定分支，所有发布从此分支创建
- `dev` — 开发分支，日常开发合并到此
- `feature/*` — 功能分支，从 `dev` 创建
- `fix/*` — 修复分支，从 `dev` 创建

## 提交 Pull Request

1. **Fork** 本仓库
2. 从 `dev` 创建你的功能分支：`git checkout -b feature/my-feature dev`
3. 编写代码并添加测试
4. 确保所有测试通过：`pytest`（后端）/ `npm run test`（前端）
5. 提交你的更改，使用清晰的 commit message
6. 推送到你的 Fork：`git push origin feature/my-feature`
7. 创建 Pull Request 到 `dev` 分支

### Commit Message 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]
```

类型说明：

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构（不新增功能或修复 Bug） |
| `test` | 添加或修改测试 |
| `chore` | 构建/工具/依赖变更 |

示例：
```
feat(agent): add timeout support for MCP tool calls
fix(ui): resolve session list scroll issue
docs: update deployment guide for MinIO config
```

## 代码规范

### 后端

- 遵循 [项目架构文档](项目架构.md) 中的分层规则和编码约束
- Domain 层禁止依赖任何框架（不导入 FastAPI / SQLAlchemy 等）
- 使用 Pydantic BaseModel 定义领域模型
- 所有异步操作使用 `async/await`
- 新增工具继承 `BaseTool` 并使用 `@tool` 装饰器

### 前端

- 使用 TypeScript，避免 `any` 类型
- 组件使用函数组件 + Hooks
- 状态管理使用 Zustand
- 样式使用 Tailwind CSS
- 为新组件编写测试

## 报告 Bug

请使用 [Bug Report](https://github.com/your-org/Actus/issues/new?template=bug_report.yml) 模板创建 Issue，并尽可能提供：

- 清晰的问题描述
- 复现步骤
- 期望行为 vs 实际行为
- 运行环境信息（OS、Docker 版本、浏览器等）
- 相关日志或截图

## 功能建议

请使用 [Feature Request](https://github.com/your-org/Actus/issues/new?template=feature_request.yml) 模板创建 Issue。

## 安全漏洞

**请不要在公开 Issue 中报告安全漏洞**。请参阅 [SECURITY.md](SECURITY.md) 了解安全问题的报告流程。

---

再次感谢你的贡献！
