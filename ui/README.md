# Actus 前端

基于 Next.js 16 + React 19 构建的 Actus 前端界面。

## 技术栈

- **框架**: Next.js 16 (App Router)
- **UI**: React 19 + Tailwind CSS 4 + Radix UI
- **状态管理**: Zustand
- **Markdown**: markdown-it
- **远程桌面**: noVNC（用于沙箱浏览器预览）
- **测试**: Vitest + Testing Library

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 代码检查
npm run lint

# 运行测试
npm run test
```

开发服务器启动后访问 http://localhost:3000。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEXT_PUBLIC_API_BASE_URL` | 后端 API 地址（构建时注入） | `http://localhost:8000/api` |

## 项目结构

```
src/
├── app/                 # Next.js App Router 页面
│   ├── layout.tsx       # 根布局
│   ├── page.tsx         # 首页
│   ├── login/           # 登录页
│   ├── register/        # 注册页
│   └── sessions/        # 会话页面
├── components/          # React 组件
├── hooks/               # 自定义 Hooks
└── lib/                 # 工具函数与配置
```

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
