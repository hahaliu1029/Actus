"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useAuth } from "@/hooks/use-auth";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [nickname, setNickname] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      await register({
        username,
        email,
        nickname,
        password,
      });
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "注册失败，请重试");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="relative mx-auto flex min-h-screen w-full max-w-md items-center px-6 py-8">
      <div className="w-full rounded-2xl border border-border bg-card/80 p-6 shadow-[var(--shadow-elevated)] backdrop-blur-xl animate-scale-in">
        <h1 className="mb-2 text-2xl font-semibold text-foreground">创建账号</h1>
        <p className="mb-6 text-sm text-muted-foreground">注册后将自动登录并进入工作台</p>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <label className="block text-sm text-foreground/80">
            用户名
            <input
              className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground outline-none transition-all focus:border-border-strong focus:ring-2 focus:ring-ring/20"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </label>

          <label className="block text-sm text-foreground/80">
            邮箱
            <input
              type="email"
              className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground outline-none transition-all focus:border-border-strong focus:ring-2 focus:ring-ring/20"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>

          <label className="block text-sm text-foreground/80">
            昵称
            <input
              className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground outline-none transition-all focus:border-border-strong focus:ring-2 focus:ring-ring/20"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
            />
          </label>

          <label className="block text-sm text-foreground/80">
            密码
            <input
              type="password"
              className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground outline-none transition-all focus:border-border-strong focus:ring-2 focus:ring-ring/20"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
          </label>

          {error ? <p className="text-sm text-destructive">{error}</p> : null}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-all active:scale-[0.98] disabled:opacity-60"
          >
            {isSubmitting ? "注册中..." : "注册并登录"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-muted-foreground">
          已有账号？
          <Link href="/login" className="ml-1 text-foreground underline">
            去登录
          </Link>
        </p>
      </div>
    </main>
  );
}
