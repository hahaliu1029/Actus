"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { useAuth } from "@/hooks/use-auth";

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuth();

  const [identity, setIdentity] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const redirectTo = searchParams.get("redirect") || "/";

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const isEmail = identity.includes("@");
      await login({
        ...(isEmail ? { email: identity } : { username: identity }),
        password,
      });
      router.replace(redirectTo);
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败，请重试");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="min-h-screen w-full bg-cover bg-center bg-no-repeat"
      style={{backgroundImage: "url('/login-bg.png')"}}
    >
    <main className="relative mx-auto flex min-h-screen w-full max-w-md items-center px-6 py-8">
      <div className="w-full rounded-2xl border border-border bg-card/80 p-6 shadow-[var(--shadow-elevated)] backdrop-blur-xl animate-scale-in">
        <h1 className="mb-2 text-2xl font-semibold text-foreground">登录 Actus</h1>
        <p className="mb-6 text-sm text-muted-foreground">使用用户名或邮箱登录</p>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <label className="block text-sm text-foreground/80">
            用户名或邮箱
            <input
              className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground outline-none transition-all focus:border-border-strong focus:ring-2 focus:ring-ring/20"
              value={identity}
              onChange={(e) => setIdentity(e.target.value)}
              required
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
            />
          </label>

          {error ? <p className="text-sm text-destructive">{error}</p> : null}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-all active:scale-[0.98] disabled:opacity-60"
          >
            {isSubmitting ? "登录中..." : "登录"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-muted-foreground">
          还没有账号？
          <Link href="/register" className="ml-1 text-foreground underline">
            立即注册
          </Link>
        </p>
      </div>
    </main>
    </div>
  );
}
