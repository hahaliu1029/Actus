"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { governanceApi } from "@/lib/api/config";
import type {
  ExtensionKind,
  GovernanceBlock,
  PluginMemberDetail,
  RuntimeCatalogItem,
  RuntimeConfigStatus,
  RuntimeExtensionItem,
  RuntimeHealth,
  RuntimeLiveness,
  RuntimeStats,
} from "@/lib/api/types";
import { useSettingsStore } from "@/lib/store/settings-store";

/**
 * B9 Task 22 — 运行时扩展总览（只读骨架，PR-4a）。
 *
 * 组织方式对齐 memory-management.tsx：数据/加载态一律组件内经
 * useSettingsStore 自取；跨组件动作（跳转管理区/预填 MCP 配置）走回调 props。
 *
 * 轮询生命周期（R6#6）：组件仅在 extensions tab 激活时挂载（父组件条件渲染），
 * 因此直接挂 mount——useEffect 立即 load 一次 + 30s 定时 poll；unmount cleanup
 * 清定时器并显式调 invalidateRuntimeRequests()（R2#5 token 失效入口）。
 * 不触发 loadAll()——父组件打开时的既有 loadAll() 不动。
 */

const POLL_INTERVAL_MS = 30_000;

type ExtensionsOverviewProps = {
  isAdmin: boolean;
  // 管理区跳转（父=setActiveTab）。
  onSelectTab: (tab: "mcp" | "a2a" | "skill") => void;
  // Task 24 才接线（4a 可不传）：父=setActiveTab("mcp")+预填局部 state+打开
  // 既有 MCP 添加弹窗。
  onPrefillMcpConfig?: (payloadJson: string) => void;
};

// probe 健康态（reachable/unreachable）与 integrity 健康态（ok/error）+ 共用态
// （unknown/skipped）的中文文案 + Badge tone 映射（R1#6）。
type BadgeTone = "reachable" | "unreachable" | "ok" | "error" | "unknown" | "skipped";

const HEALTH_LABELS: Record<BadgeTone, string> = {
  reachable: "可达",
  unreachable: "不可达",
  ok: "正常",
  error: "损坏",
  unknown: "未知",
  skipped: "已禁用（未探测）",
};

const HEALTH_TONE_CLASS: Record<BadgeTone, string> = {
  reachable: "bg-emerald-100 text-emerald-700",
  unreachable: "bg-red-100 text-red-700",
  ok: "bg-emerald-100 text-emerald-700",
  error: "bg-orange-100 text-orange-700",
  unknown: "bg-muted text-muted-foreground",
  skipped: "bg-muted/60 text-muted-foreground/80",
};

const KIND_LABELS: Record<ExtensionKind, string> = {
  mcp: "MCP",
  a2a: "A2A",
  skill: "Skill",
  plugin: "Plugin",
};

// D1a Task 26 — 治理状态 / 扫描裁决中文文案。
const GOV_STATUS_LABELS: Record<GovernanceBlock["status"], string> = {
  active: "生效中",
  quarantined: "已隔离",
  disabled: "已停用",
  deleted: "已删除",
};

const GOV_SCAN_LABELS: Record<NonNullable<GovernanceBlock["scan_verdict"]>, string> = {
  safe: "扫描安全",
  caution: "扫描存疑",
  dangerous: "扫描危险",
};

const GOV_QUARANTINE_REASON_LABELS: Record<
  NonNullable<GovernanceBlock["quarantine_reason"]>,
  string
> = {
  pin_mismatch: "pin 不匹配（观测哈希漂移）",
  admin_manual: "管理员手动隔离",
};

function govScanLabel(verdict: GovernanceBlock["scan_verdict"]): string {
  if (verdict === "safe" || verdict === "caution" || verdict === "dangerous") {
    return GOV_SCAN_LABELS[verdict];
  }
  return "扫描未知";
}

function govQuarantineReasonLabel(
  reason: GovernanceBlock["quarantine_reason"]
): string {
  if (reason === "pin_mismatch" || reason === "admin_manual") {
    return GOV_QUARANTINE_REASON_LABELS[reason];
  }
  return "未知原因";
}

// probe 按钮文案：mcp/a2a="重新探测"（网络握手）；skill="重新扫描"（短路重扫语义，R9#1）。
function probeButtonLabel(kind: ExtensionKind): string {
  return kind === "skill" ? "重新扫描" : "重新探测";
}

// PR4A-R1 Audit Fix B — catalog homepage href scheme 注入防御：仅放行 http/https。
// wire 类型是裸 str，`javascript:` / `data:` URL 是 click-to-execute 面（rel 无用）。
function safeHomepage(homepage: string): string | null {
  return /^https?:\/\//i.test(homepage) ? homepage : null;
}

/**
 * 429 冷却倒计时 tick（Task 23）。cooldowns 里存的是"可重试 epoch ms"；
 * 只要还有未过期的冷却窗，就每秒重渲染一次让按钮到点自动解禁。空表则不起定时器。
 *
 * 纯度约束（react-hooks/purity + set-state-in-effect 门）：render 阶段禁调 Date.now()，
 * effect 体也禁同步 setState。因此 `now` 由 lazy useState 首帧取一次基准，其后仅在
 * interval 回调（异步）里刷新——effect 只在存在未来冷却窗时起 1s interval，并在末次
 * 到点后清掉自己。按钮 disabled 只看 `readyAt - now > 0` 的符号，轻微时钟滞后不影响
 * 正确性（冷却期间恒 disabled）。
 */
function useCooldownTick(cooldowns: Record<string, number>): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const values = Object.values(cooldowns);
    if (values.length === 0) {
      return;
    }
    const maxReadyAt = Math.max(...values);
    const id = window.setInterval(() => {
      const current = Date.now();
      setNow(current);
      if (current >= maxReadyAt) {
        window.clearInterval(id);
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [cooldowns]);
  return now;
}

function isKnownHealthState(state: RuntimeHealth["state"]): state is BadgeTone {
  return (
    state === "reachable" ||
    state === "unreachable" ||
    state === "ok" ||
    state === "error" ||
    state === "unknown" ||
    state === "skipped"
  );
}

function HealthBadge({ health }: Readonly<{ health: RuntimeHealth }>) {
  // reason_code / state 都是 wire string——防御式渲染：未知值回退到 unknown tone
  // + 原样展示 state 文本，绝不 cast。
  const known = isKnownHealthState(health.state);
  const tone: BadgeTone = known ? health.state : "unknown";
  const baseLabel = known ? HEALTH_LABELS[tone] : health.state;
  const label = health.stale ? `${baseLabel}（可能过期）` : baseLabel;
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-xs font-medium ${HEALTH_TONE_CLASS[tone]}`}
      data-testid="health-badge"
    >
      {label}
    </span>
  );
}

function LivenessBadge({ liveness }: Readonly<{ liveness: RuntimeLiveness }>) {
  // in_use=蓝点+"使用中 ×N"；idle=灰"空闲"；unknown/not_applicable=不渲染。
  if (liveness.state === "in_use") {
    return (
      <span
        className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium bg-sky-100 text-sky-700"
        data-testid="liveness-badge"
      >
        <span className="inline-block size-1.5 rounded-full bg-sky-500" />
        使用中 ×{liveness.active_run_count}
      </span>
    );
  }
  if (liveness.state === "idle") {
    return (
      <span
        className="rounded px-1.5 py-0.5 text-xs font-medium bg-muted text-muted-foreground"
        data-testid="liveness-badge"
      >
        空闲
      </span>
    );
  }
  return null;
}

// stats.available=false 的 reason → 中文文案（R6#5/§7 冻结）。admin_only 不在此表：
// 该 reason 下整个统计区不渲染（见 StatsBlock 早退）。unsupported=A2A 无统计语义。
const STATS_UNAVAILABLE_COPY: Record<
  Exclude<NonNullable<RuntimeStats["unavailable_reason"]>, "admin_only">,
  string
> = {
  unsupported: "暂不支持统计",
  disabled: "统计未启用",
  redis_unavailable: "统计暂不可用",
};

function statsUnavailableCopy(
  reason: RuntimeStats["unavailable_reason"]
): string {
  if (
    reason === "unsupported" ||
    reason === "disabled" ||
    reason === "redis_unavailable"
  ) {
    return STATS_UNAVAILABLE_COPY[reason];
  }
  // null / 未知 reason 防御式兜底（绝不 cast）——available=false 但 reason 缺失。
  return "统计暂不可用";
}

/**
 * 调用统计区（Task 24）。仅 Admin 渲染——非 Admin 时投影层将 reason 置为
 * admin_only，此处直接早退（整个统计区不渲染，不泄露任何数字）。
 *
 * available=true：调用/成功/失败计数 + 最后活跃时间（wire string，null-safe 渲染，
 * 缺失则整行不渲染）。available=false：按 reason 渲染灰色说明文案。
 */
function StatsBlock({
  itemKey,
  isAdmin,
  stats,
}: Readonly<{
  itemKey: string;
  isAdmin: boolean;
  stats: RuntimeStats;
}>) {
  // 非 Admin：admin_only 语义——整个统计区不渲染。
  if (!isAdmin || stats.unavailable_reason === "admin_only") {
    return null;
  }

  if (!stats.available) {
    return (
      <div
        className="mt-3 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        data-testid={`stats-block-${itemKey}`}
      >
        {statsUnavailableCopy(stats.unavailable_reason)}
      </div>
    );
  }

  return (
    <div
      className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
      data-testid={`stats-block-${itemKey}`}
    >
      <span>调用 {stats.call_count}</span>
      <span>成功 {stats.success_count}</span>
      <span>失败 {stats.failure_count}</span>
      {stats.last_active_at ? (
        <span>最后活跃：{stats.last_active_at}</span>
      ) : null}
    </div>
  );
}

function UserSwitch({
  itemKey,
  config,
  disabled,
  onToggle,
}: Readonly<{
  itemKey: string;
  config: RuntimeConfigStatus;
  disabled: boolean;
  onToggle: (checked: boolean) => void;
}>) {
  // user_enablement_unknown：用户偏好暂不可用——disabled + tooltip（config_unreadable
  // 由外部 disabled prop 一并禁用）。
  const unknownUser = config.reason_code === "user_enablement_unknown";
  const control = (
    <Switch
      className="data-[state=checked]:bg-primary"
      data-testid={`user-switch-${itemKey}`}
      checked={config.enabled_user ?? false}
      disabled={disabled || unknownUser}
      onCheckedChange={onToggle}
    />
  );
  if (unknownUser) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex">{control}</span>
        </TooltipTrigger>
        <TooltipContent>用户偏好暂不可用</TooltipContent>
      </Tooltip>
    );
  }
  return control;
}

// D1a Task 26 — Admin 治理徽章行（status/trust_origin/unpinned/scan_verdict）。
function GovernanceBadges({
  itemKey,
  governance,
}: Readonly<{ itemKey: string; governance: GovernanceBlock }>) {
  return (
    <div
      className="flex flex-wrap items-center gap-1.5"
      data-testid={`gov-badges-${itemKey}`}
    >
      <span
        className="rounded px-1.5 py-0.5 text-xs font-medium bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200"
        data-testid={`gov-status-${itemKey}`}
      >
        {GOV_STATUS_LABELS[governance.status]}
      </span>
      <span
        className="rounded px-1.5 py-0.5 text-xs font-medium bg-muted/60 text-muted-foreground"
        data-testid={`gov-trust-${itemKey}`}
      >
        来源：{governance.trust_origin}
      </span>
      {governance.unpinned ? (
        <span
          className="rounded px-1.5 py-0.5 text-xs font-medium bg-amber-100 text-amber-700"
          data-testid={`gov-unpinned-${itemKey}`}
        >
          未固定
        </span>
      ) : null}
      {governance.scan_verdict !== null ? (
        <span
          className="rounded px-1.5 py-0.5 text-xs font-medium bg-muted text-muted-foreground"
          data-testid={`gov-scan-${itemKey}`}
        >
          {govScanLabel(governance.scan_verdict)}
        </span>
      ) : null}
    </div>
  );
}

// D1a Task 26 — Admin 治理行内动作（reapprove/quarantine/governance-disable/enable，带确认）。
// revision 从 governance.row_revision 取真值传给 store CAS（绝不伪造）。
function GovernanceActions({
  item,
  governance,
  disabled,
  onQuarantine,
  onReapprove,
  onGovernanceToggle,
}: Readonly<{
  item: RuntimeExtensionItem;
  governance: GovernanceBlock;
  disabled: boolean;
  onQuarantine: (kind: ExtensionKind, id: string, revision: number) => void;
  onReapprove: (kind: ExtensionKind, id: string, revision: number) => void;
  onGovernanceToggle: (
    kind: ExtensionKind,
    id: string,
    enabled: boolean,
    revision: number
  ) => void;
}>) {
  const { kind, id } = item;
  const rev = governance.row_revision;
  const confirmThen = (message: string, run: () => void) => {
    if (window.confirm(message)) {
      run();
    }
  };
  const btnClass = "h-7 rounded-lg border-border px-2 text-xs";
  return (
    <div className="flex flex-wrap gap-2" data-testid={`gov-actions-${kind}:${id}`}>
      {governance.status === "quarantined" ? (
        <Button
          variant="outline"
          size="sm"
          className={btnClass}
          data-testid={`gov-reapprove-${kind}:${id}`}
          disabled={disabled}
          onClick={() =>
            confirmThen("确认解除隔离并把当前观测重新 pin 为可信基线？", () =>
              onReapprove(kind, id, rev)
            )
          }
        >
          解除隔离
        </Button>
      ) : null}
      {governance.status === "active" ? (
        <>
          <Button
            variant="outline"
            size="sm"
            className={btnClass}
            data-testid={`gov-quarantine-${kind}:${id}`}
            disabled={disabled}
            onClick={() =>
              confirmThen("确认隔离该扩展？隔离后将不可被 Agent 调用。", () =>
                onQuarantine(kind, id, rev)
              )
            }
          >
            隔离
          </Button>
          <Button
            variant="outline"
            size="sm"
            className={btnClass}
            data-testid={`gov-disable-${kind}:${id}`}
            disabled={disabled}
            onClick={() =>
              confirmThen("确认治理停用该扩展？", () =>
                onGovernanceToggle(kind, id, false, rev)
              )
            }
          >
            治理停用
          </Button>
        </>
      ) : null}
      {governance.status === "disabled" ? (
        <Button
          variant="outline"
          size="sm"
          className={btnClass}
          data-testid={`gov-enable-${kind}:${id}`}
          disabled={disabled}
          onClick={() =>
            confirmThen("确认治理启用该扩展？", () =>
              onGovernanceToggle(kind, id, true, rev)
            )
          }
        >
          治理启用
        </Button>
      ) : null}
    </div>
  );
}

// D1a Task 26 — plugin 成员子行展开（Admin-only，惰性拉取 GET /v2/plugins）。
function PluginMembership({
  itemKey,
  extId,
  memberCount,
}: Readonly<{ itemKey: string; extId: string; memberCount: number }>) {
  const [expanded, setExpanded] = useState(false);
  const [members, setMembers] = useState<PluginMemberDetail[] | null>(null);
  const [loadError, setLoadError] = useState(false);

  async function toggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && members === null && !loadError) {
      try {
        const plugins = await governanceApi.getPlugins();
        const detail = plugins.find((p) => p.ext_id === extId);
        setMembers(detail?.members ?? []);
      } catch {
        setLoadError(true);
      }
    }
  }

  return (
    <div className="mt-3">
      <Button
        variant="ghost"
        size="sm"
        className="h-7 rounded-lg px-2 text-xs text-muted-foreground"
        data-testid={`plugin-expand-${itemKey}`}
        onClick={() => void toggle()}
      >
        {expanded ? "收起成员" : `展开成员（${memberCount}）`}
      </Button>
      {expanded ? (
        <div
          className="mt-2 space-y-1 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
          data-testid={`plugin-members-${itemKey}`}
        >
          {loadError ? (
            <p>成员加载失败</p>
          ) : members === null ? (
            <p>加载成员中…</p>
          ) : members.length === 0 ? (
            <p>无成员</p>
          ) : (
            members.map((m) => (
              <div
                key={m.declared_component_id}
                className="flex flex-wrap items-center gap-x-3 gap-y-0.5"
              >
                <span className="font-medium text-foreground/85">
                  {m.declared_component_id}
                </span>
                <span>{KIND_LABELS[m.kind as ExtensionKind] ?? m.kind}</span>
                <span className="font-mono">{m.ext_id}</span>
                <span>{m.status}</span>
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}

function ExtensionCard({
  item,
  isAdmin,
  governanceActive,
  probeEnabled,
  pending,
  notice,
  cooldownReadyAt,
  now,
  onSetGlobal,
  onSetUser,
  onProbe,
  onQuarantine,
  onReapprove,
  onGovernanceToggle,
}: Readonly<{
  item: RuntimeExtensionItem;
  isAdmin: boolean;
  governanceActive: boolean;
  probeEnabled: boolean;
  pending: boolean;
  notice: string | undefined;
  cooldownReadyAt: number | undefined;
  now: number;
  onSetGlobal: (kind: ExtensionKind, id: string, enabled: boolean) => void;
  onSetUser: (kind: ExtensionKind, id: string, enabled: boolean) => void;
  onProbe: (kind: ExtensionKind, id: string) => void;
  onQuarantine: (kind: ExtensionKind, id: string, revision: number) => void;
  onReapprove: (kind: ExtensionKind, id: string, revision: number) => void;
  onGovernanceToggle: (
    kind: ExtensionKind,
    id: string,
    enabled: boolean,
    revision: number
  ) => void;
}>) {
  const { health, liveness, config } = item;
  const isFaulted = health.state === "unreachable" || health.state === "error";
  const itemKey = `${item.kind}:${item.id}`;
  const isPlugin = item.kind === "plugin";
  // 治理块只在 Admin + governanceActive + 块存在时渲染（三层门；R7#7 非 Admin 恒不渲染）。
  const governance = item.governance;
  const showGovernance = isAdmin && governanceActive && governance !== undefined;
  // plugin 父级启停需真实 revision（governance 块必带）；缺失 → 全局 Switch 禁用（绝不 `?? 0`）。
  const pluginRevisionMissing =
    isPlugin && item.governance?.row_revision === undefined;

  // 非 Admin 的 unreachable/error 卡片：泛化文案（R8#2）。
  const showGenericFault = !isAdmin && isFaulted;

  // config_unreadable：双 Switch 禁用（本地状态不可信）。
  const configUnreadable = config.reason_code === "config_unreadable";

  // probe 按钮隐藏判据（R13#1，对齐 spec §3.1/§3.5）：仅当 enabled_global===false
  // （含 config_unreadable 保守占位 false）、或 probe_enabled=false（面板级）、或非 Admin
  // 时隐藏。disabled_user（global on + user off）条目仍显示可探测——user 级禁用不影响
  // 共享 health/probe。
  // plugin 行隐藏 probe 按钮（元容器无握手/短路重扫语义）。
  const showProbeButton =
    isAdmin && probeEnabled && config.enabled_global === true && !isPlugin;

  const cooldownRemainingMs =
    typeof cooldownReadyAt === "number" ? cooldownReadyAt - now : 0;
  const inCooldown = cooldownRemainingMs > 0;
  const cooldownSeconds = inCooldown ? Math.ceil(cooldownRemainingMs / 1000) : 0;

  return (
    <div
      className="rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]"
      data-testid="extension-card"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-base font-semibold text-foreground">{item.name}</p>
            <Badge variant="secondary" className="rounded-md bg-muted text-foreground/85">
              {KIND_LABELS[item.kind]}
            </Badge>
            <HealthBadge health={health} />
            {/* 非 Admin liveness 徽章一律不渲染（R16#4）。 */}
            {isAdmin ? <LivenessBadge liveness={liveness} /> : null}
          </div>
          {item.description ? (
            <p className="text-sm text-muted-foreground">{item.description}</p>
          ) : null}
          <p className="text-xs text-muted-foreground">
            配置：{config.effective_enabled ? "已启用" : "已禁用"}
            <span className="ml-1 opacity-70">（{config.reason_code}）</span>
          </p>
          {showGovernance && governance ? (
            <GovernanceBadges itemKey={itemKey} governance={governance} />
          ) : null}
        </div>

        {/* 操作区：全局 Switch（Admin）/ 用户级 Switch / probe 按钮。 */}
        <div className="flex shrink-0 flex-col items-end gap-2">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            {isAdmin ? (
              <label className="flex items-center gap-1.5">
                全局
                <Switch
                  className="data-[state=checked]:bg-primary"
                  data-testid={`global-switch-${itemKey}`}
                  checked={config.enabled_global}
                  disabled={pending || configUnreadable || pluginRevisionMissing}
                  onCheckedChange={(checked) =>
                    onSetGlobal(item.kind, item.id, checked)
                  }
                />
              </label>
            ) : null}
            {/* plugin 无 per-user 启停语义——隐藏个人 Switch（父级启停走 Admin 全局）。 */}
            {!isPlugin ? (
              <label className="flex items-center gap-1.5">
                个人
                <UserSwitch
                  itemKey={itemKey}
                  config={config}
                  disabled={pending || configUnreadable}
                  onToggle={(checked) => onSetUser(item.kind, item.id, checked)}
                />
              </label>
            ) : null}
          </div>
          {showProbeButton ? (
            <Button
              variant="outline"
              size="sm"
              className="h-7 rounded-lg border-border px-2 text-xs"
              data-testid={`probe-button-${itemKey}`}
              disabled={pending || inCooldown}
              onClick={() => onProbe(item.kind, item.id)}
            >
              {inCooldown
                ? `${probeButtonLabel(item.kind)}（${cooldownSeconds}s）`
                : probeButtonLabel(item.kind)}
            </Button>
          ) : null}
        </div>
      </div>

      {/* per-item inline 提示（R2#7：409 extension_disabled 等承载于 runtimeItemNotices）。 */}
      {notice ? (
        <p
          className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          role="status"
          data-testid={`item-notice-${itemKey}`}
        >
          {notice}
        </p>
      ) : null}

      {showGenericFault ? (
        <p
          className="mt-3 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
          data-testid="generic-fault-copy"
        >
          该扩展当前不可达/异常，详情仅管理员可见。
        </p>
      ) : null}

      {isAdmin && isFaulted ? (
        <div
          className="mt-3 space-y-1 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
          data-testid="admin-fault-detail"
        >
          {health.error_code || health.error_message ? (
            <p>
              错误：
              {health.error_code ? (
                <code className="font-mono">{health.error_code}</code>
              ) : null}
              {health.error_message ? (
                <span className="ml-1">{health.error_message}</span>
              ) : null}
            </p>
          ) : null}
          {typeof health.consecutive_failures === "number" &&
          health.consecutive_failures > 0 ? (
            <p>连续失败：{health.consecutive_failures} 次</p>
          ) : null}
          {health.next_probe_at ? (
            <p>下次探测：{health.next_probe_at}</p>
          ) : null}
        </div>
      ) : null}

      {isAdmin ? (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {typeof health.latency_ms === "number" ? (
            <span>握手 {health.latency_ms}ms</span>
          ) : null}
          {health.last_checked_at ? (
            <span>最后探测：{health.last_checked_at}</span>
          ) : null}
        </div>
      ) : null}

      {/* 调用统计区（Task 24）。非 Admin / admin_only 时内部早退不渲染。 */}
      <StatsBlock itemKey={itemKey} isAdmin={isAdmin} stats={item.stats} />

      {/* D1a Task 26 — 治理区（Admin + governanceActive + 块存在）：隔离横幅 + 行内动作。 */}
      {showGovernance && governance ? (
        <div className="mt-3 space-y-2" data-testid={`gov-section-${itemKey}`}>
          {governance.status === "quarantined" ? (
            <p
              className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-800 dark:text-red-200"
              role="status"
              data-testid={`gov-quarantine-banner-${itemKey}`}
            >
              已隔离：{govQuarantineReasonLabel(governance.quarantine_reason)}——已停止被 Agent 调用。
            </p>
          ) : null}
          <GovernanceActions
            item={item}
            governance={governance}
            disabled={pending}
            onQuarantine={onQuarantine}
            onReapprove={onReapprove}
            onGovernanceToggle={onGovernanceToggle}
          />
        </div>
      ) : null}

      {/* D1a Task 26 — plugin 成员展开（Admin-only，惰性拉取；item.kind 内联收窄 details）。 */}
      {isAdmin && item.kind === "plugin" ? (
        <PluginMembership
          itemKey={itemKey}
          extId={item.id}
          memberCount={item.details.member_count}
        />
      ) : null}
    </div>
  );
}

// 催填风险确认文案（R6#5/§7 冻结）——stdio 命令/外部 URL 有 RCE/SSRF 面，
// 用户须自行核实来源。不代填 secrets、不自动连接。
const PREFILL_RISK_COPY =
  "该模板将以 stdio 命令/外部 URL 运行，请自行核实来源后再保存";

/**
 * catalog "填入配置" → 风险确认 → 回调宿主完整包裹形态 JSON（Task 24, P-12 / R4#3）。
 * 完整包裹 = `{ mcpServers: { [catalog.id]: config_template } }`——满足
 * normalizeMCPConfigInput 校验器契约（顶层须 mcpServers 键，mcp-config.ts:48）。
 * 父组件回调实现：setActiveTab("mcp") + 预填局部 state + 打开既有 MCP 添加弹窗。
 */
function buildPrefillPayload(entry: RuntimeCatalogItem): string {
  return JSON.stringify(
    { mcpServers: { [entry.id]: entry.config_template } },
    null,
    2
  );
}

export function ExtensionsOverview({
  isAdmin,
  onSelectTab,
  onPrefillMcpConfig,
}: Readonly<ExtensionsOverviewProps>) {
  const runtimeExtensions = useSettingsStore((state) => state.runtimeExtensions);
  const runtimeSnapshotMeta = useSettingsStore(
    (state) => state.runtimeSnapshotMeta
  );
  const runtimeCatalog = useSettingsStore((state) => state.runtimeCatalog);
  const isRuntimeLoading = useSettingsStore((state) => state.isRuntimeLoading);
  const runtimeLoadError = useSettingsStore((state) => state.runtimeLoadError);
  const runtimePendingIds = useSettingsStore((state) => state.runtimePendingIds);
  const runtimeProbeCooldowns = useSettingsStore(
    (state) => state.runtimeProbeCooldowns
  );
  const runtimeItemNotices = useSettingsStore(
    (state) => state.runtimeItemNotices
  );
  const loadRuntimeExtensions = useSettingsStore(
    (state) => state.loadRuntimeExtensions
  );
  const loadRuntimeCatalog = useSettingsStore(
    (state) => state.loadRuntimeCatalog
  );
  const setRuntimeExtensionEnabled = useSettingsStore(
    (state) => state.setRuntimeExtensionEnabled
  );
  const setRuntimeUserEnabled = useSettingsStore(
    (state) => state.setRuntimeUserEnabled
  );
  const probeRuntimeExtension = useSettingsStore(
    (state) => state.probeRuntimeExtension
  );
  const runtimeGovernanceSummary = useSettingsStore(
    (state) => state.runtimeGovernanceSummary
  );
  const quarantineExtension = useSettingsStore(
    (state) => state.quarantineExtension
  );
  const reapproveExtension = useSettingsStore(
    (state) => state.reapproveExtension
  );
  const setGovernanceEnabled = useSettingsStore(
    (state) => state.setGovernanceEnabled
  );
  const approveAllPins = useSettingsStore((state) => state.approveAllPins);

  const pendingSet = useMemo(
    () => new Set(runtimePendingIds),
    [runtimePendingIds]
  );
  const now = useCooldownTick(runtimeProbeCooldowns);

  // Mount 驱动的轮询生命周期（R6#6）。样板 session-cost-summary.tsx。
  // cleanup 清定时器 + 显式 invalidateRuntimeRequests()（R2#5）——从 getState()
  // 读，避免把 store action 引用带进依赖数组（这些 action 引用稳定，[] 依赖足够）。
  //
  // 顺序 await（R6#2 冻结契约的必然结果）：loadRuntimeExtensions 与
  // loadRuntimeCatalog 共享同一模块级 request token（settings-store
  // runtimeRequestSeq）。若并发触发（两次 ++ 背靠背），先发的 extensions 响应会
  // 因 `token !== runtimeRequestSeq` 提前 return 而丢弃写入。故先 await 扩展加载
  // 落库，再启动目录加载——两者都是最新 token 时写入。unmount 时 cleanup 仍同步
  // 调 invalidateRuntimeRequests() 失效任何 in-flight 响应。
  // PR4A-R1 Audit Fix A — 守卫"卸载后 catalog 链继续跑"：
  // 卸载/清理 bump token 后，晚到的 extensions 请求在 store 内早退，但组件链仍会
  // 走到 loadRuntimeCatalog()——它会再 ++token 并在新 token 下写 runtimeCatalog
  // （卸载后写入；老周期的 catalog 调用也可能使更新周期的 in-flight extensions 失效）。
  // 两 await 之间（及首 await 前）用 generation 守卫；不改 store 的 token 语义。
  //
  // P3 修复：改用 generation 计数器而非单 boolean。StrictMode 的 effect replay
  // （setup → cleanup → setup）会把同一个 boolean ref 重新置 true，第一代残留的
  // 晚到链会穿过守卫并触发 catalog（bump 共享 token）。generation 计数器让每次
  // setup 捕获独立的 gen；cleanup 递增 gen 使该代所有链失效。正常 mount/unmount
  // 行为不变，replay 的两次 setup 拿到不同 gen，第一代链被 gen 不匹配拦截。
  const loadGenRef = useRef(0);
  useEffect(() => {
    const gen = ++loadGenRef.current;
    const load = async () => {
      if (loadGenRef.current !== gen) {
        return;
      }
      await loadRuntimeExtensions();
      if (loadGenRef.current !== gen) {
        return;
      }
      await loadRuntimeCatalog();
    };
    void load();
    const id = window.setInterval(() => {
      void load();
    }, POLL_INTERVAL_MS);
    return () => {
      // loadGenRef 是纯 generation 计数器（非 DOM 节点 ref）——cleanup 故意读改
      // .current 以失效本代所有链，这正是设计意图，非 stale-node 陷阱。
      // eslint-disable-next-line react-hooks/exhaustive-deps
      loadGenRef.current++;
      window.clearInterval(id);
      useSettingsStore.getState().invalidateRuntimeRequests();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 治理是否开启的 FE 侧唯一可信信号：后端仅在 governance≠off 时给扩展条目挂 `governance`
  // 块（off 时 None-omit）。据此门控 summary 拉取——OFF 部署零 governance-summary 请求。
  const hasGovernanceBlock = useMemo(
    () => runtimeExtensions.some((item) => item.governance !== undefined),
    [runtimeExtensions]
  );

  // D1a Task 26 — 治理摘要轮询（R3#11：端点全 AdminUser，非 Admin 挂载会周期性 403，故仅
  // isAdmin）。Finding #1（off-mode）：再叠加 hasGovernanceBlock 门——治理 OFF（无 governance
  // 块）时不拉取、不起 interval，避免每 30s 打一次 summary 端点（pre-D1a 从无此 I/O）。
  // 独立 effect（[isAdmin, hasGovernanceBlock] 依赖）+ generation 守卫（StrictMode replay
  // 安全，语义同上）。fetchGovernanceSummary 经 getState() 读，稳定引用不入依赖数组；其内部
  // 另有 mode:"off" back-off latch 作为 belt-and-suspenders（防不一致快照下的空转轮询）。
  const govGenRef = useRef(0);
  useEffect(() => {
    if (!isAdmin || !hasGovernanceBlock) {
      return;
    }
    const gen = ++govGenRef.current;
    const loadGov = async () => {
      if (govGenRef.current !== gen) {
        return;
      }
      await useSettingsStore.getState().fetchGovernanceSummary();
    };
    void loadGov();
    const id = window.setInterval(() => {
      void loadGov();
    }, POLL_INTERVAL_MS);
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      govGenRef.current++;
      window.clearInterval(id);
    };
  }, [isAdmin, hasGovernanceBlock]);

  // catalog "已配置" 徽章：catalog.id 与已配置 mcp 条目 id 精确匹配。
  const configuredMcpIds = useMemo(() => {
    const ids = new Set<string>();
    for (const item of runtimeExtensions) {
      if (item.kind === "mcp") {
        ids.add(item.id);
      }
    }
    return ids;
  }, [runtimeExtensions]);

  const probeDisabled = runtimeSnapshotMeta?.probe_enabled === false;
  const statsDisabled = runtimeSnapshotMeta?.stats_enabled === false;

  // D1a Task 26 — 治理 UI 总门：Admin + summary 已拉取 + mode≠off。非 Admin/off → 全隐藏。
  const governanceMode = runtimeGovernanceSummary?.mode;
  const governanceActive =
    isAdmin && governanceMode !== undefined && governanceMode !== "off";
  const unpinnedCount = runtimeGovernanceSummary?.unpinned_count ?? 0;
  const quarantinedCount = runtimeGovernanceSummary?.quarantined_count ?? 0;

  function handleApprovePins(): void {
    if (
      !window.confirm(
        `确认把 ${unpinnedCount} 个待 pin 观测批量转正为可信基线？`
      )
    ) {
      return;
    }
    void approveAllPins();
  }

  // "填入配置"：风险确认 → 回调宿主完整包裹 JSON（Task 24, P-12）。confirm 拒绝零调用。
  function handlePrefill(entry: RuntimeCatalogItem): void {
    if (!onPrefillMcpConfig) {
      return;
    }
    if (!window.confirm(PREFILL_RISK_COPY)) {
      return;
    }
    onPrefillMcpConfig(buildPrefillPayload(entry));
  }

  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-2xl font-semibold tracking-tight text-foreground">
          扩展总览
        </h3>
        <p className="text-sm text-muted-foreground">
          汇总 MCP / A2A / Skill 扩展的配置、健康与活跃状态（只读）。
        </p>
      </div>

      {isRuntimeLoading && runtimeExtensions.length === 0 ? (
        <p className="text-sm text-muted-foreground">正在加载扩展状态…</p>
      ) : null}

      {runtimeLoadError && runtimeExtensions.length > 0 ? (
        <div
          className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
          role="status"
          data-testid="runtime-load-error"
        >
          列表刷新失败（显示为旧数据）：{runtimeLoadError}
        </div>
      ) : null}

      {/* 面板级横幅：探测/统计能力未启用。 */}
      {probeDisabled ? (
        <div
          className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          role="status"
          data-testid="probe-disabled-banner"
        >
          健康探测未启用（extension_probe_enabled）——状态列仅反映配置。
        </div>
      ) : null}
      {statsDisabled ? (
        <div
          className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          role="status"
          data-testid="stats-disabled-banner"
        >
          调用统计未启用（extension_stats_enabled）——状态列仅反映配置。
        </div>
      ) : null}

      {/* D1a Task 26 — 治理工具栏（Admin + mode≠off）：模式/计数提示 + 批量 pin 转正。 */}
      {governanceActive ? (
        <div
          className="flex flex-wrap items-center gap-3 rounded-md border border-border/70 bg-muted/30 px-3 py-2 text-xs"
          data-testid="governance-toolbar"
        >
          <span className="text-muted-foreground">
            治理模式：{governanceMode === "enforce" ? "强制" : "影子"}
            {unpinnedCount > 0 ? ` · 待 pin ${unpinnedCount}` : ""}
            {quarantinedCount > 0 ? ` · 已隔离 ${quarantinedCount}` : ""}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 rounded-lg border-border px-2 text-xs"
            data-testid="approve-pins-button"
            disabled={unpinnedCount === 0}
            onClick={handleApprovePins}
          >
            批量转正 pin（{unpinnedCount}）
          </Button>
        </div>
      ) : null}

      {/* 已配置扩展区。 */}
      <div className="space-y-3 rounded-2xl border border-border/70 bg-muted/30 p-3">
        {runtimeExtensions.length === 0 ? (
          <div className="rounded-xl border border-dashed bg-card p-6 text-center text-sm text-muted-foreground">
            尚无已配置的扩展——从下方推荐目录开始。
          </div>
        ) : (
          runtimeExtensions.map((item) => {
            const itemKey = `${item.kind}:${item.id}`;
            return (
              <ExtensionCard
                key={itemKey}
                item={item}
                isAdmin={isAdmin}
                governanceActive={governanceActive}
                probeEnabled={runtimeSnapshotMeta?.probe_enabled === true}
                pending={pendingSet.has(itemKey)}
                notice={runtimeItemNotices[itemKey]}
                cooldownReadyAt={runtimeProbeCooldowns[itemKey]}
                now={now}
                onSetGlobal={setRuntimeExtensionEnabled}
                onSetUser={setRuntimeUserEnabled}
                onProbe={probeRuntimeExtension}
                onQuarantine={quarantineExtension}
                onReapprove={reapproveExtension}
                onGovernanceToggle={setGovernanceEnabled}
              />
            );
          })
        )}
      </div>

      {/* 管理区跳转。 */}
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          className="h-9 rounded-xl border-border"
          onClick={() => onSelectTab("mcp")}
        >
          管理 MCP 服务器
        </Button>
        <Button
          variant="outline"
          className="h-9 rounded-xl border-border"
          onClick={() => onSelectTab("a2a")}
        >
          管理 A2A Agent
        </Button>
        <Button
          variant="outline"
          className="h-9 rounded-xl border-border"
          onClick={() => onSelectTab("skill")}
        >
          管理 Skill
        </Button>
      </div>

      {/* 可用区：推荐目录（只读，无"填入配置"按钮——4b）。 */}
      {runtimeCatalog.length > 0 ? (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-foreground">推荐目录</h4>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {runtimeCatalog.map((entry) => {
              const configured = configuredMcpIds.has(entry.id);
              const homepageHref = safeHomepage(entry.homepage);
              return (
                <div
                  key={entry.id}
                  className="space-y-2 rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]"
                  data-testid="catalog-card"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-sm font-semibold text-foreground">
                      {entry.name}
                    </p>
                    {configured ? (
                      <Badge
                        variant="secondary"
                        className="rounded-md bg-emerald-100 text-emerald-700"
                      >
                        已配置
                      </Badge>
                    ) : null}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {entry.description}
                  </p>
                  {entry.tags.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {entry.tags.map((tag) => (
                        <Badge
                          key={`${entry.id}-${tag}`}
                          variant="outline"
                          className="rounded-md"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  ) : null}
                  <div className="flex flex-wrap items-center gap-3 pt-1">
                    {/* "填入配置"：仅在 Admin 且父组件传入回调时渲染（P2 修复）。
                        非 Admin 的整个 mutation surface（含"添加服务器"入口）已禁用，
                        prefill 也必须与之一致——即便后端 AdminUser 仍 403 写，UI 不得
                        暴露预填入口。4a 无回调时同样不渲染。
                        confirm 风险提示 → 完整包裹 JSON 交宿主预填 MCP 添加弹窗。 */}
                    {isAdmin && onPrefillMcpConfig ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 rounded-lg border-border px-2 text-xs"
                        data-testid={`prefill-button-${entry.id}`}
                        disabled={configured}
                        onClick={() => handlePrefill(entry)}
                      >
                        填入配置
                      </Button>
                    ) : null}
                    {homepageHref ? (
                      <a
                        href={homepageHref}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary underline-offset-4 hover:underline"
                      >
                        主页
                      </a>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
