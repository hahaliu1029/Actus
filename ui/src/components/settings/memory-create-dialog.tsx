"use client";

import { useCallback, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api/auth-utils";
import { memoryApi } from "@/lib/api/memory";
import type { MemoryCategory } from "@/lib/api/types";
import {
  MEMORY_CATEGORY_CREATE_OPTIONS,
  MEMORY_TAGS_MAX_COUNT,
  MEMORY_TAG_MAX_LENGTH,
  parseMemoryTagsInput,
} from "@/lib/memory-utils";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import { LoaderCircle } from "lucide-react";

// 后端 `content` 限制：min 1, max 50000（见 schemas/memory_schemas.py）
// UI 侧提前校验，避免提交后挨后端 400。
const MAX_CONTENT_LENGTH = 50_000;

export interface MemoryCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MemoryCreateDialog({
  open,
  onOpenChange,
}: MemoryCreateDialogProps) {
  // 与 memory-detail-drawer 保持一致的 pattern：绕开 settings-store 的 action 层，
  // 直接调用 memoryApi + loadMemories 刷新。这样 4xx（409/429/400）等业务错误
  // 可以 inline 就地回显，而不是先被 store.reportError() 全局 toast 一次，再
  // 被 dialog 显示一次（双重告警）。非预期错误（网络/5xx）按 drawer 一致的
  // 约定：inline + 全局 toast 双通道，保证用户在关弹窗后仍能看到错误上下文。
  const loadMemories = useSettingsStore((state) => state.loadMemories);

  // 组件依赖父层条件渲染（``{open && <MemoryCreateDialog .../>}``），每次打开
  // 都是新 mount，初始 state 直接是干净值——不需要 "open 变 false 就手动 reset"
  // 的 useEffect，也就避免了 react-hooks/set-state-in-effect 告警。
  const [category, setCategory] = useState<MemoryCategory>("user");
  const [content, setContent] = useState<string>("");
  const [pinnedPreference, setPinnedPreference] = useState<boolean>(false);
  const [tagsInput, setTagsInput] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  // 409 / 429 等后端错误就地回显——不让 toast 独吞错误，用户在 dialog 里
  // 一眼看清为什么失败，然后决定改内容还是取消。
  const [inlineError, setInlineError] = useState<string | null>(null);

  // pinned=true 只在 category='user' 合法（后端 CHECK 约束 + service 校验）。
  // 这里用 derived state 而非 useEffect 复位：``pinnedPreference`` 是用户意愿，
  // 显示 / 提交用 ``effectivePinned``（category!=user 时自动为 false）。好处：
  // 用户从 user 切到 rule 再切回 user，他们最初的 pinned 选择会自动恢复，
  // 比 useEffect 强制复位的 "切出再切回就丢了" 更友好。
  const effectivePinned = category === "user" && pinnedPreference;

  const trimmedLen = useMemo(() => content.trim().length, [content]);
  // 每次 tagsInput 变 → 实时预览解析结果，让用户看到"写了 5 个 tag"或
  // "tag X 超过 64 字符被剔除"这类反馈；submit 时直接复用 parsed.tags。
  const parsedTags = useMemo(() => parseMemoryTagsInput(tagsInput), [tagsInput]);
  // 两类 tag 错误（overLimit / tooLong）都必须闸掉 submit：
  // - overLimit（>20 条）：后端 Field(max_length=20) 直接 422
  // - tooLong（>64 字符）：后端 _normalize_tags 对超长 tag 抛 ValueError → 422
  //
  // 如果前端只闸其中一条另一条允许提交，就会出现 "UI 带警告地丢一部分 tag
  // 后成功，curl 同样输入却 422" 的契约分裂。统一在前端就 disable，让警告
  // 真正有"阻塞"意义，而不是 "你看到了但还是可以点"。
  const isValid =
    trimmedLen >= 1 &&
    content.length <= MAX_CONTENT_LENGTH &&
    parsedTags.overLimit.length === 0 &&
    parsedTags.tooLong.length === 0;

  // 用户改内容 / 切分类 / 改 pinned → 清掉旧的 inlineError，避免"改了之后错误条
  // 还挂着" 误导用户以为新内容也撞 hash。只在用户动了可能修正错误的字段时清，
  // 避免在 isSubmitting 翻状态的 render 里意外清掉（setInlineError 只在 onChange 里调）。
  const handleContentChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setContent(e.target.value);
      if (inlineError) setInlineError(null);
    },
    [inlineError],
  );
  const handleCategorySelectChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setCategory(e.target.value as MemoryCategory);
      if (inlineError) setInlineError(null);
    },
    [inlineError],
  );
  const handlePinnedChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setPinnedPreference(e.target.checked);
      if (inlineError) setInlineError(null);
    },
    [inlineError],
  );
  const handleTagsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTagsInput(e.target.value);
      if (inlineError) setInlineError(null);
    },
    [inlineError],
  );

  // Dialog 外部（Esc / backdrop / 关闭 X）触发的 onOpenChange：提交中吞掉关闭
  // 请求。如果允许关闭，pending 的 create 请求继续在后台跑，成功后 reportSuccess
  // 会弹 "记忆已创建" toast，但用户已经关了窗——可能再点 "新建" 又提交同内容
  // 撞 409。"创建 + 取消" 按钮各自 disabled 已经处理了按钮触发，这里补外部触发。
  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (isSubmitting && !next) return;
      onOpenChange(next);
    },
    [isSubmitting, onOpenChange],
  );

  const handleSubmit = useCallback(async () => {
    if (!isValid || isSubmitting) return;
    setIsSubmitting(true);
    setInlineError(null);

    // Phase 1: mutation. 4xx（409/429/400）按业务语义 inline 回显，避免 toast
    // spam；非预期错误（网络/5xx）inline + 全局 toast 双通道（与 drawer 一致）。
    try {
      // 条件 spread 省略空 tags key——tags 为 "undefined 占位" 和 "key 不存在"
      // 在 JSON 序列化上都落不进 body，但在 JS 对象 + 单元测试断言层面不同。
      // key 不存在更符合 "无 tags 的请求就不该提这个字段" 的语义。
      await memoryApi.create({
        content: content.trim(),
        category,
        pinned: effectivePinned,
        ...(parsedTags.tags.length > 0 ? { tags: parsedTags.tags } : {}),
      });
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.httpStatus === 409) {
          setInlineError("这条内容已存在（相同内容 hash），请换个表述再试");
        } else if (err.httpStatus === 429) {
          setInlineError("今日记忆写入额度已用尽，请明日再试");
        } else if (err.httpStatus === 400) {
          setInlineError(err.message || "输入不合法");
        } else {
          // 5xx / 未分类——inline + 全局 toast 双通道
          const msg = err.message || "创建记忆失败";
          setInlineError(msg);
          useUIStore.getState().setMessage({ type: "error", text: msg });
        }
      } else {
        const msg = err instanceof Error ? err.message : "创建记忆失败";
        setInlineError(msg);
        useUIStore.getState().setMessage({ type: "error", text: msg });
      }
      setIsSubmitting(false);
      return;
    }

    // Phase 2: success toast + 关弹窗（在 refresh 之前关；refresh 是 best-effort）
    useUIStore.getState().setMessage({ type: "success", text: "记忆已创建" });
    onOpenChange(false);

    // Phase 3: refresh list（best-effort）。与 drawer 一致：mutation 已成功，
    // 刷新失败时由 loadMemories 自身的 memoryLoadError + toast 负责呈现；
    // dialog 已关，不需要再在这里处理错误。
    try {
      await loadMemories({ page: 1 });
    } catch {
      // intentionally swallowed; memoryLoadError 已上屏
    }
    setIsSubmitting(false);
  }, [category, content, effectivePinned, isSubmitting, isValid, loadMemories, onOpenChange, parsedTags]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-130">
        <DialogHeader>
          <DialogTitle>新建长期记忆</DialogTitle>
          <DialogDescription>
            手动添加一条长期记忆，后续对话会把它注入 prompt。
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4">
          {/* Category */}
          <div className="grid gap-2">
            <Label htmlFor="memory-create-category">分类</Label>
            <select
              id="memory-create-category"
              aria-label="选择记忆分类"
              value={category}
              onChange={handleCategorySelectChange}
              disabled={isSubmitting}
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {MEMORY_CATEGORY_CREATE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              {
                MEMORY_CATEGORY_CREATE_OPTIONS.find(
                  (o) => o.value === category
                )?.hint
              }
            </p>
          </div>

          {/* Content */}
          <div className="grid gap-2">
            <Label htmlFor="memory-create-content">内容</Label>
            <Textarea
              id="memory-create-content"
              aria-label="记忆内容"
              placeholder="写下要让 Agent 记住的事实、偏好或规则…"
              value={content}
              onChange={handleContentChange}
              disabled={isSubmitting}
              rows={6}
              maxLength={MAX_CONTENT_LENGTH}
              className="resize-y"
            />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>
                {trimmedLen === 0
                  ? "内容不能为空"
                  : `${content.length} / ${MAX_CONTENT_LENGTH} 字符`}
              </span>
            </div>
          </div>

          {/* Tags — 可选，逗号 / 换行分隔。parse 实时预览 + 超长剔除反馈 */}
          <div className="grid gap-2">
            <Label htmlFor="memory-create-tags">
              标签 <span className="text-muted-foreground">（可选）</span>
            </Label>
            <Input
              id="memory-create-tags"
              aria-label="标签（逗号分隔）"
              placeholder="用逗号或换行分隔，例如：Go, backend, 重要"
              value={tagsInput}
              onChange={handleTagsChange}
              disabled={isSubmitting}
            />
            <div className="text-xs text-muted-foreground">
              {parsedTags.tags.length === 0 ? (
                <span>
                  最多 {MEMORY_TAGS_MAX_COUNT} 条，每条 ≤{MEMORY_TAG_MAX_LENGTH} 字符
                </span>
              ) : (
                <div
                  className="flex flex-wrap gap-1"
                  data-testid="memory-create-tags-preview"
                >
                  {parsedTags.tags.map((t) => (
                    <span
                      key={t}
                      className="inline-flex items-center rounded-md bg-muted px-2 py-0.5"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}
              {parsedTags.tooLong.length > 0 && (
                <div
                  className="mt-1 text-destructive"
                  data-testid="memory-create-tags-too-long"
                  role="alert"
                >
                  以下标签超过 {MEMORY_TAG_MAX_LENGTH} 字符，请缩短后再创建：
                  {parsedTags.tooLong
                    .map((t) => `"${t.slice(0, 20)}…"`)
                    .join("、")}
                </div>
              )}
              {parsedTags.overLimit.length > 0 && (
                <div
                  className="mt-1 text-destructive"
                  data-testid="memory-create-tags-over-limit"
                  role="alert"
                >
                  最多 {MEMORY_TAGS_MAX_COUNT} 条标签，当前多出{" "}
                  {parsedTags.overLimit.length} 条，请删减后再创建。
                </div>
              )}
            </div>
          </div>

          {/* Pinned — 仅 user 可用 */}
          <label
            className={`flex items-start gap-2 rounded-md border p-3 text-sm ${
              category === "user"
                ? "cursor-pointer hover:bg-muted/30"
                : "cursor-not-allowed opacity-60"
            }`}
          >
            <input
              type="checkbox"
              aria-label="置顶该记忆（不受 recency 截断影响）"
              checked={effectivePinned}
              disabled={category !== "user" || isSubmitting}
              onChange={handlePinnedChange}
              className="mt-1 size-4 cursor-pointer accent-primary disabled:cursor-not-allowed"
            />
            <div className="space-y-1">
              <div className="font-medium">置顶（pinned）</div>
              <p className="text-xs text-muted-foreground">
                仅用户画像（user）可置顶。置顶记忆永远优先注入 prompt，不受
                recency / budget 截断。
              </p>
            </div>
          </label>

          {inlineError && (
            <div
              role="alert"
              data-testid="memory-create-error"
              className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
            >
              {inlineError}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
          >
            取消
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!isValid || isSubmitting}
            aria-label="创建记忆"
          >
            {isSubmitting ? (
              <LoaderCircle className="mr-1 size-4 animate-spin" />
            ) : null}
            创建
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
