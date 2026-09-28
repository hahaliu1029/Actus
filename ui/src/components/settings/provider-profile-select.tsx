const PROFILES = [
  ["generic_openai", "通用 OpenAI 兼容"],
  ["openai_official", "OpenAI 官方"],
  ["glm", "GLM（旧型号兼容）"],
  ["glm_5_2", "GLM-5.2 标准 API"],
  ["glm_5_2_coding", "GLM-5.2 Coding Plan"],
  ["anthropic_compat", "Claude（OpenAI 兼容接口）"],
  ["gemini_compat", "Gemini（OpenAI 兼容接口）"],
  ["deepseek_chat", "DeepSeek Chat"],
  ["deepseek_reasoner", "DeepSeek Reasoner"],
  ["kimi_k2", "Kimi K2"],
  ["kimi_k2_6", "Kimi K2.6"],
  ["dashscope_qwen", "DashScope Qwen"],
  ["dashscope_qwen_vl", "DashScope Qwen VL"],
  ["minimax", "MiniMax"],
] as const;

export function ProviderProfileSelect({ label, value, onChange }: Readonly<{
  label: string;
  value?: string | null;
  onChange: (provider: string | null) => void;
}>) {
  return (
    <label className="text-sm text-foreground/85">
      {label}
      <select
        aria-label={label}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value || null)}
        className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
      >
        <option value="">自动识别（按地址和模型）</option>
        {value && !PROFILES.some(([id]) => id === value) ? (
          <option value={value}>{value}（未识别，请重新选择）</option>
        ) : null}
        {PROFILES.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
      </select>
      <span className="mt-1 block text-xs text-muted-foreground">
        代理域名可显式选择兼容策略；协议仍由 API 类型决定。修改地址或模型将重置为自动识别。
      </span>
    </label>
  );
}
