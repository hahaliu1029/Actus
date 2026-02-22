import type { MCPConfig, MCPServerConfig, MCPTransport } from "@/lib/api/types";

type NormalizeMCPConfigResult =
  | { ok: true; config: MCPConfig }
  | { ok: false; error: string };

const VALID_TRANSPORTS = new Set<MCPTransport>(["stdio", "sse", "streamable_http"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function normalizeTransportValue(value: unknown): MCPTransport | null {
  if (typeof value !== "string") {
    return null;
  }
  return VALID_TRANSPORTS.has(value as MCPTransport) ? (value as MCPTransport) : null;
}

function validateServerConfig(
  serverName: string,
  config: MCPServerConfig,
  transport: MCPTransport | undefined
): string | null {
  if (!transport) {
    return null;
  }
  if (transport === "stdio" && !asNonEmptyString(config.command)) {
    return `MCP 服务器 ${serverName} 在 stdio 模式下必须传递 command`;
  }
  if (
    (transport === "sse" || transport === "streamable_http") &&
    !asNonEmptyString(config.url)
  ) {
    return `MCP 服务器 ${serverName} 在 ${transport} 模式下必须传递 url`;
  }
  return null;
}

export function normalizeMCPConfigInput(input: unknown): NormalizeMCPConfigResult {
  if (!isRecord(input) || !isRecord(input.mcpServers)) {
    return {
      ok: false,
      error: "MCP 配置中缺少 mcpServers 字段",
    };
  }

  const normalizedServers: Record<string, MCPServerConfig> = {};
  for (const [serverName, rawServer] of Object.entries(input.mcpServers)) {
    if (!isRecord(rawServer)) {
      return {
        ok: false,
        error: `MCP 服务器 ${serverName} 配置格式不正确`,
      };
    }

    const rawTransport = rawServer.transport ?? rawServer.type;
    let transport = normalizeTransportValue(rawTransport) ?? undefined;
    if (rawTransport != null && !transport) {
      return {
        ok: false,
        error: `MCP 服务器 ${serverName} 的 transport/type 仅支持 stdio、sse、streamable_http`,
      };
    }

    const normalized = {
      ...(rawServer as MCPServerConfig),
    };

    const hasCommand = Boolean(asNonEmptyString(normalized.command));
    const hasUrl = Boolean(asNonEmptyString(normalized.url));
    if (!transport) {
      if (hasCommand && !hasUrl) {
        transport = "stdio";
      } else if (hasUrl && !hasCommand) {
        transport = "streamable_http";
      }
    }
    if (transport) {
      normalized.transport = transport;
    }

    const validationError = validateServerConfig(serverName, normalized, transport);
    if (validationError) {
      return {
        ok: false,
        error: validationError,
      };
    }

    normalizedServers[serverName] = normalized;
  }

  return {
    ok: true,
    config: {
      mcpServers: normalizedServers,
    },
  };
}
