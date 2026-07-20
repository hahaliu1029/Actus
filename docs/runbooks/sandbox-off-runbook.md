# Sandbox off 档部署 Runbook

> **档位语义**：`off` 禁止任何新容器创建、app 路径零 lifecycle mutation（spec §5.1 / INV-SPM-3）。
> 沙箱工具与 skill-creation 工具不注册、VNC/接管/skill-create 端点拒绝。存量遗留容器由
> **pre-switch drain + TTL** 回收（off 无 docker.sock，无法自清）。

## ⚠️ 顺序警告：先停流量并 drain，再切

**必须先从 still-on-socket 的 always/on_demand 部署 drain 遗留容器，再切 off override。**
off 容器起来后没有 `/var/run/docker.sock`，无法清理任何存量沙箱——顺序颠倒会留下长时间
遗留容器，只能等待其 TTL 到期或由 host 运维手工清理。

全新安装且从未启动过 Actus 时，下面的 drain 命令仍可安全执行并会成为 no-op；确认没有
历史 `actus.session_id` label 或自定义 `SANDBOX_NAME_PREFIX` 容器后再继续即可。

已有部署必须先进入维护窗口：在外层反向代理/防火墙阻止新请求，等待所有普通任务和包含
`parallel_work_units` 的 coordinator session 到达终态，然后停止 `ui`、`ui-app`、`api`。
直接把 coordinator flag 改为 `false` 不会终止正在运行的 coordinator；如果不先停流量并
停止旧 API，清理容器后到 off 重建前仍可能再次创建沙箱。

drain 作用域由 `SANDBOX_NAME_PREFIX` 决定，动态沙箱不属于 Compose project。**同一 Docker
daemon 上存在多个 Actus 部署时，每个部署必须预先使用唯一前缀；共享默认 `actus-sb`
前缀时停止执行并先人工拆分目标。** 不要使用全局 `label=actus.session_id` 删除，因为该 label
会跨项目匹配。下面先打印候选容器，只有确认它们全部属于当前部署后才能执行删除行。

## ⚠️ 预检：关闭 coordinator 三 flag（off × 任一 C2 flag=true → 启动 fail-fast）

`off` 模式不供给任何父沙箱，与需要沙箱的 C2 coordinator 分派路径互斥。三个 flag（
`ACTUS_C2_COORDINATOR_ENABLED` / `ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED` /
`ACTUS_C2_AGENT_TEAMS_ENABLED`）任一为 `true` 时，API 启动期
`check_sandbox_off_flag_exclusion()`（`api/app/main.py`）会抛
`RuntimeError: SANDBOX_PROVISION_MODE=off is incompatible with coordinator flags: [...]`，
容器起不来。

**当前 `.env.example` 模板默认 `ACTUS_C2_COORDINATOR_ENABLED=true`**（另两个默认
`false`；这与 `CONTRIBUTING.md` 所述“生产 rollout 策略默认关闭”是模板值与生产策略两个
口径），不关则 off 容器直接 fail-fast。下方命令 ②a 用**三键 upsert**（存在则改 `false`、
缺失则追加）确保三行确定为 `false`——replace-only 的 `sed` 补不出缺失键。

## Canonical 切换流程（实现者与运维不自行拼装）

先停入口并只预览候选，不删除：

```bash
# ⓪ 维护窗口：先在外层阻止新流量并等待所有 session 终态，再停旧入口与 API，消除重建竞态
docker compose --env-file .env stop ui ui-app api
# ① 预览 drain 候选（按当前部署唯一前缀做字面量前缀匹配，不把前缀解释为正则）
set -a; . ./.env 2>/dev/null; set +a
sandbox_target_prefix="${SANDBOX_NAME_PREFIX:-actus-sb}"
docker ps -a --format '{{.ID}}\t{{.Names}}\t{{.Status}}' | awk -v prefix="${sandbox_target_prefix}-" 'index($2, prefix) == 1'
```

**在这里暂停。** 确认预览中的每个容器都属于当前部署；若候选为空则删除是 no-op，若有任何
归属不确定则不要继续。确认完成后，另行执行下面的删除与切换代码块：

```bash
# ① pre-switch drain（重新加载同一 .env，并按字面量前缀只取已确认候选的 ID）
set -a; . ./.env 2>/dev/null; set +a
sandbox_target_prefix="${SANDBOX_NAME_PREFIX:-actus-sb}"
docker ps -a --format '{{.ID}}\t{{.Names}}' | awk -v prefix="${sandbox_target_prefix}-" 'index($2, prefix) == 1 {print $1}' | xargs -r docker rm -f
# ②a 关闭 coordinator 三 flag（R10 必需——off × 任一 C2 flag=true 在 API 启动期 fail-fast；
#     .env.example 默认 ACTUS_C2_COORDINATOR_ENABLED=true，不关则 off 容器起不来）。
#     **三键 upsert（R19-P2 修正——存在则改 false、缺失则追加）**：replace-only `sed` 补不出缺失键，
#     旧 .env / 默认模板缺 ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED 时「三行均 false」预检只显 2 行、
#     且该键在别处被置 true 时漏关。upsert 保证三行确定为 false：
for k in ACTUS_C2_COORDINATOR_ENABLED ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED ACTUS_C2_AGENT_TEAMS_ENABLED; do
  if grep -qE "^${k}=" .env; then sed -i.bak -E "s/^${k}=.*/${k}=false/" .env; else printf '%s=false\n' "$k" >> .env; fi
done
grep -cE '^ACTUS_C2_(COORDINATOR_ENABLED|COORDINATOR_SHELL_MODE_ENABLED|AGENT_TEAMS_ENABLED)=false' .env   # 期望 3
# ② off 部署（唯一 canonical 命令）
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env up -d --build
# ③ 验证（值级，同 CI 验收——四断言缺一不可）
test "$(docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config --services | grep -c '^sandbox-image$' || true)" -eq 0
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config | grep -A1 "SANDBOX_PROVISION_MODE" | grep -q "off"
test "$(docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config | grep -c "docker.sock" || true)" -eq 0
curl -fsS http://localhost:${API_PORT:-8000}/docs >/dev/null
```

## 回退（还原 always）

**正常切换到 off 时不要执行本节。** 只有明确决定退出 off 后，才单独运行以下回退命令：

```bash
# 必须显式 pin always：.env 中已有 off/on_demand 时，Compose 的 :-always 默认值不会生效
if grep -qE "^SANDBOX_PROVISION_MODE=" .env; then sed -i.bak -E "s/^SANDBOX_PROVISION_MODE=.*/SANDBOX_PROVISION_MODE=always/" .env; else printf 'SANDBOX_PROVISION_MODE=always\n' >> .env; fi
grep -E "^SANDBOX_PROVISION_MODE=" .env          # 期望 always
docker compose --env-file .env up -d --build     # 去 off override + .env pin=always → 真还原 always
```

该流程会**显式 upsert `SANDBOX_PROVISION_MODE=always` 到 `.env`**（不依赖 `:-always`
默认——`.env` 若含 `off`/`on_demand` 会盖默认）+ 去 off override 重启。回退后 base 单文件重新携带
`sandbox-image` service 与 `docker.sock` 挂载，always 供给面完整恢复。

如目标是直接回到 `on_demand`，把命令 ④ upsert 的目标值改为 `on_demand`，仍然去掉 off
override 并用标准 Compose 重建；不要先短暂启动 `always`。三个 coordinator flag 不会自动
恢复：只有在 API 健康、沙箱镜像/socket 结构验证通过，并重新完成 `CONTRIBUTING.md` 的
coordinator rollout gate 后，才按部署前记录恢复需要的 flag。

## 相关文件

- Override：`docker-compose.sandbox-off.yml`（DD-19：literal-pin mode + `depends_on`/`volumes` 均 `!override` + `sandbox-image` 去激活 profile）
- 值级验收（CI）：`api/tests/structure/test_sandbox_off_compose.py`
- 启动互斥校验：`api/app/main.py` `check_sandbox_off_flag_exclusion()`
- 设计依据：spec §10.3（迁移与灰度）、DD-16 / DD-19、INV-SPM-2 / INV-SPM-3
