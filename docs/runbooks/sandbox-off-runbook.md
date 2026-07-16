# Sandbox off 档部署 Runbook

> **档位语义**：`off` 禁止任何新容器创建、app 路径零 lifecycle mutation（spec §5.1 / INV-SPM-3）。
> 沙箱工具与 skill-creation 工具不注册、VNC/接管/skill-create 端点拒绝。存量遗留容器由
> **pre-switch drain + TTL** 回收（off 无 docker.sock，无法自清）。

## ⚠️ 顺序警告：先 drain 再切

**必须先在仍有 socket 的 always/on_demand 状态下 drain 遗留容器，再切 off override。**
off 容器起来后没有 `/var/run/docker.sock`，无法清理任何存量沙箱——顺序颠倒会留下永久孤儿
（只能靠 TTL 慢慢自灭）。

## ⚠️ 预检：关闭 coordinator 三 flag（off × 任一 C2 flag=true → 启动 fail-fast）

`off` 模式不供给任何父沙箱，与需要沙箱的 C2 coordinator 分派路径互斥。三个 flag（
`ACTUS_C2_COORDINATOR_ENABLED` / `ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED` /
`ACTUS_C2_AGENT_TEAMS_ENABLED`）任一为 `true` 时，API 启动期
`check_sandbox_off_flag_exclusion()`（`api/app/main.py`）会抛
`RuntimeError: SANDBOX_PROVISION_MODE=off is incompatible with coordinator flags: [...]`，
容器起不来。

**`.env.example` 默认 `ACTUS_C2_COORDINATOR_ENABLED=true`**（另两个默认 `false`），不关则 off
容器直接 fail-fast。下方命令 ②a 用**三键 upsert**（存在则改 `false`、缺失则追加）确保三行确定
为 `false`——replace-only 的 `sed` 补不出缺失键。

## Canonical 四命令（spec §10.3，逐字冻结，实现者与运维不自行拼装）

```bash
# ① pre-switch drain（在 repo 根、仍有 socket 时执行；label 优先 + 可配置前缀兜底——
#    SANDBOX_NAME_PREFIX 可自定义（compose:134），且 PR-1a 之前创建的容器无 label；
#    先加载 .env 使自定义前缀在本 shell 生效（R9 修正——drain 不经 compose，不会自动读 .env）
set -a; . ./.env 2>/dev/null; set +a
docker ps -aq --filter "label=actus.session_id" | xargs -r docker rm -f
docker ps -aq --filter "name=${SANDBOX_NAME_PREFIX:-actus-sb}-" | xargs -r docker rm -f
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
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config --services | grep -c sandbox-image   # 期望 0
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config | grep -A1 "SANDBOX_PROVISION_MODE"  # 断言值为 off
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env config | grep -c "docker.sock"              # 期望 0（socket 已移除）
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml --env-file .env logs api 2>&1 | grep -c "incompatible with coordinator flags"  # 期望 0（未 fail-fast）
# ④ 回退（还原 always）——**r20/codex R20-P2-T3 修正：必须显式 pin `SANDBOX_PROVISION_MODE=always`**。
#    base compose 读 `${SANDBOX_PROVISION_MODE:-always}`（plan Task 12），若 `.env` 已显式设 off/on_demand
#    （on_demand 部署即在 .env 设该值），仅去 override 后默认值**不生效**、"还原 always" 不发生。upsert 保证：
if grep -qE "^SANDBOX_PROVISION_MODE=" .env; then sed -i.bak -E "s/^SANDBOX_PROVISION_MODE=.*/SANDBOX_PROVISION_MODE=always/" .env; else printf 'SANDBOX_PROVISION_MODE=always\n' >> .env; fi
grep -E "^SANDBOX_PROVISION_MODE=" .env          # 期望 always
docker compose --env-file .env up -d --build     # 去 off override + .env pin=always → 真还原 always
```

## 回退（还原 always）

命令 ④ 即回退：**显式 upsert `SANDBOX_PROVISION_MODE=always` 到 `.env`**（不依赖 `:-always`
默认——`.env` 若含 `off`/`on_demand` 会盖默认）+ 去 off override 重启。回退后 base 单文件重新携带
`sandbox-image` service 与 `docker.sock` 挂载，always 供给面完整恢复。

## 相关文件

- Override：`docker-compose.sandbox-off.yml`（DD-19：literal-pin mode + `depends_on`/`volumes` 均 `!override` + `sandbox-image` 去激活 profile）
- 值级验收（CI）：`api/tests/structure/test_sandbox_off_compose.py`
- 启动互斥校验：`api/app/main.py` `check_sandbox_off_flag_exclusion()`
- 设计依据：spec §10.3（迁移与灰度）、DD-16 / DD-19、INV-SPM-2 / INV-SPM-3
