## 描述

<!-- 简要描述此 PR 的变更内容 -->

## 变更类型

- [ ] Bug 修复 (fix)
- [ ] 新功能 (feat)
- [ ] 文档更新 (docs)
- [ ] 代码重构 (refactor)
- [ ] 测试 (test)
- [ ] 构建/依赖 (chore)

## 关联 Issue

<!-- 如果有相关 Issue，请链接 -->
Closes #

## 变更内容

<!-- 详细描述你做了哪些更改 -->

-
-

## 测试

<!-- 描述你如何测试了这些变更 -->

- [ ] 已通过现有测试
- [ ] 已添加新测试覆盖变更
- [ ] 已本地验证功能正常

## 检查清单

- [ ] 代码遵循项目编码规范
- [ ] 已自我 review 代码
- [ ] 已更新相关文档（如需要）
- [ ] 变更不引入新的警告或错误

### CS3 ToolOutcome variant 扩展 checklist

若本 PR 引入新 `ToolOutcome` variant（e.g. `PartialSuccess`），下列条目**全部**必须勾：

- [ ] 已补 `_function_result_from_outcome` 里新 variant 的 `isinstance` 分支
- [ ] 已补 `_project_unknown_variant_fallback` 的 regression fixture（防止降级链路漂移）
- [ ] 已确保 `test_projector_covers_all_outcome_variants.py` 不 regress
- [ ] 已在 `api/config.yaml.example` 的 `tool_runtime.enabled_outcome_variants` 加默认值
- [ ] 已和运维确认 deploy 顺序：**先 deploy 带新 projector 的 backend → config rollout → 再启用 wrapper**（参见 `docs/adr/CS3-tool-event-envelope-v1.md`）
- [ ] 已跑 codex Round 3+ 对新 variant projector 实现做独立审阅
