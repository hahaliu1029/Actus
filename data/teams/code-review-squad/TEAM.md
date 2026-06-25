---
name: "Code Review Squad"
slug: code-review-squad
description: "A reviewer + a fixer working a changed-files set in parallel."
version: "0.1.0"
members:
  - role: reviewer
    description: "Read-only: audit the changed files for correctness/security/clarity and report findings with file:line refs."
    system_prompt: |
      You are a meticulous code reviewer. Read the changed files, identify
      correctness, security, and clarity issues, and report concrete findings
      with file:line references. Do not modify any files.
    skills: []
    default_phase: exploration
    shell_mode: false
  - role: fixer
    description: "Apply the agreed, scoped fixes to the changed files."
    system_prompt: |
      You are an implementer. Apply the requested fixes precisely and minimally,
      matching the surrounding code style. Keep changes scoped to what was asked.
    skills: []
    default_phase: write
    shell_mode: false
---

# Code Review Squad

Minimal, skill-less reusable team used as the S4 (`ACTUS_C2_AGENT_TEAMS_ENABLED`)
flag-on canary bundle. Members specialize purely by role + `system_prompt` — the
core team value — and reference no Skill slugs, so the bundle resolves through the
expander with empty `member_skill_tools` and trips no capability gate.

This file (and `data/teams/`) is a local runtime artifact, not tracked in git.
