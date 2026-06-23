# C2 Shell-Mode (S2) — shell-enabled patch-producing child

**Status**: behind a default-OFF master flag (`ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED`).
**Epic**: C2-full S2. Spec: `docs/superpowers/specs/2026-06-22-c2full-s2-shell-capable-task-mode-design.md`.

## What it is

A coordinator child work-unit may opt into **shell-mode** (`shell_mode=True`),
which unblocks the 5 raw shell tools (`shell_execute`, `shell_wait_process`,
`shell_kill_process`, `shell_write_input`, `shell_read_output`) inside that
child's **isolated, ephemeral** sandbox. Despite running raw shell, the child's
**sole product is still a parent-generated, lease-validated `PatchManifest`**
that flows through the existing all-or-nothing `PatchApplier`. There is **no
shared-workspace direct write** in v1.

How the diff is captured (instead of typed file-tool events, which are blind to
raw shell writes):

1. A **PRE snapshot** of the child workspace (`/home/ubuntu`) is taken right
   after seed-install, before the ReAct loop.
2. The child runs (possibly raw shell).
3. Shell sessions are **quiesced** (process-group kill + a double-scan stability
   check) so no background writer races the snapshot.
4. A **POST snapshot** is taken; `_extract_patch_files_from_snapshot` diffs
   PRE→POST by the `(kind, sha256, size, mode, link_target)` tuple and builds a
   `PatchManifest`.

Capture + lease-revalidation run in **trusted app-layer code** (the API
process), so the reducer/applier keep trusting manifest paths exactly as before.

## Tree leases are ADD-only (v1)

A `TreeLease{prefix, ops=frozenset({"add"})}` authorizes **creating new files**
under a directory prefix. It does **NOT** authorize modifying or deleting
existing files — those require an explicit, seeded **file** `PathLease`
(op=`modify`/`delete`), so the PRE-scan digest equals the parent base the
applier diffs against.

Consequences:
- A `modify`/`delete` of a file covered **only** by a tree lease → **group
  zero-apply** (the whole manifest is discarded; nothing is written).
- `write_tree_lease` implies `shell_mode=True`; a non-empty tree lease on a
  non-shell unit is rejected at validation/dispatch.
- Monorepo-wide `make`/formatter over un-enumerated existing files is **NOT
  supported** (trips the snapshot caps or zero-applies) — deferred to C5+.

## Fail-closed boundary (group zero-apply)

Child-finalize is the single authoritative integrity boundary. The **whole
manifest is discarded** (→ `NEEDS_AUTHORIZATION`, apply skipped) on ANY of:
out-of-lease path; op mismatch; tree-only `modify`/`delete`; special file
(fifo/socket/block/char); symlink; `kind=="other"`; mode-only change; kind
change; bare / non-directory-qualified path; `scan.truncated`; over a snapshot
cap; an `add` whose **parent** path is an existing regular file
(`tree_add_target_exists`); or a parent path whose inode kind violates the
add⇒`missing` / modify·delete⇒`regular` invariant (`parent_not_regular`).

## Caps

`CoordinatorLimits` (env-overridable, each `>0`): `max_snapshot_paths`,
`max_snapshot_files`, `max_snapshot_total_bytes`, `max_snapshot_seconds`.
Caps are enforced **during** the walk (early-abort); a breach sets
`truncated=True` → fail-closed zero-apply.

## Two-level gating + fail-safe default

A unit runs shell-mode **iff `flag_on AND wu.shell_mode`** (both affirmative):
- Master env flag `ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED` (default OFF).
- Per-unit `shell_mode: bool = False` (default False).

Flag OFF ⇒ the planner is not taught the shell/tree schema, and `dispatch_node`
**actively fail-closes** any stale/hand-crafted `shell_mode=True` /
`write_tree_lease` payload: a unit that still carries a typed file lease is
**coerced to typed-only** (shell stripped, the typed write kept), while a unit
with NO typed lease (tree-only / shell-only) is **hard-rejected** at dispatch.
Either way no shell path activates — behavior is byte-for-byte the pre-S2 typed
coordinator path.

## Security framing (honest v1)

v1 protects **parent-APPLY integrity, NOT child runtime confinement.** A
shell child can trash its own ephemeral sandbox, write anywhere inside it,
create special files, and use sudo (the sandbox container runs root with
NOPASSWD sudo — see `sandbox/Dockerfile` and `docker_sandbox.py` container
posture). The enforcement is entirely post-hoc: the fresh per-child sandbox +
whole-writable-workspace scan + group zero-apply + the destroyed-after-run
sandbox (M1) mean **nothing out-of-lease ever reaches the parent**. We claim
apply integrity, not runtime confinement. **Malicious-child confinement is
deferred to C5.**
