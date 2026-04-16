"""Skill 安全扫描引擎 — 12 类威胁静态检测"""

from __future__ import annotations

import hashlib
import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanFinding:
    pattern_id: str
    category: str
    severity: str       # critical / high / medium
    file: str
    line: int
    match: str


@dataclass(frozen=True)
class ScanReport:
    verdict: str        # safe / caution / dangerous
    findings: tuple[ScanFinding, ...]
    content_hash: str
    scanned_at: datetime

    @classmethod
    def from_findings(
        cls, findings: list[ScanFinding], content_hash: str
    ) -> ScanReport:
        if any(f.severity == "critical" for f in findings):
            verdict = "dangerous"
        elif any(f.severity in ("high", "medium") for f in findings):
            verdict = "caution"
        else:
            verdict = "safe"
        return cls(
            verdict=verdict,
            findings=tuple(findings),
            content_hash=content_hash,
            scanned_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "findings": [
                {
                    "pattern_id": f.pattern_id,
                    "category": f.category,
                    "severity": f.severity,
                    "file": f.file,
                    "line": f.line,
                    "match": f.match,
                }
                for f in self.findings
            ],
            "content_hash": self.content_hash,
            "scanned_at": self.scanned_at.isoformat(),
        }


# ---------- 威胁模式定义 ----------

@dataclass(frozen=True)
class ThreatPattern:
    pattern_id: str
    category: str
    severity: str
    regex: re.Pattern[str]
    description: str


def _compile(pattern_id: str, category: str, severity: str,
             regex_str: str, desc: str) -> ThreatPattern:
    return ThreatPattern(
        pattern_id=pattern_id, category=category, severity=severity,
        regex=re.compile(regex_str, re.IGNORECASE),
        description=desc,
    )


THREAT_PATTERNS: list[ThreatPattern] = [
    # --- exfiltration (critical) ---
    _compile("exfil_curl_key", "exfiltration", "critical",
             r"curl\b.*\$[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)", "curl exfiltrating env vars"),
    _compile("exfil_dns", "exfiltration", "critical",
             r"dig\b.*\bTXT\b.*\$|nslookup\b.*\$", "DNS exfiltration"),
    _compile("exfil_md_image", "exfiltration", "critical",
             r"!\[.*?\]\(https?://(?!(?:github|githubusercontent|shields\.io))", "markdown image exfiltration"),

    # --- injection (critical) ---
    _compile("inject_ignore", "injection", "critical",
             r"ignore\s+(?:previous|all|above|prior)\s+instructions", "prompt injection: ignore instructions"),
    _compile("inject_role_hijack", "injection", "critical",
             r"(?:act|behave|pretend)\s+as\s+(?:an?\s+)?(?:admin|root|system)", "role hijacking"),
    _compile("inject_hidden_html", "injection", "critical",
             r"<!--\s*(?:ignore|override|system|inject)", "hidden HTML injection"),

    # --- destructive (critical) ---
    _compile("destruct_rm_rf", "destructive", "critical",
             r"rm\s+(?:-[a-z]*)?r(?:[a-z]*)?f\b", "recursive forced deletion"),
    _compile("destruct_mkfs", "destructive", "critical",
             r"\bmkfs\b", "filesystem format"),
    _compile("destruct_dd_zero", "destructive", "critical",
             r"\bdd\b.*if=/dev/(?:zero|urandom).*of=/dev/", "disk overwrite"),
    _compile("destruct_drop_table", "destructive", "critical",
             r"\bDROP\s+(?:TABLE|DATABASE)\b", "SQL DROP"),

    # --- persistence (critical) ---
    _compile("persist_crontab", "persistence", "critical",
             r"\bcrontab\b", "crontab modification"),
    _compile("persist_bashrc", "persistence", "critical",
             r">>?\s*~?/?\.\b(?:bashrc|zshrc|profile|bash_profile)\b", "shell RC injection"),
    _compile("persist_ssh_key", "persistence", "critical",
             r"authorized_keys", "SSH key injection"),
    _compile("persist_systemd", "persistence", "critical",
             r"/etc/systemd/system/", "systemd service persistence"),

    # --- network (critical) ---
    _compile("net_reverse_shell", "network", "critical",
             r"(?:bash|sh)\s+-i\s+>&\s+/dev/tcp/", "reverse shell"),
    _compile("net_nc_exec", "network", "critical",
             r"\bnc\b.*-[a-z]*e\s+/bin/", "netcat exec"),
    _compile("net_raw_socket", "network", "critical",
             r"socket\s*\(\s*socket\.AF_INET", "raw socket"),

    # --- obfuscation (high) ---
    _compile("obfusc_base64_eval", "obfuscation", "high",
             r"eval\s*\(\s*(?:base64\.b64decode|__import__)", "base64 eval"),
    _compile("obfusc_chr_build", "obfuscation", "high",
             r"chr\s*\(\s*\d+\s*\)\s*\+\s*chr", "chr() string building"),
    _compile("obfusc_exec_compile", "obfuscation", "high",
             r"exec\s*\(\s*compile\s*\(", "exec(compile())"),

    # --- execution (high for direct, medium for meta) ---
    _compile("exec_subprocess_popen", "execution", "high",
             r"subprocess\.(?:Popen|call|run|check_output|check_call)\s*\(", "subprocess execution"),
    _compile("exec_os_system", "execution", "high",
             r"os\.(?:system|popen|exec[a-z]*)\s*\(", "os.system execution"),
    _compile("exec_eval_exec", "execution", "high",
             r"(?<!\w)(?:eval|exec)\s*\((?!.*#\s*nosec)", "eval/exec call"),

    # --- traversal (medium) ---
    _compile("trav_dotdot", "traversal", "medium",
             r"\.\./\.\./", "path traversal"),
    _compile("trav_etc_passwd", "traversal", "medium",
             r"/etc/passwd", "sensitive file access"),
    _compile("trav_proc", "traversal", "medium",
             r"/proc/(?:self|[0-9]+)/", "proc filesystem access"),

    # --- mining (critical) ---
    _compile("mine_xmrig", "mining", "critical",
             r"\bxmrig\b", "xmrig miner"),
    _compile("mine_pool", "mining", "critical",
             r"(?:stratum|pool)\+?(?:tcp|ssl)://", "mining pool connection"),

    # --- supply_chain (high) ---
    _compile("supply_pipe_shell", "supply_chain", "high",
             r"curl\b.*\|\s*(?:bash|sh|zsh)\b", "pipe to shell"),
    _compile("supply_pip_no_pin", "supply_chain", "high",
             r"pip\s+install\s+(?!.*==)[a-z]", "unpinned pip install"),

    # --- privilege_escalation (high) ---
    _compile("privesc_nopasswd", "privilege_escalation", "high",
             r"NOPASSWD", "sudo NOPASSWD"),
    _compile("privesc_setuid", "privilege_escalation", "high",
             r"chmod\s+[ug]\+s\b", "setuid/setgid"),

    # --- credential_exposure (critical) ---
    _compile("cred_gh_token", "credential_exposure", "critical",
             r"\bghp_[a-zA-Z0-9]{36}\b", "GitHub personal access token"),
    _compile("cred_openai", "credential_exposure", "critical",
             r"\bsk-(?:proj-)?[a-zA-Z0-9]{10,}", "OpenAI API key"),
    _compile("cred_aws", "credential_exposure", "critical",
             r"\bAKIA[A-Z0-9]{16}\b", "AWS access key"),
    _compile("cred_pem", "credential_exposure", "critical",
             r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "private key PEM block"),
]

# import statements are legitimate code
IMPORT_ALLOWLIST = re.compile(r"^\s*(?:import|from)\s+", re.IGNORECASE)

# a2a limited scan categories
A2A_CATEGORIES = frozenset({"injection", "credential_exposure"})

# invisible Unicode characters (16 zero-width/direction control chars)
INVISIBLE_CHARS = frozenset({
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',  # zero-width + LRM/RLM
    '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',  # word joiner + invisible math
    '\ufeff',                                           # BOM / zero-width no-break
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',  # bidi control
})

# binary file extensions
BINARY_EXTENSIONS = frozenset({
    '.exe', '.dll', '.so', '.bin', '.o', '.msi',
    '.bat', '.cmd', '.com', '.elf', '.a',
})

# structural limits
MAX_FILES = 50
MAX_TOTAL_BYTES = 1_048_576       # 1 MB
MAX_SINGLE_FILE_BYTES = 262_144   # 256 KB
MAX_LINE_LENGTH = 10_000          # ReDoS protection


class SkillsGuard:
    """Skill bundle static security scanner"""

    def __init__(self, patterns: list[ThreatPattern] | None = None) -> None:
        self._patterns = patterns or THREAT_PATTERNS

    def _scan_text(
        self,
        text: str,
        file_path: str,
        line_num: int,
        categories: frozenset[str] | None = None,
    ) -> list[ScanFinding]:
        """Scan a single line of text, return all matching findings"""
        findings: list[ScanFinding] = []

        # Skip import statements (legitimate code)
        if IMPORT_ALLOWLIST.match(text):
            return findings

        for pattern in self._patterns:
            if categories and pattern.category not in categories:
                continue
            if pattern.regex.search(text):
                findings.append(ScanFinding(
                    pattern_id=pattern.pattern_id,
                    category=pattern.category,
                    severity=pattern.severity,
                    file=file_path,
                    line=line_num,
                    match=text[:200],
                ))
        return findings

    def _scan_file(
        self,
        file_path: Path,
        relative_path: str,
        categories: frozenset[str] | None = None,
        timeout_seconds: float = 5.0,
    ) -> list[ScanFinding]:
        """Scan a single file (with per-file timeout, fail-closed)"""
        import signal
        import threading

        # signal.SIGALRM only works on Unix main thread; degrade gracefully otherwise
        if threading.current_thread() is not threading.main_thread():
            return self._scan_file_inner(file_path, relative_path, categories)

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"scan timeout for {relative_path}")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_seconds))
        try:
            return self._scan_file_inner(file_path, relative_path, categories)
        except TimeoutError:
            return [ScanFinding(
                pattern_id="structural_scan_timeout",
                category="structural",
                severity="critical",
                file=relative_path,
                line=0,
                match=f"file scan timed out after {timeout_seconds}s",
            )]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _scan_file_inner(
        self,
        file_path: Path,
        relative_path: str,
        categories: frozenset[str] | None = None,
    ) -> list[ScanFinding]:
        """Internal scan implementation (no timeout wrapper)"""
        findings: list[ScanFinding] = []

        # Structural check: oversized file
        file_size = file_path.stat().st_size
        if file_size > MAX_SINGLE_FILE_BYTES:
            findings.append(ScanFinding(
                pattern_id="structural_oversized",
                category="structural",
                severity="medium",
                file=relative_path,
                line=0,
                match=f"file size {file_size} > {MAX_SINGLE_FILE_BYTES}",
            ))

        # Try reading as UTF-8
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(ScanFinding(
                pattern_id="structural_non_utf8",
                category="structural",
                severity="medium",
                file=relative_path,
                line=0,
                match="file contains non-UTF-8 bytes",
            ))
            return findings

        # Invisible Unicode detection
        for i, char in enumerate(content):
            if char in INVISIBLE_CHARS:
                findings.append(ScanFinding(
                    pattern_id="structural_invisible_unicode",
                    category="structural",
                    severity="high",
                    file=relative_path,
                    line=content[:i].count('\n') + 1,
                    match=f"invisible unicode U+{ord(char):04X}",
                ))
                break  # report once per file

        # Line-by-line scan
        for line_num, line in enumerate(content.splitlines(), 1):
            if len(line) > MAX_LINE_LENGTH:
                findings.append(ScanFinding(
                    pattern_id="structural_long_line",
                    category="structural",
                    severity="medium",
                    file=relative_path,
                    line=line_num,
                    match=f"line length {len(line)} > {MAX_LINE_LENGTH}",
                ))
                continue  # skip regex on long lines (ReDoS protection)
            findings.extend(self._scan_text(line, relative_path, line_num, categories))

        return findings

    def _scan_manifest_entry_commands(
        self,
        manifest: dict,
        relative_path: str,
        categories: frozenset[str] | None = None,
    ) -> list[ScanFinding]:
        """Scan manifest entry.command / exec_dir / tool_name fields"""
        findings: list[ScanFinding] = []
        tools = manifest.get("tools", [])
        if isinstance(tools, list):
            for i, tool in enumerate(tools):
                entry = tool.get("entry", {}) if isinstance(tool, dict) else {}
                for field_name in ("command", "exec_dir", "tool_name"):
                    value = entry.get(field_name)
                    if isinstance(value, str) and value:
                        line_findings = self._scan_text(
                            value, f"{relative_path}:tools[{i}].entry.{field_name}", 0,
                            categories,
                        )
                        findings.extend(line_findings)
        return findings

    # Root-level bookkeeping files excluded from content_hash to avoid
    # self-referential hash (meta.json contains scan_report.content_hash).
    # Only excluded at skill root, NOT inside bundle/ subdirectories.
    _HASH_EXCLUDE_ROOT_RELPATHS = frozenset({"meta.json", "bundle_index.json"})

    @staticmethod
    def compute_content_hash(bundle_dir: Path) -> str:
        """Compute content_hash over the canonical scan surface only.

        Includes: manifest.json, SKILL.md, bundle/**/*.py|.sh|.md|.json|.txt
        Excludes: ROOT/meta.json, ROOT/bundle_index.json (repo bookkeeping).
        Does NOT exclude bundle/meta.json etc. — those are real bundle content.
        """
        target_suffixes = {'.py', '.sh', '.bash', '.md', '.json', '.txt'}
        files = sorted(
            f for f in bundle_dir.rglob('*')
            if f.is_file()
            and f.suffix.lower() in target_suffixes
            and str(f.relative_to(bundle_dir)) not in SkillsGuard._HASH_EXCLUDE_ROOT_RELPATHS
        )
        h = hashlib.sha256()
        for f in files:
            rel = str(f.relative_to(bundle_dir))
            try:
                content = f.read_bytes()
            except OSError:
                content = b""
            h.update(rel.encode() + b"\0" + content)
        return f"sha256:{h.hexdigest()}"

    def scan(self, bundle_dir: Path) -> ScanReport:
        """Full scan (native / mcp) — all 12 categories"""
        return self._do_scan(bundle_dir, categories=None)

    def scan_limited(self, bundle_dir: Path) -> ScanReport:
        """Limited scan (a2a) — injection + credential_exposure only"""
        return self._do_scan(bundle_dir, categories=A2A_CATEGORIES)

    def _do_scan(
        self,
        bundle_dir: Path,
        categories: frozenset[str] | None,
    ) -> ScanReport:
        """Internal scan implementation, fail-closed"""
        try:
            return self._do_scan_inner(bundle_dir, categories)
        except Exception:
            logger.exception("SkillsGuard scan failed (fail-closed)")
            return ScanReport(
                verdict="dangerous",
                findings=(ScanFinding(
                    pattern_id="structural_scan_exception",
                    category="structural",
                    severity="critical",
                    file="<scanner>",
                    line=0,
                    match="scan engine exception (fail-closed)",
                ),),
                content_hash=self.compute_content_hash(bundle_dir)
                    if bundle_dir.exists() else "sha256:error",
                scanned_at=datetime.now(timezone.utc),
            )

    def _do_scan_inner(
        self,
        bundle_dir: Path,
        categories: frozenset[str] | None,
    ) -> ScanReport:
        if not bundle_dir.exists():
            raise FileNotFoundError(f"bundle_dir does not exist: {bundle_dir}")
        findings: list[ScanFinding] = []
        all_files = list(bundle_dir.rglob('*'))
        file_count = sum(1 for f in all_files if f.is_file())
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())

        # Structural checks
        if file_count > MAX_FILES:
            findings.append(ScanFinding(
                "structural_too_many_files", "structural", "medium",
                str(bundle_dir), 0, f"file count {file_count} > {MAX_FILES}",
            ))
        if total_size > MAX_TOTAL_BYTES:
            findings.append(ScanFinding(
                "structural_too_large", "structural", "medium",
                str(bundle_dir), 0, f"total size {total_size} > {MAX_TOTAL_BYTES}",
            ))

        # Binary file check
        for f in all_files:
            if f.is_file() and f.suffix.lower() in BINARY_EXTENSIONS:
                findings.append(ScanFinding(
                    "structural_binary", "structural", "critical",
                    str(f.relative_to(bundle_dir)), 0,
                    f"binary file: {f.name}",
                ))

        # File-level scan
        scan_map: dict[str, frozenset[str] | None] = {
            '.py': categories,
            '.sh': categories,
            '.bash': categories,
        }

        for f in all_files:
            if not f.is_file():
                continue
            rel = str(f.relative_to(bundle_dir))
            suffix = f.suffix.lower()

            if suffix in scan_map:
                findings.extend(self._scan_file(f, rel, scan_map[suffix]))
            elif f.name == 'manifest.json':
                findings.extend(self._scan_file(
                    f, rel,
                    frozenset({"credential_exposure"}) if categories else None,
                ))
                try:
                    manifest = json.loads(f.read_text(encoding="utf-8"))
                    findings.extend(self._scan_manifest_entry_commands(manifest, rel, categories))
                except (json.JSONDecodeError, OSError):
                    pass
            elif f.name.upper() == 'SKILL.MD':
                skill_cats = (
                    categories if categories
                    else frozenset({"injection", "exfiltration"})
                )
                findings.extend(self._scan_file(f, rel, skill_cats))
            elif f.name == 'requirements.txt':
                findings.extend(self._scan_file(
                    f, rel, frozenset({"supply_chain"}) if not categories else categories,
                ))

        content_hash = self.compute_content_hash(bundle_dir)
        return ScanReport.from_findings(findings, content_hash)
