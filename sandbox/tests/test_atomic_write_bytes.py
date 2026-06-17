"""S1 `_atomic_write_bytes` helper — the core atomic-overwrite recipe.

Covers spec §5 sandbox tests 1-3, 4-8, 9-10, 15-16, 18 (consumer-routed tests
17/11/12/13 live in the per-consumer test files).
"""
import os
import stat

import pytest


# --- happy path + crash safety (spec tests 1, 2, 3, 10) -------------------

def test_atomic_overwrite_replaces_content(tmp_path):
    from app.services import file as filemod

    target = tmp_path / "out.txt"
    target.write_bytes(b"OLD-LONGER-CONTENT")
    n = filemod._atomic_write_bytes(str(target), [b"NEW"])
    assert n == 3
    assert target.read_bytes() == b"NEW"


def test_no_truncation_on_midwrite_error(tmp_path):
    """A failure WHILE writing leaves the original target byte-for-byte
    intact and removes the temp (no `.actus-tmp-*` orphan)."""
    from app.services import file as filemod

    target = tmp_path / "out.txt"
    target.write_bytes(b"ORIGINAL")

    def bad_chunks():
        yield b"PARTIAL"
        raise RuntimeError("boom mid-write")

    with pytest.raises(RuntimeError, match="boom mid-write"):
        filemod._atomic_write_bytes(str(target), bad_chunks())

    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir()
                 if p.name.startswith(".actus-tmp-")]
    assert leftovers == []


def test_temp_created_in_target_dir_no_exdev(tmp_path, monkeypatch):
    """mkstemp must use dir=dirname(target) so os.replace is intra-FS."""
    from app.services import file as filemod

    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "out.txt"
    captured = {}
    real_mkstemp = filemod.tempfile.mkstemp

    def spy(*args, **kwargs):
        captured["dir"] = kwargs.get("dir")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(filemod.tempfile, "mkstemp", spy)
    filemod._atomic_write_bytes(str(target), [b"data"])
    assert captured["dir"] == os.path.dirname(str(target))


def test_nested_missing_parent_created(tmp_path):
    from app.services import file as filemod

    target = tmp_path / "a" / "b" / "c.txt"
    n = filemod._atomic_write_bytes(str(target), [b"deep"])
    assert n == 4
    assert target.read_bytes() == b"deep"


def test_bare_path_rejected(tmp_path, monkeypatch):
    """A path with no directory component (`os.makedirs("")`) raises,
    exactly as today — S1 does NOT widen bare-path acceptance."""
    from app.services import file as filemod

    monkeypatch.chdir(tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        filemod._atomic_write_bytes("bare.txt", [b"x"])


def test_atomic_write_uses_write_all_short_write_loop(tmp_path, monkeypatch):
    """End-to-end constraint that `_atomic_write_bytes` writes via the
    `_write_all` short-write LOOP, not a single `os.write`. A mutant that
    replaced `n += _write_all(fd, chunk)` with `os.write(fd, chunk)` would drop
    bytes on a short write and fail this (Task 2's unit test alone wouldn't
    catch the helper bypassing `_write_all`)."""
    from app.services import file as filemod

    target = tmp_path / "sw.txt"
    real_write = filemod.os.write
    state = {"first": True}

    def short_write(fd, data):
        if state["first"]:
            state["first"] = False
            return real_write(fd, data[:2])  # short first write
        return real_write(fd, data)

    monkeypatch.setattr(filemod.os, "write", short_write)
    n = filemod._atomic_write_bytes(str(target), [b"abcdef"])
    assert n == 6
    assert target.read_bytes() == b"abcdef"


# --- mode semantics (spec tests 4, 5, 6) ----------------------------------

def test_mode_preserved_on_overwrite(tmp_path):
    from app.services import file as filemod

    target = tmp_path / "m.txt"
    target.write_bytes(b"old")
    os.chmod(target, 0o640)
    filemod._atomic_write_bytes(str(target), [b"new"])
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o640


def test_new_file_mode_matches_umask(tmp_path):
    from app.services import file as filemod

    # Independent oracle: a file created the OLD way (plain open("wb")) gets the
    # OS-umask mode (0o666 & ~umask) — the parity target — WITHOUT trusting the
    # impl's own `_UMASK` constant (so a bug in `_read_umask_once` is caught).
    baseline = tmp_path / "baseline.txt"
    baseline.write_bytes(b"")  # write_bytes -> open(..., "wb") -> umask-applied
    expected_mode = stat.S_IMODE(os.lstat(baseline).st_mode)

    target = tmp_path / "n.txt"
    filemod._atomic_write_bytes(str(target), [b"x"])
    # New helper file gets the same mode as old open(...,"wb") (NOT mkstemp 0o600).
    assert stat.S_IMODE(os.lstat(target).st_mode) == expected_mode


def test_setuid_bit_stripped_on_overwrite(tmp_path):
    """Copying a 04xxx bit onto a root-owned file would mint setuid-root;
    S1 strips setuid/setgid/sticky."""
    from app.services import file as filemod

    target = tmp_path / "s.txt"
    target.write_bytes(b"old")
    os.chmod(target, 0o4755)
    filemod._atomic_write_bytes(str(target), [b"new"])
    st = os.lstat(target)
    assert not (st.st_mode & stat.S_ISUID)
    assert stat.S_IMODE(st.st_mode) == 0o755


# --- symlink semantics (spec tests 7, 8) ----------------------------------

def test_final_symlink_replaced_not_followed(tmp_path):
    """D9: a final-component symlink is REPLACED in place; the old referent
    is untouched (write is not redirected through the link)."""
    from app.services import file as filemod

    real = tmp_path / "real.txt"
    real.write_bytes(b"REAL-REFERENT")
    link = tmp_path / "link.txt"
    link.symlink_to(real)

    filemod._atomic_write_bytes(str(link), [b"NEW"])

    assert not os.path.islink(link)
    assert link.read_bytes() == b"NEW"
    assert real.read_bytes() == b"REAL-REFERENT"  # referent untouched


def test_intermediate_dir_symlink_resolves(tmp_path):
    """An intermediate directory symlink is resolved by the kernel; the file
    lands in the real directory."""
    from app.services import file as filemod

    realdir = tmp_path / "realdir"
    realdir.mkdir()
    dirlink = tmp_path / "dirlink"
    dirlink.symlink_to(realdir, target_is_directory=True)

    filemod._atomic_write_bytes(str(dirlink / "f.txt"), [b"X"])
    assert (realdir / "f.txt").read_bytes() == b"X"


# --- fsync / replace-failure cleanup (spec tests 15, 16) ------------------

def test_fsync_called_before_replace(tmp_path, monkeypatch):
    from app.services import file as filemod

    target = tmp_path / "out.txt"
    order = []
    real_fsync = filemod.os.fsync
    real_replace = filemod.os.replace
    monkeypatch.setattr(filemod.os, "fsync",
                        lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(filemod.os, "replace",
                        lambda s, d: (order.append("replace"), real_replace(s, d))[1])
    filemod._atomic_write_bytes(str(target), [b"x"])
    assert order == ["fsync", "replace"]


def test_replace_failure_cleans_temp_and_preserves_original(tmp_path, monkeypatch):
    from app.services import file as filemod

    target = tmp_path / "out.txt"
    target.write_bytes(b"ORIGINAL")

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(filemod.os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        filemod._atomic_write_bytes(str(target), [b"NEW"])

    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir()
                 if p.name.startswith(".actus-tmp-")]
    assert leftovers == []


# --- special-file fallback (spec test 18 — D12; MUST be non-blocking) ------

def test_special_file_falls_back_to_write_through(tmp_path, monkeypatch):
    """D12: an EXISTING FIFO/device is NOT replaced (that would clobber the
    node); the helper routes to `_direct_write_through`. The real
    write-through is STUBBED — a writer-only `open(fifo,"wb")` with no reader
    blocks and would hang CI."""
    from app.services import file as filemod

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    replace_calls = []
    monkeypatch.setattr(filemod.os, "replace",
                        lambda s, d: replace_calls.append((s, d)))
    wt_calls = []

    def spy_wt(target, source_chunks):
        # Record the PAYLOAD (not just target) so a content-dropping mutant is
        # caught; mimic the real byte count WITHOUT the blocking writer-open.
        chunks = list(source_chunks)
        wt_calls.append((target, chunks))
        return sum(len(c) for c in chunks)

    monkeypatch.setattr(filemod, "_direct_write_through", spy_wt)

    n = filemod._atomic_write_bytes(str(fifo), [b"data"])

    assert replace_calls == []                    # atomic replace NOT used on a FIFO
    assert wt_calls == [(str(fifo), [b"data"])]   # routed to fallback WITH the payload
    assert n == 4                                 # helper returns the write-through byte count
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)  # node still a FIFO


# --- os.write raises INSIDE _write_all (codex R1 P3#3) ---------------------

def test_oswrite_raises_midchunk_preserves_original_no_orphan(tmp_path, monkeypatch):
    """[codex R1 P3#3] If ``os.write`` itself raises mid-chunk (e.g. ENOSPC)
    AFTER a partial write into the temp — i.e. the failure happens INSIDE
    ``_write_all``'s loop, not merely between source chunks — the original
    target is left byte-for-byte intact and the temp is cleaned up (no
    ``.actus-tmp-*`` orphan). Distinct from
    ``test_no_truncation_on_midwrite_error`` (which raises between chunks)."""
    from app.services import file as filemod

    target = tmp_path / "out.txt"
    target.write_bytes(b"ORIGINAL")
    real_write = filemod.os.write
    state = {"n": 0}

    def failing_write(fd, data):
        state["n"] += 1
        if state["n"] == 1:
            real_write(fd, data[:2])  # partial bytes land in the temp...
            raise OSError("ENOSPC mid-chunk")  # ...then os.write fails
        return real_write(fd, data)

    monkeypatch.setattr(filemod.os, "write", failing_write)
    with pytest.raises(OSError, match="ENOSPC mid-chunk"):
        filemod._atomic_write_bytes(str(target), [b"abcdef"])

    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir()
                 if p.name.startswith(".actus-tmp-")]
    assert leftovers == []


# --- lstat-vs-stat: symlink mode + D12 symlink-to-special (codex R4 P3) ----

def test_symlink_overwrite_uses_new_file_mode_not_referent_mode(tmp_path):
    """[codex R4 P3a] A final-component symlink is replaced in place (D9) and
    the NEW regular file gets ``_NEW_FILE_MODE``, NOT the referent's mode —
    ``_apply_target_mode`` uses ``lstat`` so the symlink hits the else branch.
    Pins the lstat-vs-stat distinction in MODE handling: a ``stat`` mutant
    would follow the link and copy the referent's mode onto the new file."""
    from app.services import file as filemod

    real = tmp_path / "real.txt"
    real.write_bytes(b"REF")
    # A referent mode guaranteed distinct from _NEW_FILE_MODE so the assert is
    # non-vacuous regardless of the process umask.
    referent_mode = 0o600 if filemod._NEW_FILE_MODE != 0o600 else 0o640
    os.chmod(real, referent_mode)
    link = tmp_path / "link.txt"
    link.symlink_to(real)

    filemod._atomic_write_bytes(str(link), [b"NEW"])

    assert not os.path.islink(link)            # symlink replaced in place (D9)
    assert link.read_bytes() == b"NEW"
    new_mode = stat.S_IMODE(os.lstat(link).st_mode)
    assert new_mode == filemod._NEW_FILE_MODE  # new-file mode, NOT referent
    assert new_mode != referent_mode           # a `stat` mutant would copy this
    assert stat.S_IMODE(os.lstat(real).st_mode) == referent_mode  # referent untouched


def test_symlink_to_special_replaced_not_followed_to_writethrough(tmp_path, monkeypatch):
    """[codex R4 P3b] The D12 dispatch uses ``lstat``: a symlink whose REFERENT
    is a special file (FIFO) is classified as a SYMLINK (``S_ISLNK``) and
    replaced in place per D9 — it is NOT followed to the FIFO and routed to
    ``_direct_write_through``. Pins the lstat-vs-stat distinction in the D12
    gate: a ``stat`` mutant would follow the link to the FIFO, route to
    write-through (which would clobber/block on the node). ``_direct_write_through``
    is stubbed so a mutant cannot block CI on a reader-less FIFO open."""
    from app.services import file as filemod

    fifo = tmp_path / "real_pipe"
    os.mkfifo(fifo)
    link = tmp_path / "link_to_pipe"
    link.symlink_to(fifo)

    wt_calls = []

    def spy_wt(target, source_chunks):
        wt_calls.append((target, list(source_chunks)))
        return 0

    monkeypatch.setattr(filemod, "_direct_write_through", spy_wt)

    filemod._atomic_write_bytes(str(link), [b"NEW"])

    assert wt_calls == []                          # symlink NOT followed to the FIFO
    assert not os.path.islink(link)                # symlink replaced in place (D9)
    assert link.read_bytes() == b"NEW"
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)   # referent FIFO untouched


# --- Task 2: refuse_special strict refuse branch (2b) ---------------------

def test_refuse_special_raises_einval_on_fifo_no_blocking_open(tmp_path, monkeypatch):
    """refuse_special=True over a FIFO → OSError(EINVAL), and the blocking
    write-through is NEVER entered (spy, non-blocking — never perform the
    real writer-only open)."""
    import errno
    from app.services import file as filemod

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    wt_calls = []
    monkeypatch.setattr(
        filemod, "_direct_write_through",
        lambda target, chunks: wt_calls.append((target, list(chunks))),
    )

    with pytest.raises(OSError) as ei:
        filemod._atomic_write_bytes(str(fifo), [b"data"], refuse_special=True)
    assert ei.value.errno == errno.EINVAL
    assert "refusing to write special file" in str(ei.value)
    assert wt_calls == []  # never fell through to the blocking write-through
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)  # node untouched


def test_refuse_special_directory_not_special_falls_through(tmp_path, monkeypatch):
    """A directory is NOT in the special set — refuse_special=True must NOT
    raise EINVAL; it falls to _direct_write_through→open()→IsADirectoryError.
    Pins _is_special = fifo/socket/block/char only."""
    from app.services import file as filemod

    d = tmp_path / "adir"
    d.mkdir()
    # Real _direct_write_through on a directory → open() raises IsADirectoryError,
    # NOT the special-refuse EINVAL. Assert we reached that branch.
    with pytest.raises(IsADirectoryError):
        filemod._atomic_write_bytes(str(d), [b"x"], refuse_special=True)


def test_refuse_special_true_on_regular_still_atomic(tmp_path):
    """refuse_special only triggers on a direct special inode — a regular
    target still atomic-writes."""
    from app.services import file as filemod

    p = tmp_path / "f.txt"
    p.write_text("old")
    n = filemod._atomic_write_bytes(str(p), [b"NEW"], refuse_special=True)
    assert n == 3
    assert p.read_bytes() == b"NEW"


def test_refuse_special_true_on_missing_still_creates(tmp_path):
    from app.services import file as filemod

    p = tmp_path / "new.txt"
    n = filemod._atomic_write_bytes(str(p), [b"hi"], refuse_special=True)
    assert n == 2
    assert p.read_bytes() == b"hi"


def test_is_special_covers_exactly_fifo_socket_block_char():
    """`_is_special` returns True for the FULL special set (fifo/socket/block/
    char) and False for every non-special kind. A mutation that drops
    socket/block/char (or adds directory/regular) goes RED. Synthetic st_mode
    bits — no root/mknod needed."""
    import stat
    from app.services import file as filemod

    for special in (stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFBLK, stat.S_IFCHR):
        assert filemod._is_special(special | 0o644) is True
    for non_special in (stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK):
        assert filemod._is_special(non_special | 0o644) is False
