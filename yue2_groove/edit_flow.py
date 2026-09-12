"""Edit-iteration helpers: freeze a baseline, check invariants, build manifests.

The workflow these functions support is symbolic-first:

1. **freeze** a generated work (a copy of its small text artifacts plus hashes;
   the original run directory is never touched and stays the listening baseline);
2. **edit** the ABC (and style/lyrics) in the UI;
3. **check** exact sounding-note / meter invariants with the vendored portable
   checker, explicitly permitting a tempo change when that is intended;
4. **generate** again from the edited score — the ABC is always submitted, so an
   edit can never silently degrade into a fresh plan;
5. **compare** baseline and new render with ``vendor/listen.py``.

Everything here is pure Python; the Gradio wiring lives in ``webui.py``.
"""
from __future__ import annotations

import hashlib
import json
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import adapter, library
from .vendor import abc_tools

BASELINE_DIR = "baselines"
#: text/sidecar files recorded in a baseline snapshot (audio/latents are not copied)
BASELINE_FILES = ("score.abc", "request.json", "plan.json", "result.json", "edit_manifest.json")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def snapshot_hashes(directory) -> dict[str, str]:
    """SHA-256 of the artifacts that identify a work (missing files are absent)."""
    directory = Path(directory)
    hashes = {}
    for name in BASELINE_FILES + ("audio.flac", "latent.npy", "semantic.npy"):
        digest = sha256_file(directory / name)
        if digest:
            hashes[name] = digest
    return hashes


def clean_abc(text: str) -> str:
    return textwrap.dedent(text or "").strip()


def validate_edited_abc(text: str) -> str:
    """Return the cleaned ABC or raise ``ValueError``; an empty edit is refused."""
    source = clean_abc(text)
    if not source:
        raise ValueError("The edited ABC is empty; generation would discard the edit and plan a "
                         "new score")
    abc_tools.parse_abc(source)
    return source


def check_invariants(before_abc: str, after_abc: str, *, voices: str = "both",
                     allow_tempo_change: bool = False) -> dict:
    """Exact sounding-note / meter comparison between the baseline and the edit."""
    before_text, after_text = clean_abc(before_abc), clean_abc(after_abc)
    if not before_text or not after_text:
        raise ValueError("Both the baseline ABC and the edited ABC are required for a check")
    names = abc_tools.VOICES if voices == "both" else (voices,)
    before = abc_tools.parse_abc(before_text)
    after = abc_tools.parse_abc(after_text)
    result = abc_tools.compare(before, after, names=names, allow_tempo_change=bool(allow_tempo_change))
    result["before_sha256"] = sha256_text(before_text)
    result["after_sha256"] = sha256_text(after_text)
    result["checked_at"] = datetime.now().isoformat(timespec="seconds")
    return result


def build_edit_request(style: str, lyrics: str, abc_text: str, *, cot: str = "full",
                       seed: int = 831001, cfg_scale: float | None = None, id: str | None = None,
                       request_factory: Callable | None = None):
    """YuE2 request for an edited score; ``abc`` is always present."""
    if cot not in ("full", "melody"):
        raise ValueError("An edited score requires cot=full (melody+harmony) or cot=melody")
    text = validate_edited_abc(abc_text)
    kwargs = {"style": style, "lyrics": lyrics, "cot": cot, "seed": int(seed), "abc": text}
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    if (id or "").strip():
        kwargs["id"] = id.strip()
    factory = request_factory or adapter.song_request
    return factory(**kwargs)


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find a free directory next to {path}")


def freeze_baseline(root, rel: str, *, baseline_root=None, now=None) -> dict:
    """Record an immutable reference to one work.

    Writes ``<baseline_root>/<stamp>-<name>-baseline/`` containing ``baseline.json``
    plus copies of the source's ``score.abc`` and ``request.json`` (small files only;
    audio and latents stay in the original run directory).  The source directory is
    never modified.  Returns the baseline record.
    """
    root = Path(root).resolve()
    source = (root / rel).resolve()
    if source == root or root not in source.parents or not source.is_dir():
        raise ValueError(f"Not a work directory under {root}: {rel}")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    name = library.display_name(source.name)[:60] or "work"
    baselines = Path(baseline_root).expanduser() if baseline_root else root / BASELINE_DIR
    destination = _unique_dir(baselines / f"{stamp}-{name}-baseline")
    destination.mkdir(parents=True, exist_ok=False)

    record = {
        "schema": "yue2-groove-baseline-v1",
        "created": datetime.fromtimestamp(now if now is not None else time.time()).isoformat(timespec="seconds"),
        "source": {"rel": rel, "path": str(source)},
        "baseline": {"rel": destination.relative_to(root).as_posix() if destination.is_relative_to(root)
                     else str(destination), "path": str(destination)},
        "hashes": snapshot_hashes(source),
    }
    (destination / "baseline.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for copied in ("score.abc", "request.json"):
        if (source / copied).is_file():
            (destination / copied).write_text(
                (source / copied).read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    record["abc"] = (source / "score.abc").read_text(encoding="utf-8") if (source / "score.abc").is_file() else ""
    return record


def build_edit_manifest(*, source_rel: str, before_abc: str, after_abc: str, cot: str,
                        seed: int, cfg_scale, invariants: dict | None, voices: str,
                        allow_tempo_change: bool, allow_changes: bool,
                        baseline: dict | None = None, now=None) -> dict:
    """The ``edit_manifest.json`` written next to a regenerated song."""
    before_text, after_text = clean_abc(before_abc), clean_abc(after_abc)
    return {
        "schema": "yue2-groove-edit-v1",
        "created": datetime.fromtimestamp(now if now is not None else time.time()).isoformat(timespec="seconds"),
        "source": {"rel": source_rel, "baseline": baseline or None},
        "abc": {"before_sha256": sha256_text(before_text) if before_text else None,
                "after_sha256": sha256_text(after_text)},
        "request": {"cot": cot, "seed": int(seed), "cfg_scale": cfg_scale},
        "invariants": invariants or None,
        "permitted": {"compared_voices": ["Vocal", "Ins"] if voices == "both" else [voices],
                      "tempo_change": bool(allow_tempo_change),
                      "changes_override": bool(allow_changes)},
    }
