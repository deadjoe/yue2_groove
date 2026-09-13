"""Library backend for YUE2 // GROOVE.

This module is deliberately self-contained: it reads the artifact directories
written by the pipeline **by file convention** (``request.json``,
``result.json``, ``config.json``, ``score.abc``, ``audio.flac``, ``plan.json``,
``decode.json``) and never imports from the ``yue2`` package.  Upstream
refactors therefore cannot break the Library; if an artifact name ever changes,
the affected field simply renders as "—".

Everything here is plain Python returning strings/structures.  The Gradio
wiring lives in ``webui.py``; the CSS and the (small) player/score JS are
kept here so the whole feature ships as one file.
"""
from __future__ import annotations

import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# a directory that holds any of these is a Library entry
ARTIFACTS = ("result.json", "audio.flac", "score.abc")
# freeze records keep a copy of score.abc; they are provenance, not works
SKIP_DIRS = ("baselines",)
STAMP_RE = re.compile(r"^(\d{8})-(\d{6})-(.*)$")
SORTS = ("time_desc", "time_asc", "name_asc", "name_desc")
MAX_NAME = 80


# ─────────────────────────────────────────────────────────────── helpers ──
def format_seconds(value) -> str:
    try:
        total = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return "—"
    return f"{total // 60}:{total % 60:02d}"


def format_bytes(value) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "—"


def display_name(folder: str) -> str:
    match = STAMP_RE.match(folder)
    return match.group(3) if match else folder


def _created(path: Path) -> datetime:
    match = STAMP_RE.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _request_of(path: Path):
    """request.json, else the request embedded in plan.json (PLAN ONLY runs)."""
    request = _read_json(path / "request.json")
    if request is None:
        plan = _read_json(path / "plan.json")
        if isinstance(plan, dict) and isinstance(plan.get("request"), dict):
            request = plan["request"]
    return request


# ──────────────────────────────────────────────────────────────── scan ──
def _kind(result: dict, has_audio: bool, has_score: bool) -> str:
    if result and (result.get("weights") or has_audio):
        return "song"
    if result and result.get("task"):          # SheetSage2 transcription result.json
        return "transcription"
    if result:
        return "song"
    if has_audio:
        return "decode"
    if has_score:
        return "plan"
    return "data"


def _item(root: Path, path: Path) -> dict:
    request = _request_of(path) or {}
    result = _read_json(path / "result.json") or {}
    has_audio = (path / "audio.flac").is_file()
    has_score = (path / "score.abc").is_file()
    kind = _kind(result, has_audio, has_score)
    truncated = result.get("truncated")
    if isinstance(truncated, dict):
        truncated = any(bool(v) for v in truncated.values())
    return {
        "rel": path.relative_to(root).as_posix(),
        "path": str(path),
        "folder": path.name,
        "name": display_name(path.name),
        "created": _created(path),
        "created_label": _created(path).strftime("%Y-%m-%d %H:%M"),
        "duration": result.get("audio_seconds"),
        "status": result.get("status", ""),
        "truncated": bool(truncated),
        "cot": request.get("cot", ""),
        "seed": request.get("seed"),
        "cfg": request.get("cfg_scale"),
        "kind": kind,
        "has_audio": has_audio,
        "has_score": has_score or bool(request.get("abc")),
        "size": _dir_bytes(path),
    }


def scan(root) -> list[dict]:
    """Every artifact directory under *root* (one level of nesting for batches)."""
    root = Path(root).resolve()
    items: list[dict] = []
    if not root.is_dir():
        return items

    def visit(directory: Path, depth: int) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for path in entries:
            if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIRS:
                continue
            if (path / "index.html").is_file() and (path / "manifest.json").is_file():
                continue  # a listening comparison bundle, not a work
            if any((path / name).exists() for name in ARTIFACTS):
                items.append(_item(root, path))
                continue  # never descend into a work directory
            if depth < 2:
                visit(path, depth + 1)

    visit(root, 1)
    return items


def sort_items(items, mode: str = "time_desc") -> list[dict]:
    if mode == "time_asc":
        return sorted(items, key=lambda i: (i["created"], i["name"].lower()))
    if mode == "name_asc":
        return sorted(items, key=lambda i: (i["name"].lower(), i["created"]))
    if mode == "name_desc":
        return sorted(items, key=lambda i: (i["name"].lower(), i["created"]), reverse=True)
    return sorted(items, key=lambda i: (i["created"], i["name"].lower()), reverse=True)


def label(item: dict) -> str:
    parts = [item["name"], item["created"].strftime("%m-%d %H:%M")]
    if item["kind"] == "plan":
        parts.append("PLAN")
    elif item["kind"] == "decode":
        parts.append("DECODE")
    elif item["kind"] == "transcription":
        parts.append("TRANSCRIPTION")
    else:
        parts.append(format_seconds(item.get("duration")))
    if item.get("cot"):
        parts.append(str(item["cot"]).upper())
    cfg = item.get("cfg")
    if isinstance(cfg, (int, float)) and float(cfg) != 1.0:
        parts.append(f"cfg {float(cfg):g}")
    if item.get("truncated"):
        parts.append("TRUNCATED")
    elif item.get("status") and item["status"] != "complete":
        parts.append(str(item["status"]).upper())
    return " · ".join(str(p) for p in parts if p not in (None, "", "—"))


def summarize(items) -> str:
    songs = sum(1 for i in items if i["kind"] == "song")
    plans = sum(1 for i in items if i["kind"] == "plan")
    decodes = sum(1 for i in items if i["kind"] == "decode")
    transcriptions = sum(1 for i in items if i["kind"] == "transcription")
    total = sum(int(i.get("size") or 0) for i in items)
    bits = [f"{len(items)} item(s)"]
    for count, word in ((songs, "song"), (plans, "plan"), (decodes, "decode"),
                        (transcriptions, "transcription")):
        if count:
            bits.append(f"{count} {word}(s)")
    bits.append(format_bytes(total))
    return " · ".join(bits)


def load(root, rel: str):
    """(item, details) for a relative directory name, or (None, None)."""
    root = Path(root).resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_dir():
        return None, None
    return _item(root, path), details(path)


# ───────────────────────────────────────────────────────────── details ──
def details(path) -> dict:
    path = Path(path)
    request = _request_of(path) or {}
    result = _read_json(path / "result.json") or {}
    config = _read_json(path / "config.json") or {}
    local_env = _read_json(path / "local_env.json") or {}
    abc = _read_text(path / "score.abc") or str(request.get("abc") or "")
    audio = path / "audio.flac"
    files = []
    try:
        for entry in sorted(path.iterdir()):
            if entry.is_file():
                try:
                    files.append({"name": entry.name, "bytes": entry.stat().st_size})
                except OSError:
                    pass
    except OSError:
        pass
    return {"path": str(path), "request": request, "result": result, "config": config,
            "local_env": local_env,
            "abc": abc, "audio": str(audio) if audio.is_file() else None, "files": files}


def player_html(audio_path: str) -> str:
    src = "/gradio_api/file=" + quote(str(audio_path))
    return (
        '<div class="bb-player" data-bb-player>'
        f'<audio preload="metadata" src="{html.escape(src, quote=True)}"></audio>'
        '<div class="bb-prow">'
        '<button type="button" class="bb-pback" data-bb-back'
        ' aria-label="Restart from the beginning" title="Restart from the beginning"></button>'
        '<button type="button" class="bb-pjump" data-bb-jump="-15"'
        ' aria-label="Back 15 seconds" title="Back 15 seconds">15</button>'
        '<button type="button" class="bb-pbtn" data-bb-play'
        ' aria-label="Play" title="Play"></button>'
        '<button type="button" class="bb-pjump bb-fwd" data-bb-jump="15"'
        ' aria-label="Forward 15 seconds" title="Forward 15 seconds">15</button>'
        '<button type="button" class="bb-pend" data-bb-end'
        ' aria-label="Jump to the end" title="Jump to the end"></button>'
        '<span class="bb-ptime" data-bb-time>0:00 / 0:00</span>'
        '</div>'
        '<div class="bb-pseek" data-bb-seek>'
        '<canvas class="bb-pviz" data-bb-viz aria-hidden="true"></canvas>'
        '<div class="bb-pfill"></div>'
        '</div>'
        '</div>'
    )


def _table(title: str, rows) -> str:
    rows = [(name, v) for name, v in rows if v not in (None, "", "—", {})]
    if not rows:
        return ""
    out = [f'<div class="bb-lib-section">{html.escape(title)}</div>',
           '<table class="bb-lib-table"><tbody>']
    for name, value in rows:
        out.append(f'<tr><td>{html.escape(str(name))}</td>'
                   f'<td>{html.escape(str(value))}</td></tr>')
    out.append('</tbody></table>')
    return "".join(out)


def render_info_html(item: dict, det: dict) -> str:
    request = det.get("request") or {}
    result = det.get("result") or {}
    config = det.get("config") or {}
    local_env = det.get("local_env") or {}
    generation = config.get("generation") or {}
    abc_sampling = generation.get("abc") or {}
    sem_sampling = generation.get("semantic") or {}
    timing = result.get("timing") or {}
    semantic = timing.get("semantic") or {}

    out = ['<div class="bb-lib-card">',
           f'<div class="bb-lib-title">{html.escape(item["name"])}</div>',
           '<div class="bb-lib-sub">' + " · ".join(html.escape(str(b)) for b in (
               item.get("created_label"), format_seconds(item.get("duration")),
               item.get("kind", "").upper()) if b) + '</div>',
           f'<div class="bb-lib-sub bb-lib-dim">{html.escape(item.get("rel", ""))}'
           f' · {html.escape(format_bytes(item.get("size")))}</div>']

    chips = []
    if request.get("cot"):
        chips.append("cot " + str(request["cot"]).upper())
    if request.get("seed") is not None:
        chips.append("seed " + str(request["seed"]))
    if isinstance(request.get("cfg_scale"), (int, float)):
        chips.append(f"cfg {float(request['cfg_scale']):g}")
    if config.get("device"):
        chips.append(f"{config['device']} / {config.get('model_dtype', '')}".strip(" /"))
    if semantic.get("output_tokens"):
        chips.append(f"{int(semantic['output_tokens']):,} tok")
    if semantic.get("output_tps"):
        chips.append(f"{float(semantic['output_tps']):.1f} tok/s")
    if item.get("truncated"):
        chips.append("TRUNCATED")
    if chips:
        out.append('<div class="bb-lib-chips">' + "".join(
            f'<span class="bb-chip">{html.escape(c)}</span>' for c in chips) + '</div>')

    if det.get("audio"):
        out.append(player_html(det["audio"]))
    else:
        out.append('<div class="bb-lib-note">No audio in this directory — score-only plan.</div>')

    # The styled score below must render even while the (lazily mounted) ABC
    # accordion is closed, so the source travels with the card itself.
    if det.get("abc"):
        out.append(f'<pre id="bb-lib-abc-src" hidden>{html.escape(det["abc"])}</pre>')

    out.append(_table("request", [
        ("PLAN MODE", request.get("cot")), ("SEED", request.get("seed")),
        ("CFG SCALE", request.get("cfg_scale")), ("REQUEST ID", request.get("id")),
        ("EXTERNAL ABC", "yes" if request.get("abc") else "no"),
    ]))
    out.append(_table("sampling", [
        ("ABC temperature", abc_sampling.get("temperature")),
        ("ABC top_p", abc_sampling.get("top_p")), ("ABC top_k", abc_sampling.get("top_k")),
        ("ABC max_tokens", abc_sampling.get("max_tokens")),
        ("SEMANTIC temperature", sem_sampling.get("temperature")),
        ("SEMANTIC top_p", sem_sampling.get("top_p")), ("SEMANTIC top_k", sem_sampling.get("top_k")),
        ("SEMANTIC max_tokens", sem_sampling.get("max_tokens")),
    ]))
    ode = " · ".join(str(x) for x in (generation.get("ode_steps"), generation.get("ode_method")) if x)
    # local_env.json (a sidecar written by the web UI) records the device/dtype that
    # actually ran; upstream's config.json hardcodes model_dtype=bfloat16, so the two
    # disagree after an explicit float32 cast -- shown only when they differ.
    actual = None
    if local_env:
        bits = []
        if local_env.get("dtype") and local_env["dtype"] != config.get("model_dtype"):
            bits.append(f"{local_env['dtype']} (cast at load; config.json records "
                        f"{config.get('model_dtype') or '—'})")
        if local_env.get("device") and local_env["device"] != config.get("device"):
            bits.append(f"device {local_env['device']}")
        actual = " · ".join(bits) or None
    out.append(_table("runtime", [
        ("DEVICE", config.get("device")), ("MODEL DTYPE", config.get("model_dtype")),
        ("WEBUI ACTUAL", actual),
        ("VAE DTYPE", config.get("vae_dtype")), ("BACKEND", config.get("backend")),
        ("QUANTIZATION", config.get("quantization")), ("ODE", ode),
        ("CONTEXT", generation.get("context")), ("VAE CORE FRAMES", config.get("vae_core_frames")),
        ("VAE DECODE", config.get("vae_decode")), ("OFFLOAD AR", config.get("offload_ar")),
        ("MEMORY BUDGET (GiB)", config.get("memory_budget_gib")),
    ]))
    out.append(_table("timing", [
        ("AUDIO LENGTH", format_seconds(result.get("audio_seconds"))),
        ("SAMPLE RATE", f"{int(result['sample_rate']):,} Hz" if result.get("sample_rate") else None),
        ("SEMANTIC TOKENS", f"{int(semantic['output_tokens']):,}" if semantic.get("output_tokens") else None),
        ("SEMANTIC tok/s", f"{float(semantic['output_tps']):.2f}" if semantic.get("output_tps") else None),
        ("ABC SECONDS", (timing.get("abc") or {}).get("seconds")),
        ("NAR SECONDS", timing.get("nar_seconds")), ("VAE SECONDS", timing.get("vae_seconds")),
        ("TOTAL SECONDS", timing.get("e2e_seconds")), ("STATUS", result.get("status")),
    ]))
    provenance = []
    for key, name in (("mot", "MODEL"), ("vae", "VAE")):
        for filename, meta in (((result.get("weights") or {}).get(key) or {}).get("files") or {}).items():
            meta = meta or {}
            provenance.append((f"{name} WEIGHT",
                               f"{filename} · {format_bytes(meta.get('bytes'))} · {str(meta.get('sha256', ''))[:12]}"))
    if config.get("runtime_sha256"):
        provenance.append(("RUNTIME SHA", str(config["runtime_sha256"])[:12]))
    if config.get("validation_status"):
        provenance.append(("VALIDATION", config["validation_status"]))
    out.append(_table("provenance", provenance))

    files = det.get("files") or []
    if files:
        out.append('<div class="bb-lib-section">files</div>'
                   '<table class="bb-lib-table bb-lib-filetable"><tbody>')
        for entry in files:
            out.append(f'<tr><td>{html.escape(entry["name"])}</td>'
                       f'<td>{html.escape(format_bytes(entry["bytes"]))}</td></tr>')
        out.append('</tbody></table>')
    out.append('</div>')
    return "".join(out)


def render_empty_html(message: str = "No work selected.") -> str:
    return f'<div class="bb-lib-card"><div class="bb-lib-note">{html.escape(message)}</div></div>'


def render_multi_html(rels) -> str:
    rows = "".join(f"<li>{html.escape(str(r))}</li>" for r in rels)
    return (f'<div class="bb-lib-card"><div class="bb-lib-section">{len(rels)} works selected</div>'
            f'<ul class="bb-lib-list-plain">{rows}</ul>'
            '<div class="bb-lib-note">Select a single work to see its details.</div></div>')


def render_confirm_html(items) -> str:
    if not items:
        return ""
    rows = "".join(
        f'<li>{html.escape(i["name"])}<span class="bb-lib-dim"> · {html.escape(i["rel"])}</span></li>'
        for i in items)
    return ('<div class="bb-lib-confirm">'
            f'<div class="bb-lib-section">delete {len(items)} item(s) — this cannot be undone</div>'
            f'<ul class="bb-lib-list-plain">{rows}</ul></div>')


# ─────────────────────────────────────────────────────────── mutations ──
def _safe_rel(root: Path, rel: str) -> Path | None:
    try:
        path = (Path(root).resolve() / rel).resolve()
    except (OSError, ValueError):
        return None
    if path == Path(root).resolve() or Path(root).resolve() not in path.parents:
        return None
    return path


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^\w.\-]+", "-", name, flags=re.UNICODE)
    name = re.sub(r"-{2,}", "-", name).strip("-._")
    return name[:MAX_NAME]


def rename(root, rel: str, new_name: str):
    """Rename one work; the ``YYYYMMDD-HHMMSS-`` prefix is preserved."""
    root = Path(root).resolve()
    source = _safe_rel(root, rel)
    if source is None or not source.is_dir():
        return False, "That work no longer exists.", None
    clean = sanitize_name(new_name)
    if not clean:
        return False, "Enter a name (letters, digits, dashes, dots).", None
    match = STAMP_RE.match(source.name)
    prefix = f"{match.group(1)}-{match.group(2)}-" if match else ""
    target = source.parent / (prefix + clean)
    if target == source:
        return True, "Name unchanged.", source.relative_to(root).as_posix()
    if target.exists():
        return False, f"“{target.name}” already exists.", None
    try:
        source.rename(target)
    except OSError as exc:
        return False, f"Rename failed: {exc}", None
    return True, f"Renamed to “{clean}”.", target.relative_to(root).as_posix()


def delete(root, rels):
    """Delete the given work directories (refuses anything outside the root)."""
    root = Path(root).resolve()
    removed, errors = [], []
    for rel in rels or []:
        path = _safe_rel(root, rel)
        if path is None or path == root or not path.is_dir():
            errors.append(f"{rel}: not a work directory")
            continue
        if not any((path / name).exists() for name in ARTIFACTS):
            errors.append(f"{rel}: no known artifacts")
            continue
        try:
            shutil.rmtree(path)
            removed.append(rel)
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
    if removed and errors:
        return False, f"Deleted {len(removed)} item(s); failed: " + "; ".join(errors)
    if errors:
        return False, "Nothing deleted — " + "; ".join(errors)
    return True, f"Deleted {len(removed)} item(s)."


# ───────────────────────────────────────────────────────────────── css ──
LIBRARY_CSS = """
/* ── Library ─────────────────────────────────────────────────────────── */
#bb-lib-list { max-height: 520px; overflow: auto; border: 1px solid var(--bb-line);
  border-radius: 10px; background: var(--bb-panel); }
#bb-lib-list .wrap { display: flex; flex-direction: column; gap: 0; }
#bb-lib-list label { display: flex; align-items: flex-start; gap: 8px; width: 100%;
  margin: 0 !important; padding: 9px 10px; border: none !important;
  border-bottom: 1px solid var(--bb-line) !important; border-radius: 0 !important;
  background: transparent !important; color: var(--bb-ink2) !important; cursor: pointer; }
#bb-lib-list label:last-child { border-bottom: none !important; }
#bb-lib-list label:hover { background: var(--bb-well) !important; color: var(--bb-ink) !important; }
/* this rule carries an ID, so it beats the global label.selected chip rule:
   keep the selected row readable in both scenes */
#bb-lib-list label.selected { background: var(--bb-chip-bg) !important;
  border-bottom-color: var(--bb-chip-bg) !important; }
#bb-lib-list label.selected span { color: var(--bb-chip-fg) !important; }
#bb-lib-list label span { white-space: normal; line-height: 1.45; font-size: 11.5px;
  letter-spacing: .04em; text-transform: none; }
#bb-lib-confirm:empty { display: none; }
/* hidden bridge for the viewed work (row clicks write here) */
#bb-lib-active { display: none !important; }
/* a checked row is a deletion candidate; the viewed row gets a left accent */
#bb-lib-list label { cursor: default; }
#bb-lib-list label input[type="checkbox"] { cursor: pointer; }
#bb-lib-list label span[data-bb-row="1"] { cursor: pointer; }
#bb-lib-list label span[data-bb-row="1"]:hover { color: var(--bb-ink) !important;
  text-decoration: underline; text-underline-offset: 3px; }
#bb-lib-list label.bb-active { border-left: 2px solid var(--bb-ink) !important; }
/* compact sort chips: match the 11.5px used by the list, not Gradio's 14px */
#bb-lib-sort label, #bb-lib-order label { padding: 5px 10px !important; }
#bb-lib-sort label span, #bb-lib-order label span {
  font-size: 11.5px !important; letter-spacing: .06em !important; text-transform: uppercase; }
#bb-lib-sort, #bb-lib-order { margin-bottom: 2px; }
.bb-lib-confirm { margin-top: 10px; padding: 10px 12px; border: 1px solid var(--bb-line2);
  border-radius: 10px; background: var(--bb-panel); }
.bb-lib-list-plain { margin: 6px 0 2px; padding-left: 18px; }
.bb-lib-list-plain li { font-size: 11.5px; line-height: 1.5; color: var(--bb-ink2); overflow-wrap: anywhere; }
.bb-lib-dim { color: var(--bb-ink3) !important; }
.bb-lib-card { border: 1px solid var(--bb-line); border-radius: 10px;
  background: var(--bb-panel); padding: 14px 14px 12px; }
.bb-lib-title { font-size: 14px; letter-spacing: .06em; color: var(--bb-ink); overflow-wrap: anywhere; }
.bb-lib-sub { margin-top: 5px; font-size: 10.5px; letter-spacing: .08em;
  color: var(--bb-ink3); text-transform: uppercase; overflow-wrap: anywhere; }
.bb-lib-note { padding: 10px 2px; font-size: 11px; letter-spacing: .06em; color: var(--bb-ink3); }
.bb-lib-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 2px; }
.bb-chip { border: 1px solid var(--bb-line); border-radius: 999px; padding: 3px 9px;
  font-size: 10px; letter-spacing: .08em; color: var(--bb-ink3); text-transform: uppercase; }
.bb-lib-section { margin: 16px 0 6px; font-size: 10px; letter-spacing: .16em;
  color: var(--bb-ink3); text-transform: uppercase; }
.bb-lib-table { width: 100%; border-collapse: collapse; }
.bb-lib-table td { padding: 7px 10px; border-bottom: 1px solid var(--bb-line);
  vertical-align: top; font-size: 11.5px; }
.bb-lib-table tr:last-child td { border-bottom: none; }
.bb-lib-table td:first-child { width: 36%; color: var(--bb-ink3); text-transform: uppercase;
  letter-spacing: .1em; font-size: 10.5px; white-space: nowrap; }
.bb-lib-table td:last-child { color: var(--bb-ink2); overflow-wrap: anywhere; }
.bb-lib-filetable td:first-child { width: auto; text-transform: none; letter-spacing: .02em; }
.bb-lib-filetable td:last-child { width: 90px; text-align: right; white-space: nowrap; }
/* player */
.bb-player { display: flex; flex-direction: column; gap: 10px; margin: 12px 0 2px;
  padding: 10px 12px; border: 1px solid var(--bb-line); border-radius: 10px; background: var(--bb-well); }
.bb-prow { display: flex; align-items: center; gap: 10px; }
.bb-pbtn { position: relative; flex: 0 0 auto; width: 22px; height: 22px; padding: 0;
  border: 1px solid var(--bb-ink3); border-radius: 4px; background: var(--bb-panel);
  color: var(--bb-ink3); cursor: pointer; }
.bb-pbtn:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
.bb-pbtn:active { background: var(--bb-well); }
/* paused: CSS-drawn ▶ (no font glyphs needed) */
.bb-pbtn::after { content: ""; position: absolute; left: 8px; top: 50%; margin-top: -5px;
  border-style: solid; border-width: 5px 0 5px 7px;
  border-color: transparent transparent transparent currentColor; }
/* playing: CSS-drawn pause bars */
.bb-pbtn.bb-playing::before { content: ""; position: absolute; left: 6px; top: 50%;
  margin-top: -5px; width: 3px; height: 10px; background: currentColor; }
.bb-pbtn.bb-playing::after { left: 12px; margin-top: -5px; width: 3px; height: 10px;
  border-width: 0; background: currentColor; }
/* restart-to-zero: CSS-drawn |◀ icon (no font glyphs needed) */
.bb-pback { position: relative; flex: 0 0 auto; width: 22px; height: 22px; padding: 0;
  border: 1px solid var(--bb-ink3); border-radius: 4px; background: var(--bb-panel);
  color: var(--bb-ink3); cursor: pointer; }
.bb-pback:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
.bb-pback:active { background: var(--bb-well); }
.bb-pback::before { content: ""; position: absolute; left: 4px; top: 50%; margin-top: -4px;
  width: 2px; height: 8px; background: currentColor; }
.bb-pback::after { content: ""; position: absolute; left: 7px; top: 50%; margin-top: -4px;
  border-style: solid; border-width: 4px 5px 4px 0; border-color: transparent currentColor transparent transparent; }
/* to-the-end: mirror of |◀ → ▶| */
.bb-pend { position: relative; flex: 0 0 auto; width: 22px; height: 22px; padding: 0;
  border: 1px solid var(--bb-ink3); border-radius: 4px; background: var(--bb-panel);
  color: var(--bb-ink3); cursor: pointer; }
.bb-pend:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
.bb-pend:active { background: var(--bb-well); }
.bb-pend::before { content: ""; position: absolute; right: 4px; top: 50%; margin-top: -4px;
  width: 2px; height: 8px; background: currentColor; }
.bb-pend::after { content: ""; position: absolute; right: 7px; top: 50%; margin-top: -4px;
  border-style: solid; border-width: 4px 0 4px 5px; border-color: transparent transparent transparent currentColor; }
/* ±15s: triangle + the digits, both inside one small square-ish chip */
.bb-pjump { position: relative; flex: 0 0 auto; width: 34px; height: 22px; padding: 0 4px 0 11px;
  display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--bb-ink3); border-radius: 4px; background: var(--bb-panel);
  color: var(--bb-ink3); cursor: pointer; font: inherit; font-size: 9.5px; line-height: 1;
  letter-spacing: .02em; }
.bb-pjump:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
.bb-pjump:active { background: var(--bb-well); }
.bb-pjump::before { content: ""; position: absolute; left: 5px; top: 50%; margin-top: -4px;
  border-style: solid; border-width: 4px 5px 4px 0; border-color: transparent currentColor transparent transparent; }
.bb-pjump.bb-fwd { padding: 0 11px 0 4px; }
.bb-pjump.bb-fwd::before { left: auto; right: 5px; border-width: 4px 0 4px 5px;
  border-color: transparent transparent transparent currentColor; }
.bb-pseek { position: relative; width: 100%; height: 30px; cursor: pointer; touch-action: none; }
.bb-pseek::before { content: ""; position: absolute; left: 0; right: 0; top: 50%;
  height: 2px; margin-top: -1px; background: var(--bb-line); }
.bb-pviz { position: absolute; inset: 0; width: 100%; height: 100%; display: block;
  pointer-events: none; }
.bb-pfill { position: absolute; left: 0; top: 50%; height: 2px; margin-top: -1px;
  width: 0%; background: var(--bb-ink); }
.bb-ptime { margin-left: auto; flex: 0 0 auto; font-size: 10.5px; letter-spacing: .08em;
  color: var(--bb-ink3); font-variant-numeric: tabular-nums; }
#bb-lib-score { display: block; margin-top: 2px; border: 1px solid var(--bb-line);
  border-radius: 8px; background: var(--bb-panel); padding: 14px 10px; min-height: 96px;
  max-height: 60vh; overflow: auto; }
#bb-lib-score-inner { display: block; }
#bb-lib-score-inner svg { max-width: 100%; height: auto; }
@media (max-width: 700px) {
  .bb-prow { gap: 8px; flex-wrap: wrap; }
  .bb-ptime { flex: 1 1 100%; order: 5; margin-left: 0; text-align: right; }
  #bb-lib-list { max-height: 44vh; }
}
"""

# ────────────────────────────────────────────────────────────────── js ──
LIBRARY_JS = r"""(function () {
  function fmt(value) {
    if (!isFinite(value) || value < 0) value = 0;
    var m = Math.floor(value / 60), s = Math.floor(value % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  // One shared AudioContext for every player: creating one per selection would
  // hit the browser's context limit. Each element gets its own source+analyser.
  var BB_AUDIO_CTX = null;
  var BB_BOUND = [];
  function audioCtx() {
    if (BB_AUDIO_CTX) return BB_AUDIO_CTX;
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    try { BB_AUDIO_CTX = new AC(); } catch (error) { BB_AUDIO_CTX = null; }
    return BB_AUDIO_CTX;
  }
  function readInk() {
    return getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
  }
  function drawViz(state) {
    var canvas = state.canvas;
    if (!canvas || !canvas.clientWidth || !canvas.clientHeight) return;
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (state.inkTick++ % 90 === 0) state.ink = readInk();
    ctx.fillStyle = state.ink;
    var mid = h / 2;
    if (!state.analyser || !state.playing) {
      ctx.globalAlpha = 0.4;
      ctx.fillRect(0, mid - 0.5, w, 1);
      ctx.globalAlpha = 1;
      return;
    }
    state.analyser.getByteFrequencyData(state.data);
    var bins = state.data.length;
    var step = 6, barW = 3;
    var bars = Math.max(1, Math.floor(w / step));
    var usable = Math.max(1, Math.floor(bins * 0.85));
    ctx.globalAlpha = 0.9;
    for (var i = 0; i < bars; i++) {
      // log-ish bin mapping: give the low end more bars, like a real analyser
      var f0 = Math.floor(Math.pow(i / bars, 1.7) * usable);
      var f1 = Math.max(f0 + 1, Math.floor(Math.pow((i + 1) / bars, 1.7) * usable));
      var sum = 0, n = 0;
      for (var j = f0; j < f1 && j < bins; j++) { sum += state.data[j] || 0; n++; }
      var level = n ? (sum / n) / 255 : 0;
      // gentle high-frequency tilt so the right half keeps moving too
      level = Math.min(1, level * (0.6 + 1.3 * Math.pow(i / bars, 0.7)));
      var bh = Math.max(1.5, level * (mid - 1) * 1.25);
      ctx.fillRect(i * step, mid - bh, barW, bh * 2);
    }
    ctx.globalAlpha = 1;
  }
  var BB_VIZ_RUNNING = false;
  function vizFrame() {
    var active = 0;
    var players = document.querySelectorAll('[data-bb-player]');
    for (var i = 0; i < players.length; i++) {
      var state = players[i].__bbVizState;
      if (!state) continue;
      if (state.playing) active++;
      drawViz(state);
    }
    if (active) { requestAnimationFrame(vizFrame); }
    else { BB_VIZ_RUNNING = false; }
  }
  function startViz() {
    if (BB_VIZ_RUNNING) return;
    BB_VIZ_RUNNING = true;
    requestAnimationFrame(vizFrame);
  }
  function ensureAnalyser(state) {
    if (state.analyser) return;
    var ctx = audioCtx();
    if (!ctx) return;
    try {
      var source = ctx.createMediaElementSource(state.audio);
      var analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.82;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      state.analyser = analyser;
      state.data = new Uint8Array(analyser.frequencyBinCount);
    } catch (error) { state.analyser = null; }
  }
  function bind(player) {
    var audio = player.querySelector('audio');
    if (!audio || player.getAttribute('data-bb-bound') === '1') return;
    player.setAttribute('data-bb-bound', '1');
    var btn = player.querySelector('[data-bb-play]');
    var fill = player.querySelector('.bb-pfill');
    var time = player.querySelector('[data-bb-time]');
    var state = { audio: audio, canvas: player.querySelector('[data-bb-viz]'),
                  analyser: null, data: null, playing: false, ink: readInk(), inkTick: 0 };
    player.__bbVizState = state;
    BB_BOUND.push({ player: player, audio: audio });
    function paint() {
      if (btn) {
        var playing = !audio.paused;
        btn.classList.toggle('bb-playing', playing);
        var label = playing ? 'Pause' : 'Play';
        btn.setAttribute('aria-label', label);
        btn.title = label;
      }
      if (fill && audio.duration) {
        fill.style.width = Math.min(100, (audio.currentTime / audio.duration) * 100) + '%';
      }
      if (time) time.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
    }
    audio.addEventListener('timeupdate', paint);
    audio.addEventListener('loadedmetadata', paint);
    audio.addEventListener('play', function () { state.playing = true; startViz(); paint(); });
    audio.addEventListener('pause', function () { state.playing = false; paint(); });
    audio.addEventListener('ended', function () { state.playing = false; paint(); });
    audio.addEventListener('error', function () {
      if (time) time.textContent = 'audio unavailable';
    });
    if (btn) btn.addEventListener('click', function () {
      if (audio.paused) {
        var ctx = audioCtx();
        if (ctx && ctx.state === 'suspended') { try { ctx.resume(); } catch (error) {} }
        ensureAnalyser(state);
        var p = audio.play(); if (p && p.catch) p.catch(function () {});
      } else { audio.pause(); }
    });
    var back = player.querySelector('[data-bb-back]');
    if (back) back.addEventListener('click', function () {
      // jump back to the top; the play/pause state is left alone, so a
      // running track simply continues from the beginning
      try { audio.currentTime = 0; } catch (error) {}
      paint();
    });
    function jumpTo(target) {
      try {
        var dur = isFinite(audio.duration) ? audio.duration : 0;
        var next = target;
        if (next < 0) next = 0;
        if (dur && next > dur - 0.05) next = Math.max(0, dur - 0.05);
        audio.currentTime = next;
      } catch (error) {}
      paint();
    }
    var jumps = player.querySelectorAll('[data-bb-jump]');
    for (var j = 0; j < jumps.length; j++) {
      (function (node) {
        node.addEventListener('click', function () {
          var delta = parseFloat(node.getAttribute('data-bb-jump')) || 0;
          jumpTo(audio.currentTime + delta);
        });
      })(jumps[j]);
    }
    var end = player.querySelector('[data-bb-end]');
    if (end) end.addEventListener('click', function () {
      var dur = isFinite(audio.duration) ? audio.duration : 0;
      jumpTo(dur ? dur - 0.05 : audio.currentTime);
    });
    var seek = player.querySelector('[data-bb-seek]');
    if (seek) {
      // (name kept distinct from jumpTo: function declarations hoist,
      // and a shared name would silently replace the transport helpers)
      function seekBar(event) {
        var rect = seek.getBoundingClientRect();
        var clientX = (event.touches && event.touches[0] ? event.touches[0].clientX : event.clientX);
        if (!audio.duration || !rect.width) return;
        var ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        audio.currentTime = ratio * audio.duration;
        paint();
      }
      var dragging = false;
      seek.addEventListener('pointerdown', function (event) { dragging = true; seekBar(event); });
      seek.addEventListener('click', seekBar);
      window.addEventListener('pointerup', function () { dragging = false; });
      window.addEventListener('pointermove', function (event) { if (dragging) seekBar(event); });
    }
    paint();
    drawViz(state);   // idle baseline until playback starts
  }
  function renderScore() {
    var box = document.getElementById('bb-lib-score-inner');
    if (!box) return;
    var source = document.getElementById('bb-lib-abc-src');
    var abc = source ? (source.textContent || '') : '';
    if (!abc) {
      var wrap = document.getElementById('bb-lib-abc');
      var area = wrap ? wrap.querySelector('textarea') : null;
      abc = area ? (area.value || '') : '';
    }
    var ink = getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
    var key = ink + '|' + abc;
    var hasSvg = !!box.querySelector('svg');
    if (box.getAttribute('data-bb-key') === key && (hasSvg || !abc.trim())) return;
    box.setAttribute('data-bb-key', key);
    if (!abc.trim()) {
      box.innerHTML = '<div class="bb-score-empty">Select a work to view its score.</div>';
      return;
    }
    box.innerHTML = '';
    try {
      if (window.ABCJS && ABCJS.renderAbc) {
        var avail = Math.max(240, Math.min(900, (box.clientWidth || 340) - 18));
        ABCJS.renderAbc(box, abc, {
          responsive: 'resize', foregroundColor: ink,
          scale: avail < 520 ? 0.95 : 1.1,
          staffwidth: avail, paddingtop: 4, paddingbottom: 4,
        });
      } else {
        box.innerHTML = '<div class="bb-score-error">Score renderer not loaded.</div>';
      }
    } catch (error) {
      box.innerHTML = '<div class="bb-score-error">Could not render this ABC: '
        + String(error && error.message ? error.message : error).slice(0, 160) + '</div>';
    }
  }
  function rowRel(label) {
    var input = label ? label.querySelector('input[type="checkbox"]') : null;
    if (!input) return null;
    return input.getAttribute('name') || input.getAttribute('title') || input.value || null;
  }
  function viewRow(rel) {
    if (!rel) return;
    var box = document.getElementById('bb-lib-active');
    var area = box ? box.querySelector('textarea') : null;
    if (!area || area.value === rel) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(area, rel);
    area.dispatchEvent(new Event('input', { bubbles: true }));
    area.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function markRows() {
    var spans = document.querySelectorAll('#bb-lib-list label span');
    for (var i = 0; i < spans.length; i++) {
      var span = spans[i];
      if (span.getAttribute('data-bb-row') === '1') continue;
      span.setAttribute('data-bb-row', '1');
      span.setAttribute('role', 'button');
      span.setAttribute('tabindex', '0');
      span.setAttribute('title', 'View details');
    }
  }
  function markActive() {
    var box = document.getElementById('bb-lib-active');
    var area = box ? box.querySelector('textarea') : null;
    var rel = area ? area.value : '';
    var labels = document.querySelectorAll('#bb-lib-list label');
    for (var i = 0; i < labels.length; i++) {
      labels[i].classList.toggle('bb-active', !!rel && rowRel(labels[i]) === rel);
    }
  }
  // Clicking the row text views a work; only the checkbox itself selects it.
  document.addEventListener('click', function (event) {
    var label = event.target && event.target.closest ? event.target.closest('#bb-lib-list label') : null;
    if (!label) return;
    var input = label.querySelector('input[type="checkbox"]');
    if (!input) return;
    // direct hit on the input (mouse, keyboard or programmatic) always toggles
    var target = event.target;
    if (target === input || (target.closest && target.closest('input[type="checkbox"]'))) return;
    var box = input.getBoundingClientRect();
    var lab = label.getBoundingClientRect();
    var left = box.width ? box.left : lab.left;
    var right = box.width ? box.right : lab.left + 30;
    var top = box.height ? box.top : lab.top;
    var bottom = box.height ? box.bottom : lab.bottom;
    var pad = 6;
    var onCheckbox = event.clientX >= left - pad && event.clientX <= right + pad &&
                     event.clientY >= top - pad && event.clientY <= bottom + pad;
    if (onCheckbox) return;
    event.preventDefault();
    event.stopPropagation();
    viewRow(rowRel(label));
  }, true);
  document.addEventListener('keydown', function (event) {
    var span = event.target && event.target.closest ? event.target.closest('#bb-lib-list label span') : null;
    if (!span) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      viewRow(rowRel(span.closest('label')));
    }
  });
  function reap() {
    // a replaced details pane detaches the old <audio>; without this it can
    // keep playing (and keep its analyser graph alive) in some browsers
    for (var i = BB_BOUND.length - 1; i >= 0; i--) {
      var entry = BB_BOUND[i];
      if (!document.contains(entry.player)) {
        try { entry.audio.pause(); } catch (error) {}
        entry.audio.removeAttribute('src');
        BB_BOUND.splice(i, 1);
      }
    }
  }
  function tick() {
    var players = document.querySelectorAll('[data-bb-player]');
    for (var i = 0; i < players.length; i++) bind(players[i]);
    markRows();
    markActive();
    reap();
    renderScore();
  }
  setInterval(tick, 700);
  var pending = null;
  window.addEventListener('resize', function () {
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () {
      var box = document.getElementById('bb-lib-score-inner');
      if (box) box.removeAttribute('data-bb-key');
    }, 250);
  });
})();"""
