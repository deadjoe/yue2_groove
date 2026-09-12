"""Regression tests for ``yue2_groove.library``.

The module is stdlib-only by design (it reads saved artifacts by file convention
and never imports ``yue2``), so these tests are fast and model-free.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from yue2_groove import library as lib


def make_work(root: Path, name: str, kind: str = "song") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "request.json").write_text(json.dumps({
        "id": name, "style": "English, test tone", "lyrics": "[Verse]\nHello",
        "cot": "full", "seed": 7, "cfg_scale": 1.25}), encoding="utf-8")
    (d / "score.abc").write_text(
        'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\n'
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
        'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
        'V: Vocal\n"C"E4E4G4G4E8z8|\nV: Ins\nZ|\n', encoding="utf-8")
    if kind == "song":
        (d / "result.json").write_text(json.dumps({
            "status": "complete", "truncated": {"abc": False, "semantic": False},
            "sample_rate": 48000, "audio_seconds": 12.5,
            "timing": {"semantic": {"seconds": 2.0, "output_tokens": 25, "output_tps": 12.5},
                       "nar_seconds": 1.0, "vae_seconds": 0.2, "e2e_seconds": 3.7},
            "weights": {"mot": {"files": {"model.safetensors": {"bytes": 10, "sha256": "ab" * 32}}},
                        "vae": {"files": {"model.safetensors": {"bytes": 5, "sha256": "cd" * 32}}}}},
            ), encoding="utf-8")
        (d / "config.json").write_text(json.dumps({
            "generation": {"abc": {"temperature": .7, "max_tokens": 4096},
                           "semantic": {"temperature": 1.0, "max_tokens": 9000},
                           "ode_steps": 32, "ode_method": "midpoint", "context": 24576},
            "cot": "full", "cfg_scale": 1.25, "device": "mps", "model_dtype": "bfloat16",
            "vae_dtype": "float32", "validation_status": "ok"}), encoding="utf-8")
        (d / "audio.flac").write_bytes(b"fLaC" + b"\x00" * 64)
    return d


def test_transcription_result_is_its_own_kind(tmp_path: Path) -> None:
    """SheetSage2 outputs are result.json + score.abc without weights/audio."""
    d = tmp_path / "transcriptions" / "20260901-130000-reference"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "status": "complete", "task": "melody-vocal", "melody_only": True,
        "abc": "X:1", "warnings": [], "output_dir": str(d)}), encoding="utf-8")
    (d / "score.abc").write_text(
        'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\n'
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
        'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
        'V: Vocal\nz32|\nV: Ins\nZ|\n', encoding="utf-8")

    items = lib.scan(tmp_path)
    item = next(i for i in items if i["kind"] == "transcription")
    assert item["rel"] == "transcriptions/20260901-130000-reference"
    assert "TRANSCRIPTION" in lib.label(item)
    assert "1 transcription(s)" in lib.summarize(items)
    _item, det = lib.load(tmp_path, item["rel"])
    assert det is not None and det["abc"].startswith("X:1")


def test_comparison_bundle_is_not_scanned_as_works(tmp_path: Path) -> None:
    make_work(tmp_path, "20260901-120000-song", "song")
    bundle = tmp_path / "20260901-130000-comparison"
    (bundle / "case-001").mkdir(parents=True)
    (bundle / "case-001" / "audio.flac").write_bytes(b"fLaC")
    (bundle / "case-001" / "result.json").write_text("{}", encoding="utf-8")
    (bundle / "index.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")

    rels = {item["rel"] for item in lib.scan(tmp_path)}
    assert rels == {"20260901-120000-song"}


def test_scan_kinds_sort_and_label(tmp_path: Path) -> None:
    make_work(tmp_path, "20260901-120000-old", "song")
    make_work(tmp_path, "20260902-130000-plan", "plan")
    make_work(tmp_path, "batch-1/song-a", "song")
    (tmp_path / "not-a-work").mkdir()
    (tmp_path / "20260903-140000-decode").mkdir()
    (tmp_path / "20260903-140000-decode" / "audio.flac").write_bytes(b"fLaC")
    (tmp_path / "20260903-140000-decode" / "decode.json").write_text("{}", encoding="utf-8")

    items = lib.scan(tmp_path)
    kinds = {i["rel"]: i["kind"] for i in items}
    assert kinds == {"20260901-120000-old": "song", "20260902-130000-plan": "plan",
                     "batch-1/song-a": "song", "20260903-140000-decode": "decode"}

    # compare only entries with a timestamp prefix (unprefixed dirs, e.g. batch songs, sort by mtime)
    timed = [i["rel"] for i in lib.sort_items(items, "time_desc")
             if re.match(r"\d{8}-\d{6}-", i["rel"])]
    assert timed[0] == "20260903-140000-decode"
    # name order uses the display name (timestamp prefix removed)
    assert [i["rel"] for i in lib.sort_items(items, "name_asc")][0] == "20260903-140000-decode"
    names = [i["name"] for i in lib.sort_items(items, "name_asc")]
    assert names == sorted(names, key=str.lower)

    song = next(i for i in items if i["kind"] == "song" and i["rel"] == "20260901-120000-old")
    label = lib.label(song)
    assert "old" in label and "0:12" in label and "FULL" in label and "cfg 1.25" in label
    assert "item(s)" in lib.summarize(items)


def test_details_and_html_render(tmp_path: Path) -> None:
    make_work(tmp_path, "20260901-120000-song", "song")
    item, det = lib.load(tmp_path, "20260901-120000-song")
    assert item is not None and det is not None
    assert det["request"]["style"].startswith("English")
    assert det["abc"].startswith("X:1")
    assert det["audio"] and det["audio"].endswith("audio.flac")

    html = lib.render_info_html(item, det)
    for needle in ("bb-lib-card", "bb-player", "bb-lib-table", "bb-lib-abc-src",
                   "bfloat16", "complete", "12.5"):
        assert needle in html, needle
    assert "bb-lib-card" in lib.render_multi_html(["a", "b"])
    assert "cannot be undone" in lib.render_confirm_html([item])
    assert "nothing" in lib.render_empty_html("nothing")
    player = lib.player_html("/tmp/a b.flac")
    assert "data-bb-play" in player and "data-bb-back" in player and "data-bb-viz" in player
    assert "a%20b.flac" in player


def test_rename_guards_and_prefix(tmp_path: Path) -> None:
    make_work(tmp_path, "20260901-120000-old", "song")
    make_work(tmp_path, "20260901-120000-other", "song")

    ok, msg, rel = lib.rename(tmp_path, "20260901-120000-old", "My  New Song!")
    assert ok and rel == "20260901-120000-My-New-Song", msg

    ok, msg, rel = lib.rename(tmp_path, rel, "夜の歌")
    assert ok and rel == "20260901-120000-夜の歌", msg

    ok, msg, _ = lib.rename(tmp_path, rel, "other")
    assert not ok and "exists" in msg

    ok, msg, rel = lib.rename(tmp_path, rel, "../../evil")
    assert ok and "/" not in rel

    ok, msg, _ = lib.rename(tmp_path, "does-not-exist", "x")
    assert not ok

    assert lib.sanitize_name("  a  b  ") == "a-b"
    assert lib.sanitize_name("///") == ""
    assert lib.sanitize_name("x" * 200) == "x" * lib.MAX_NAME


def test_delete_guards_and_real_delete(tmp_path: Path) -> None:
    make_work(tmp_path, "20260901-120000-old", "song")
    make_work(tmp_path, "batch-1/song-a", "song")
    (tmp_path / "not-a-work").mkdir()

    ok, _ = lib.delete(tmp_path, ["../../etc"])
    assert not ok
    ok, _ = lib.delete(tmp_path, ["not-a-work"])
    assert not ok
    assert (tmp_path / "not-a-work").exists()

    ok, msg = lib.delete(tmp_path, ["20260901-120000-old", "batch-1/song-a"])
    assert ok, msg
    assert not (tmp_path / "20260901-120000-old").exists()
    assert not (tmp_path / "batch-1" / "song-a").exists()


def test_local_env_sidecar_reports_actual_dtype(tmp_path: Path) -> None:
    """config.json hardcodes bfloat16; the sidecar must expose an explicit fp32 cast."""
    d = make_work(tmp_path, "20260901-120000-fp32", "song")
    (d / "local_env.json").write_text(json.dumps({
        "tool": "yue2_groove", "device": "mps", "dtype": "float32",
        "torch": "2.14.0", "note": ""}), encoding="utf-8")
    item, det = lib.load(tmp_path, "20260901-120000-fp32")
    assert det["local_env"]["dtype"] == "float32"
    html = lib.render_info_html(item, det)
    assert "WEBUI ACTUAL" in html and "cast at load" in html and "bfloat16" in html

    # the same dtype as config.json -> no row at all (filtered out)
    d2 = make_work(tmp_path, "20260901-120000-bf16", "song")
    (d2 / "local_env.json").write_text(json.dumps({
        "device": "mps", "dtype": "bfloat16", "torch": "2.14.0"}), encoding="utf-8")
    item2, det2 = lib.load(tmp_path, "20260901-120000-bf16")
    assert "WEBUI ACTUAL" not in lib.render_info_html(item2, det2)
    # and a run without the sidecar still renders fine
    (d2 / "local_env.json").unlink()
    item3, det3 = lib.load(tmp_path, "20260901-120000-bf16")
    assert "WEBUI ACTUAL" not in lib.render_info_html(item3, det3)


def test_format_helpers() -> None:
    assert lib.format_seconds(0) == "—"
    assert lib.format_seconds(59.6) == "1:00"
    assert lib.format_seconds("bad") == "—"
    assert lib.format_bytes(0) == "0 B"
    assert lib.format_bytes(1536) == "1.5 KB"
    assert lib.format_bytes("bad") == "—"
    assert lib.display_name("20260901-120000-hello") == "hello"
    assert lib.display_name("plain") == "plain"


def test_scan_and_load_are_thread_safe_enough(tmp_path: Path) -> None:
    """The UI refreshes from Gradio worker threads; a scan must not explode."""
    make_work(tmp_path, "20260901-120000-a", "song")
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                lib.scan(tmp_path)
                lib.load(tmp_path, "20260901-120000-a")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
