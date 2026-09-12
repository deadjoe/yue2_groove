"""``yue2_groove.edit_flow``: baselines, invariant checks, edit manifests."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from yue2_groove import edit_flow
from yue2_groove.vendor import abc_tools

BASE_ABC = textwrap.dedent("""\
    X:1
    T:
    M:4/4
    L:1/16
    Q:1/4=88
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:C
    % verse
    V: Vocal
    "C"E2G2A2G2E2D2C4|"Am"D2E2G2E2D2C2D4|
    V: Ins
    Z2|
""")

PITCH_EDIT = BASE_ABC.replace('"C"E2G2A2G2E2D2C4', '"C"F2G2A2G2E2D2C4')
CHORD_EDIT = BASE_ABC.replace('"C"E2G2A2G2E2D2C4', '"C"E2"F"G2A2G2E2D2C4')
TEMPO_EDIT = BASE_ABC.replace("Q:1/4=88", "Q:1/4=104")


def make_work(root: Path, name: str = "20260901-120000-source") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "score.abc").write_text(BASE_ABC, encoding="utf-8")
    (directory / "request.json").write_text(json.dumps({
        "id": "source", "style": "English piano pop", "lyrics": "[Verse]\nla",
        "cot": "full", "seed": 5}), encoding="utf-8")
    (directory / "audio.flac").write_bytes(b"fLaC" + b"\x00" * 16)
    (directory / "result.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return directory


class FakeRequest:
    def __init__(self, **fields):
        self.fields = fields
        for name, value in fields.items():
            setattr(self, name, value)


# ── baseline ─────────────────────────────────────────────────────────────

def test_freeze_baseline_copies_small_artifacts_and_hashes(tmp_path: Path) -> None:
    source = make_work(tmp_path)
    original = (source / "score.abc").read_text(encoding="utf-8")

    record = edit_flow.freeze_baseline(tmp_path, source.name)

    assert record["schema"] == "yue2-groove-baseline-v1"
    assert record["source"]["rel"] == source.name
    assert record["hashes"]["score.abc"] == edit_flow.sha256_text(BASE_ABC)
    assert record["hashes"]["audio.flac"] and "latent.npy" not in record["hashes"]
    assert record["abc"].strip() == BASE_ABC.strip()

    baseline = Path(record["baseline"]["path"])
    assert (baseline / "baseline.json").is_file()
    assert (baseline / "score.abc").read_text(encoding="utf-8").strip() == BASE_ABC.strip()
    assert (baseline / "request.json").is_file()
    assert not (baseline / "audio.flac").exists()  # big artifacts stay in the source
    # the frozen source directory is untouched
    assert (source / "score.abc").read_text(encoding="utf-8") == original


def test_freeze_baseline_is_unique_and_guards_the_root(tmp_path: Path) -> None:
    make_work(tmp_path)
    first = edit_flow.freeze_baseline(tmp_path, "20260901-120000-source", now=0)
    second = edit_flow.freeze_baseline(tmp_path, "20260901-120000-source", now=0)
    assert first["baseline"]["path"] != second["baseline"]["path"]
    with pytest.raises(ValueError, match="Not a work directory"):
        edit_flow.freeze_baseline(tmp_path, "../escape")
    with pytest.raises(ValueError, match="Not a work directory"):
        edit_flow.freeze_baseline(tmp_path, ".")


# ── invariants ───────────────────────────────────────────────────────────

def test_check_invariants_matches_identical_scores() -> None:
    result = edit_flow.check_invariants(BASE_ABC, BASE_ABC)
    assert result["match"] is True and result["differences"] == []
    assert result["before_sha256"] == result["after_sha256"]
    assert result["compared_voices"] == ["Vocal", "Ins"]


def test_check_invariants_permits_chord_only_edits() -> None:
    result = edit_flow.check_invariants(BASE_ABC, CHORD_EDIT)
    assert result["match"] is True, result["differences"]
    assert result["before_sha256"] != result["after_sha256"]


def test_check_invariants_reports_pitch_changes() -> None:
    result = edit_flow.check_invariants(BASE_ABC, PITCH_EDIT)
    assert result["match"] is False
    assert any("sounding notes differ" in d for d in result["differences"])


def test_check_invariants_tempo_change_needs_the_explicit_flag() -> None:
    strict = edit_flow.check_invariants(BASE_ABC, TEMPO_EDIT)
    assert strict["match"] is False and "tempo differs" in strict["differences"][0]
    allowed = edit_flow.check_invariants(BASE_ABC, TEMPO_EDIT, allow_tempo_change=True)
    assert allowed["match"] is True and allowed["tempo_change_allowed"] is True


def test_check_invariants_can_compare_one_voice() -> None:
    instrumental_edit = BASE_ABC.replace("Z2|", "C8C8|C8C8|")
    result = edit_flow.check_invariants(BASE_ABC, instrumental_edit, voices="Vocal")
    assert result["match"] is True and result["compared_voices"] == ["Vocal"]
    full = edit_flow.check_invariants(BASE_ABC, instrumental_edit)
    assert full["match"] is False


def test_check_invariants_requires_both_scores() -> None:
    with pytest.raises(ValueError, match="required"):
        edit_flow.check_invariants("", BASE_ABC)
    with pytest.raises(abc_tools.AbcError):
        edit_flow.check_invariants(BASE_ABC, "X:1\nT:\nbroken")


# ── request / manifest ───────────────────────────────────────────────────

def test_validate_edited_abc_refuses_to_lose_edits() -> None:
    assert edit_flow.validate_edited_abc("  " + BASE_ABC) == BASE_ABC.strip()
    with pytest.raises(ValueError, match="empty"):
        edit_flow.validate_edited_abc("\n  ")
    with pytest.raises(ValueError):
        edit_flow.validate_edited_abc("not abc at all")


def test_build_edit_request_always_carries_the_abc() -> None:
    request = edit_flow.build_edit_request("jazz", "la", BASE_ABC, cot="full", seed=9,
                                           cfg_scale=1.1, id=" edit ", request_factory=FakeRequest)
    assert request.cot == "full" and request.abc.startswith("X:1")
    assert request.seed == 9 and request.cfg_scale == 1.1 and request.id == "edit"
    melody = edit_flow.build_edit_request("pop", "la", BASE_ABC, cot="melody",
                                          request_factory=FakeRequest)
    assert melody.cot == "melody" and melody.abc
    with pytest.raises(ValueError, match="cot=full"):
        edit_flow.build_edit_request("pop", "la", BASE_ABC, cot="off", request_factory=FakeRequest)
    with pytest.raises(ValueError):
        edit_flow.build_edit_request("pop", "la", "", cot="full", request_factory=FakeRequest)


def test_build_edit_manifest_records_hashes_and_permissions() -> None:
    invariants = edit_flow.check_invariants(BASE_ABC, CHORD_EDIT)
    manifest = edit_flow.build_edit_manifest(
        source_rel="20260901-120000-source", before_abc=BASE_ABC, after_abc=CHORD_EDIT,
        cot="full", seed=5, cfg_scale=None, invariants=invariants, voices="Vocal",
        allow_tempo_change=True, allow_changes=False, now=0)
    assert manifest["schema"] == "yue2-groove-edit-v1"
    assert manifest["source"]["frozen"] is False and manifest["source"]["baseline"] is None
    assert manifest["abc"]["before_sha256"] != manifest["abc"]["after_sha256"]
    assert manifest["request"] == {"cot": "full", "seed": 5, "cfg_scale": None}
    assert manifest["permitted"] == {"compared_voices": ["Vocal"], "tempo_change": True,
                                     "changes_override": False}
    assert manifest["invariants"]["match"] is True
    from datetime import datetime
    assert manifest["created"] == datetime.fromtimestamp(0).isoformat(timespec="seconds")

    frozen = edit_flow.build_edit_manifest(
        source_rel="20260901-120000-source", before_abc=BASE_ABC, after_abc=CHORD_EDIT,
        cot="full", seed=5, cfg_scale=None, invariants=invariants, voices="both",
        allow_tempo_change=False, allow_changes=False,
        baseline={"schema": "yue2-groove-baseline-v1", "baseline": {"path": "/tmp/b"}}, now=0)
    assert frozen["source"]["frozen"] is True
    assert frozen["source"]["baseline"]["baseline"]["path"] == "/tmp/b"
