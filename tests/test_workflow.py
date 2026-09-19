"""``yue2_groove.workflow``: current-work identity, stage and family, no browser."""
from __future__ import annotations

import json
from pathlib import Path

from yue2_groove import workflow

ABC = ("X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\n"
       'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
       'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
       "V: Vocal\nE4E4G4G4E8z8|D4D4F4F4D8z8|\nV: Ins\nZ2|\n")


def make_run(root: Path, name: str, kind: str, *, request=None, abc=None, edit_source=None,
             with_baseline=False, finished=False, group_of=None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    request = request if request is not None else {"id": name.rsplit("-", maxsplit=1)[-1], "style": "pop",
                                                   "lyrics": "la", "cot": "full", "seed": 7}
    if kind == "group":
        (d / "run.json").write_text("{}", encoding="utf-8")
        for mode in group_of or ("full", "melody"):
            child = make_run(root, f"{name}/{mode}", "song")
            return child if False else None
    if kind in ("song", "decode", "transcription"):
        (d / "request.json").write_text(json.dumps(request), encoding="utf-8")
    if kind == "plan":
        (d / "plan.json").write_text(json.dumps({"request": request}), encoding="utf-8")
    if abc is not None:
        (d / "score.abc").write_text(abc, encoding="utf-8")
    if kind in ("song", "decode"):
        (d / "audio.flac").write_bytes(b"fLaC")
    if kind == "decode":
        (d / "decode.json").write_text("{}", encoding="utf-8")
        (d / "latent.npy").write_bytes(b"npy")
    if kind in ("song", "transcription"):
        result = {"status": "complete", "audio_seconds": 1.0}
        if kind == "song":
            result["weights"] = {"mot": {"files": {}}}
            (d / "request.json").write_text(json.dumps(request), encoding="utf-8")
        else:
            result["task"] = "melody-full"
            (d / "score.abc").write_text(abc or "", encoding="utf-8")
        (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if edit_source:
        (d / "edit_manifest.json").write_text(json.dumps(
            {"schema": "yue2-groove-edit-v1", "source": {"rel": edit_source}}), encoding="utf-8")
    if with_baseline:
        baseline = root / "baselines" / f"{name}-baseline"
        baseline.mkdir(parents=True)
        (baseline / "baseline.json").write_text(json.dumps(
            {"schema": "yue2-groove-baseline-v1", "source": {"rel": name, "path": str(d)}}),
            encoding="utf-8")
    if finished:
        (d / "finished.json").write_text("{}", encoding="utf-8")
    return d


def test_identify_kinds_and_stages(tmp_path: Path) -> None:
    make_run(tmp_path, "20260901-100000-song", "song", abc=ABC)
    make_run(tmp_path, "20260901-110000-plan", "plan", abc=ABC)
    make_run(tmp_path, "20260901-120000-tx", "transcription", abc=ABC)
    make_run(tmp_path, "20260901-130000-dec", "decode")
    make_run(tmp_path, "20260901-140000-baselined", "song", abc=ABC, with_baseline=True)
    make_run(tmp_path, "20260901-150000-edited", "song", abc=ABC, edit_source="20260901-100000-song")
    make_run(tmp_path, "20260901-160000-done", "song", abc=ABC, finished=True)

    def stage(name):
        work = workflow.identify(tmp_path, tmp_path / name)
        assert work is not None
        return work["kind"], work["stage"]

    assert stage("20260901-100000-song") == ("song", "audio")
    assert stage("20260901-110000-plan") == ("plan", "score")
    assert stage("20260901-120000-tx") == ("transcription", "score")
    assert stage("20260901-130000-dec") == ("decode", "audio")
    assert stage("20260901-140000-baselined") == ("song", "revise")   # frozen for editing
    assert stage("20260901-150000-edited") == ("song", "revise")
    assert stage("20260901-160000-done") == ("song", "done")          # explicit marker only


def test_identify_rejects_non_works(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert workflow.identify(tmp_path, tmp_path / "empty") is None
    assert workflow.identify(tmp_path, tmp_path) is None
    assert workflow.identify(tmp_path, tmp_path / "..") is None


def test_identify_group_dirs(tmp_path: Path) -> None:
    make_run(tmp_path, "20260901-170000-group/full", "song", abc=ABC)
    make_run(tmp_path, "20260901-170000-group/melody", "song", abc=ABC)
    (tmp_path / "20260901-170000-group" / "run.json").write_text("{}", encoding="utf-8")
    work = workflow.identify(tmp_path, tmp_path / "20260901-170000-group")
    assert work is not None and work["kind"] == "group" and work["stage"] == "audio"
    assert len(work["children"]) == 2


def test_band_is_one_factual_line(tmp_path: Path) -> None:
    make_run(tmp_path, "20260901-100000-Hallelujah_Machine", "song", abc=ABC)
    text = workflow.band(tmp_path, tmp_path / "20260901-100000-Hallelujah_Machine")
    assert "CURRENT" in text and "Hallelujah_Machine" in text
    assert "AUDIO" in text and "generated" in text
    assert "疑似" not in text and "listening" not in text.lower()
    assert workflow.band(tmp_path, tmp_path / "missing") == ""


def test_family_links_by_id_edit_and_cover(tmp_path: Path) -> None:
    source = make_run(tmp_path, "20260901-100000-Blue",
                      "song", abc=ABC, request={"id": "blue", "seed": 1})
    make_run(tmp_path, "20260901-110000-Blue", "song", abc=ABC, request={"id": "blue", "seed": 2})
    make_run(tmp_path, "20260901-120000-other", "song", abc=ABC, request={"id": "other"})
    edit = make_run(tmp_path, "20260901-130000-Blue-edit", "song", abc=ABC,
                    edit_source="20260901-100000-Blue")
    tx = make_run(tmp_path, "transcriptions/20260901-140000-ref", "transcription", abc=ABC)
    tx_rel = workflow.identify(tmp_path, tx)["rel"]
    cover = make_run(tmp_path, "20260901-150000-cover", "song",
                     request={"id": "cover", "abc": ABC})

    run = workflow.identify(tmp_path, source)
    entries = {entry["rel"]: entry for entry in workflow.family(tmp_path, run)}
    assert "same request id" in entries["20260901-110000-Blue"]["relations"]
    assert entries["20260901-110000-Blue"]["confidence"] == "exact"
    assert "edit source / attempt" in entries[edit.name]["relations"]
    assert "20260901-120000-other" not in entries

    # the cover run finds its transcription by the ABC digest (heuristic, one-way)
    cover_run = workflow.identify(tmp_path, cover)
    cover_entries = {entry["rel"]: entry for entry in workflow.family(tmp_path, cover_run)}
    assert "cover source" in cover_entries[tx_rel]["relations"]
    assert cover_entries[tx_rel]["confidence"] == "heuristic"


def test_next_actions_map_to_existing_handlers(tmp_path: Path) -> None:
    make_run(tmp_path, "20260901-100000-plan", "plan", abc=ABC)
    plan = workflow.identify(tmp_path, tmp_path / "20260901-100000-plan")
    assert [a["id"] for a in workflow.next_actions(plan)] == ["render", "library"]

    make_run(tmp_path, "20260901-110000-song", "song", abc=ABC,
             request={"id": "s", "seed": 10})
    song = workflow.identify(tmp_path, tmp_path / "20260901-110000-song")
    actions = workflow.next_actions(song)
    assert [a["id"] for a in actions] == ["listen", "edit", "retry", "library"]
    assert next(a for a in actions if a["id"] == "retry")["seed"] == 11

    make_run(tmp_path, "20260901-120000-tx", "transcription", abc=ABC)
    tx = workflow.identify(tmp_path, tmp_path / "20260901-120000-tx")
    assert [a["id"] for a in workflow.next_actions(tx)] == ["render", "send", "library"]
    assert workflow.next_actions({"stage": "draft", "kind": None}) == []
