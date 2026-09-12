"""``yue2_groove.cover``: transcription ABC → cover request, model-free."""
from __future__ import annotations

import textwrap

import pytest

from yue2_groove import cover
from yue2_groove.vendor import abc_tools

# two 4/4 bars in the native dialect, with chord symbols in the Vocal voice
CHORDED_ABC = textwrap.dedent("""\
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

MELODY_ABC = textwrap.dedent("""\
    X:1
    T:
    M:4/4
    L:1/16
    Q:1/4=88
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:C
    V: Vocal
    E2G2A2G2E2D2C4|D2E2G2E2D2C2D4|
    V: Ins
    Z2|
""")


class FakeRequest:
    def __init__(self, **fields):
        self.fields = fields
        for name, value in fields.items():
            setattr(self, name, value)


def test_cot_for_task_never_uses_off() -> None:
    assert cover.cot_for_task("melody-vocal") == "melody"
    assert cover.cot_for_task("melody-full") == "melody"
    assert cover.cot_for_task("full") == "full"
    with pytest.raises(ValueError, match="Unknown transcription task"):
        cover.cot_for_task("whatever")


def test_prepare_cover_abc_strips_chords_and_keeps_melody() -> None:
    prepared = cover.prepare_cover_abc(CHORDED_ABC)
    assert '"C"' not in prepared and '"Am"' not in prepared
    # the portable checker proves the sounding notes/meter are untouched
    check = abc_tools.compare(abc_tools.parse_abc(CHORDED_ABC), abc_tools.parse_abc(prepared))
    assert check["match"] is True, check["differences"]
    assert not any(v.chords for v in abc_tools.parse_abc(prepared).voices.values())


def test_prepare_cover_abc_can_keep_one_voice() -> None:
    prepared = cover.prepare_cover_abc(CHORDED_ABC, keep_voice="Vocal")
    check = abc_tools.compare(abc_tools.parse_abc(CHORDED_ABC), abc_tools.parse_abc(prepared),
                              names=("Vocal",))
    assert check["match"] is True
    assert abc_tools.parse_abc(prepared).voices["Ins"].notes == []


def test_prepare_cover_abc_passes_through_chord_free_scores() -> None:
    assert cover.prepare_cover_abc(MELODY_ABC) == MELODY_ABC.strip()
    with pytest.raises(ValueError, match="no ABC score"):
        cover.prepare_cover_abc("   ")
    with pytest.raises(ValueError):
        cover.prepare_cover_abc("X:1\nT:\nnot really abc")


def test_inspect_abc_reports_native_structure() -> None:
    report = cover.inspect_abc(CHORDED_ABC)
    assert report["bpm"] == 88
    assert report["voices"]["Vocal"]["sounding_notes"] == 14
    assert report["voices"]["Vocal"]["chords"][0][1] == "C"


def test_build_cover_request_carries_abc_and_melody_mode() -> None:
    request = cover.build_cover_request("English jazz", "[Verse]\nla", CHORDED_ABC,
                                        task="melody-full", seed=7, cfg_scale=1.2, id=" my-cover ",
                                        request_factory=FakeRequest)
    assert request.cot == "melody"
    assert request.abc.strip().startswith("X:1") and '"C"' not in request.abc
    assert request.style == "English jazz" and request.seed == 7
    assert request.cfg_scale == 1.2 and request.id == "my-cover"

    full = cover.build_cover_request("pop", "la", CHORDED_ABC, task="full",
                                     request_factory=FakeRequest)
    assert full.cot == "full" and '"C"' in full.abc
    assert not hasattr(full, "cfg_scale") and not hasattr(full, "id")


def test_build_cover_request_refuses_an_empty_score() -> None:
    with pytest.raises(ValueError, match="requires an ABC score"):
        cover.build_cover_request("pop", "la", "  ", request_factory=FakeRequest)


def test_cover_request_uses_the_real_yue2_factory(monkeypatch) -> None:
    """The default factory is ``adapter.song_request``; no yue2 needed for the shape."""
    seen = {}

    def fake_song_request(**fields):
        seen.update(fields)
        return FakeRequest(**fields)

    monkeypatch.setattr(cover.adapter, "song_request", fake_song_request)
    cover.build_cover_request("pop", "la", MELODY_ABC, task="melody-full")
    assert seen["cot"] == "melody" and seen["abc"].startswith("X:1")
