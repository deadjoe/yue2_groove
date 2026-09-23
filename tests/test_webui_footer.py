"""Project metadata surfaced in the UI: version and the developer credit footer."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

yue2_groove = pytest.importorskip("yue2_groove")
webui = pytest.importorskip("yue2_groove.webui")

REPO_URL = "https://github.com/deadjoe/yue2_groove"


def test_release_version_is_consistent() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    assert match.group(1) == "1.0.5"
    assert yue2_groove.__version__ == "1.0.5"


def test_footer_credits_the_repository(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(webui.runtime, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    demo = webui.build_ui(
        {
            "device": "cpu",
            "dtype": "float32",
            "model": "m-a-p/YuE2-3B",
            "vae": "standard",
            "tab": 0,
            "status": "",
        }
    )
    footer = next(
        getattr(c, "value", "")
        for c in demo.blocks.values()
        if isinstance(getattr(c, "value", None), str) and "Developed by DEADJOE@GITHUB" in c.value
    )
    assert f"GROOVE {yue2_groove.__version__}" in footer
    assert f'href="{REPO_URL}"' in footer
    assert 'target="_blank"' in footer and 'rel="noopener"' in footer
    # only the repository name is the link, inside the parentheses
    assert ">yue2_groove</a>)" in footer
