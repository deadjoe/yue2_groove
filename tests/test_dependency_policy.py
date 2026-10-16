"""Keep ordinary dependency resolution from restoring the defective MPS runtime."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement


def test_macos_torch_requirement_and_uv_overrides_agree():
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(s) for s in project["project"]["dependencies"]]
    overrides = [Requirement(s) for s in project["tool"]["uv"]["override-dependencies"]]
    for platform, version in (("darwin", "2.14.1"), ("linux", "2.10.0"), ("win32", "2.10.0")):
        env = {"sys_platform": platform}
        active = [
            r
            for r in requirements
            if r.name == "torch" and (not r.marker or r.marker.evaluate(env))
        ]
        effective = [
            r for r in overrides if r.name == "torch" and (not r.marker or r.marker.evaluate(env))
        ]
        assert len(active) == len(effective) == 1
        assert version in active[0].specifier
        assert version in effective[0].specifier
        if platform == "darwin":
            assert "2.10.0" not in active[0].specifier
            assert "2.10.0" not in effective[0].specifier
    macos = (root / "overrides/macos.txt").read_text(encoding="utf-8")
    assert "torch==2.14.1" in macos.splitlines()
    assert any(r.name == "huggingface-hub" and "0.36.2" in r.specifier for r in overrides)
