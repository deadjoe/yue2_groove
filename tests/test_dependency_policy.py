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


def test_runtime_overrides_match_install_files():
    """Both uv entry points must retain our validated replacements for YuE2 pins."""
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {r.name: r for r in map(Requirement, project["project"]["dependencies"])}
    overrides = {
        r.name: r for r in map(Requirement, project["tool"]["uv"]["override-dependencies"])
    }
    assert str(deps["gradio"].specifier) == "==6.17.3"
    for name, version in (("soundfile", "0.14.0"), ("safetensors", "0.8.0")):
        expected = f"=={version}"
        assert str(deps[name].specifier) == expected
        assert str(overrides[name].specifier) == expected
        for platform in ("macos", "linux"):
            lines = (root / "overrides" / f"{platform}.txt").read_text().splitlines()
            assert f"{name}{expected}" in lines


def test_docker_sheetsage_installs_ignore_main_project_overrides():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "deploy/docker/Dockerfile").read_text()
    installs = [
        line
        for line in dockerfile.splitlines()
        if "pip install" in line and ".venv-sheetsage2" in line
    ]
    assert len(installs) == 3
    assert all(
        "uv --no-config pip install --python .venv-sheetsage2/bin/python" in line
        for line in installs
    )
    assert "torch==2.8.0+cu126 torchaudio==2.8.0+cu126" in dockerfile
    assert "import torch, torchaudio, transformers" in dockerfile
    assert "torch.__version__ == torchaudio.__version__ == '2.8.0+cu126'" in dockerfile
    assert "torchaudio.functional.resample" in dockerfile
